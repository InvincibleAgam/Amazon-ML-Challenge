"""Experiment: domain adaptation of the stage-2 matcher to an unlabelled country by self-training.

The current model's confident test predictions on that country (p >= POS_T as
positives, p <= NEG_T as negatives) are added to the labelled training pairs,
and the matcher is re-fitted with the same 2-fold scheme (pseudo pairs are
assigned to folds by the parity of their Source-1 id).  Records of the adapted
country are then re-scored with the adapted models; other countries keep the
base scores.  Writes work/test_scores_adapt.parquet and, for reference, the
out-of-fold macro F0.5 of the adapted models on the training split.

    python stage2_adapt.py [country=france] [pseudo_weight=1.0]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import polars as pl

import models
import stage1
from common import WORK, Timer
from evaluate import macro_f05
from features import STAGE2_FEATURES, load_norm
from stage2 import decode_threshold, reduce_argmax

POS_T, NEG_T = 0.97, 0.05


def main(country: str = "france", w_ps: float = 1.0):
    norm_tr = load_norm("train")
    pos, fold = stage1.train_labels(norm_tr)
    with Timer("adapt: load training pairs"):
        tr = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"] + STAGE2_FEATURES)
                        for p in sorted(glob.glob(str(WORK / "train_feat" / "*.parquet")))])
        tr = tr.join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(pl.col("y").fill_null(0))
        qf = fold[tr["q_gi"].to_numpy()]
        Xtr, ytr = models.to_matrix(tr, STAGE2_FEATURES), tr["y"].to_numpy()
        del tr
    with Timer(f"adapt: build pseudo-labelled pairs for {country}"):
        norm_te = load_norm("test")
        ids = norm_te.select("entity_id", "country").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
        sc = pl.read_parquet(WORK / "test_scores.parquet").select("q_gi", "s1_gi", "p2")
        te = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"] + STAGE2_FEATURES)
                        for p in sorted(glob.glob(str(WORK / "test_feat" / "*.parquet")))])
        te = te.join(sc, on=["q_gi", "s1_gi"]).join(ids.select(pl.col("gi").alias("q_gi"), "country"), on="q_gi")
        fr = te.filter(pl.col("country") == country)
        s1par = ids.select(pl.col("gi").alias("s1_gi"),
                           pl.col("entity_id").str.split("-").list.last().cast(pl.Int64).mod(2).cast(pl.Int8).alias("pf"))
        pseudo = (fr.filter((pl.col("p2") >= POS_T) | (pl.col("p2") <= NEG_T))
                    .with_columns((pl.col("p2") >= POS_T).cast(pl.Int8).alias("y")).join(s1par, on="s1_gi"))
        print(f"  {country}: {fr.height} scored pairs -> {pseudo.height} pseudo-labelled "
              f"({pseudo['y'].mean():.3f} positive); {fr.height - pseudo.height} uncertain pairs left out")
        Xps, yps, pf = models.to_matrix(pseudo, STAGE2_FEATURES), pseudo["y"].to_numpy(), pseudo["pf"].to_numpy()
        Xfr = models.to_matrix(fr, STAGE2_FEATURES)
        fr_keys = fr.select("q_gi", "s1_gi")
        del te, pseudo
    fitted = []
    for f in (0, 1):
        X = np.vstack([Xtr[qf == f], Xps[pf == f]])
        y = np.concatenate([ytr[qf == f], yps[pf == f]])
        w = np.concatenate([np.ones(int((qf == f).sum()), np.float32), np.full(int((pf == f).sum()), w_ps, np.float32)])
        with Timer(f"adapt: fit fold {f} on {len(y)} pairs ({int((pf == f).sum())} pseudo, weight {w_ps})"):
            fitted.append([models.fit(X, y, "stage2", sample_weight=w), models.fit(X, y, "stage2b", sample_weight=w)])
        del X, y, w
    # sanity: out-of-fold score on the labelled countries must not degrade
    with Timer("adapt: out-of-fold check on train"):
        p_tr = models.predict_oof(fitted, Xtr, qf)
        keys = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"])
                          for p in sorted(glob.glob(str(WORK / "train_feat" / "*.parquet")))])
        ids_tr = norm_tr.select("entity_id", "src").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
        qmap = ids_tr.select(pl.col("gi").alias("q_gi"), pl.col("entity_id").alias("q_id"))
        smap = ids_tr.select(pl.col("gi").alias("s1_gi"), pl.col("entity_id").alias("s1_id"))
        top = reduce_argmax(keys.with_columns(pl.Series("p2", p_tr)))
        a = decode_threshold(top, 0.75, 0.1).join(qmap, on="q_gi").join(smap, on="s1_gi")
        truth = pos.join(qmap, on="q_gi").join(smap, on="s1_gi")
        r = macro_f05(a, truth, ids_tr.filter(pl.col("src") == 1)["entity_id"])
        print(f"  adapted models, train OOF (threshold 0.75/0.1): F0.5={r['macro_f05']:.5f} P={r['pair_precision']:.4f} R={r['pair_recall']:.4f}")
    with Timer(f"adapt: re-score {country} on test"):
        p_fr = models.predict(fitted, Xfr)
        new = fr_keys.with_columns(pl.Series("p2_adapt", p_fr))
        out = sc.join(new, on=["q_gi", "s1_gi"], how="left").with_columns(pl.coalesce(pl.col("p2_adapt"), pl.col("p2")).alias("p2")).drop("p2_adapt")
        out.write_parquet(WORK / "test_scores_adapt.parquet")
        ch = new.join(sc, on=["q_gi", "s1_gi"])
        print(f"  {country}: mean |p2 change| {np.abs(ch['p2_adapt'].to_numpy() - ch['p2'].to_numpy()).mean():.4f}; "
              f"crossing the 0.75 threshold: {int(((ch['p2'] >= 0.75) != (ch['p2_adapt'] >= 0.75)).sum())} pairs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "france", float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
