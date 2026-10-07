"""Step 4: stage-1 pruning model.

A light gradient-boosted model over retrieval scores + cheap string features
scores every blocked pair; per query we keep at most PRUNE_TOP candidates with
p1 >= PRUNE_MIN.  The surviving pairs are the candidate set fed to the final
matching model (and are what candidate_pairs.tsv reports).

Training uses 2-fold cross-fitting on the training split: queries are assigned
to a fold by the id of their true Source-1 entity (so all records of one real
business share a fold); each fold's model scores the other fold.  For the test
split the two fold models are averaged.
"""
from __future__ import annotations

import glob
import sys

import joblib
import numpy as np
import polars as pl

import models
from common import WORK, Timer, read_ground_truth
from features import STAGE1_FEATURES, context_features, load_norm, pair_features

PRUNE_TOP = 2
PRUNE_MIN = 0.01
SAMPLE_PCT = 8          # % of training queries used to fit stage 1
SUB = 3_000_000         # pairs per feature sub-chunk (memory bound)
RECALL_AT = (2, 4, 6, 8, 10)


def id_num(col: str) -> pl.Expr:
    return pl.col(col).str.split("-").list.last().cast(pl.Int64)


def train_labels(norm: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Returns (truth pairs as (q_gi, s1_gi, y=1), fold per gi)."""
    gi = norm.select("entity_id").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
    gt = read_ground_truth()
    pos = (gt.join(gi.rename({"entity_id": "q_id", "gi": "q_gi"}), on="q_id")
             .join(gi.rename({"entity_id": "s1_id", "gi": "s1_gi"}), on="s1_id"))
    # fold: parity of the true S1 id for matched queries / S1 records, own id otherwise
    f = gi.join(pos.select(pl.col("q_id").alias("entity_id"), "s1_id"), on="entity_id", how="left")
    f = f.with_columns(pl.coalesce(id_num("s1_id"), id_num("entity_id")).mod(2).cast(pl.Int8).alias("fold"))
    fold = f.sort("gi")["fold"].to_numpy()
    return pos.select("q_gi", "s1_gi", pl.lit(1, pl.Int8).alias("y")), fold


def sample_mask(col: str = "q_gi", pct: int = SAMPLE_PCT) -> pl.Expr:
    return (pl.col(col).cast(pl.Int64) * 2654435761 % 100) < pct


def featurize_chunk(cand: pl.DataFrame, norm: pl.DataFrame, full: bool = False) -> pl.DataFrame:
    cand = context_features(cand)
    parts = [pair_features(cand.slice(i, SUB), norm, full) for i in range(0, cand.height, SUB)]
    return pl.concat(parts)


def train():
    norm = load_norm("train")
    pos, fold = train_labels(norm)
    frames = []
    with Timer("stage1: build training sample"):
        for p in sorted(glob.glob(str(WORK / "train_cand" / "*.parquet"))):
            cand = pl.read_parquet(p).filter(sample_mask())
            df = featurize_chunk(cand, norm)
            df = df.join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(pl.col("y").fill_null(0))
            frames.append(df.select(["q_gi", "y"] + STAGE1_FEATURES))
            print(f"    {p.split('/')[-1]}: {df.height} pairs, {df['y'].sum()} positives", flush=True)
    data = pl.concat(frames)
    del frames
    qf = fold[data["q_gi"].to_numpy()]
    X = models.to_matrix(data, STAGE1_FEATURES)
    y = data["y"].to_numpy()
    del data
    fitted = []
    for f in (0, 1):
        with Timer(f"stage1: fit fold {f} on {int((qf == f).sum())} pairs"):
            fitted.append(models.fit(X[qf == f], y[qf == f], "stage1"))
    joblib.dump(fitted, WORK / "stage1_models.joblib")
    imp = sorted(zip(STAGE1_FEATURES, fitted[0].feature_importances_), key=lambda x: -x[1])
    print("  feature importance (fold 0):", imp[:12])


def predict_and_prune(split: str):
    """Score every blocked pair of a split with stage 1 and keep the pruned set."""
    fitted = joblib.load(WORK / "stage1_models.joblib")
    norm = load_norm(split)
    pos, fold = train_labels(norm) if split == "train" else (None, None)
    out_dir = WORK / f"{split}_pruned"
    out_dir.mkdir(exist_ok=True)
    kept = total = 0
    rec = {"blocked": 0, "kept": 0, **{k: 0 for k in RECALL_AT}}
    with Timer(f"stage1: score + prune {split}"):
        for p in sorted(glob.glob(str(WORK / f"{split}_cand" / "*.parquet"))):
            cand = pl.read_parquet(p)
            df = featurize_chunk(cand, norm)
            X = models.to_matrix(df, STAGE1_FEATURES)
            if fold is not None:
                p1 = models.predict_oof(fitted, X, fold[df["q_gi"].to_numpy()])
            else:
                p1 = models.predict(fitted, X)
            del X
            df = df.with_columns(pl.Series("p1", p1))
            df = df.with_columns(pl.col("p1").rank("ordinal", descending=True).over("q_gi").alias("_r"))
            if pos is not None:
                hit = df.join(pos, on=["q_gi", "s1_gi"]).filter(pl.col("p1") >= PRUNE_MIN)
                rec["blocked"] += df.join(pos, on=["q_gi", "s1_gi"]).height
                for k in RECALL_AT:
                    rec[k] += hit.filter(pl.col("_r") <= k).height
            df = df.filter((pl.col("_r") <= PRUNE_TOP) & (pl.col("p1") >= PRUNE_MIN)).drop("_r")
            df.write_parquet(out_dir / p.split("/")[-1])
            kept += df.height
            total += cand.height
            print(f"    {p.split('/')[-1]}: {cand.height} -> {df.height}", flush=True)
    print(f"[{split}] stage-1 kept {kept} of {total} pairs")
    if pos is not None:
        n = pos.height
        print(f"[{split}] recall of true pairs: blocking {rec['blocked'] / n:.4f}; after pruning "
              + ", ".join(f"top{k}: {rec[k] / n:.4f}" for k in RECALL_AT)
              + f"  (PRUNE_TOP={PRUNE_TOP}, PRUNE_MIN={PRUNE_MIN})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    else:
        predict_and_prune(cmd)
