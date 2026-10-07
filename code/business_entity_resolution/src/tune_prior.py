"""Re-tune the decoding rule under a test-like class prior.

The test split holds more unmatched (distractor) records than train (estimated
~40 % vs 26 %: see the stage-1 confidence statistics and the records-per-entity
ratio).  Under label shift the pair probabilities learned on train are too
optimistic on test.  We simulate the test prior on the *training* out-of-fold
scores by up-weighting distractor queries (repeating them), then pick the
decoding parameters that maximise macro F0.5 under that prior.

    python tune_prior.py [target_distractor_share]      # default 0.40

Writes work/decision_testprior.json; `make_submission.py` uses it when present
and the environment variable ER_DECISION=testprior is set (or when copied over
work/decision.json).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

import stage1
from common import WORK, Timer
from evaluate import macro_f05
from features import load_norm
from stage2 import decode_expected_f, decode_threshold, reduce_argmax


def main(target_share: float = 0.40):
    norm = load_norm("train")
    ids = norm.select("entity_id", "src", "country").with_row_index("gi")
    s1 = ids.filter(pl.col("src") == 1)
    pos, _ = stage1.train_labels(norm)
    gid = ids.select(pl.col("gi").cast(pl.Int32), "entity_id")
    qmap = gid.rename({"gi": "q_gi", "entity_id": "q_id"})
    smap = gid.rename({"gi": "s1_gi", "entity_id": "s1_id"})
    truth = pos.join(qmap, on="q_gi").join(smap, on="s1_gi")
    by = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("group"))

    top = reduce_argmax(pl.read_parquet(WORK / "train_scores.parquet"))
    matched_q = pos.select("q_gi").unique().with_columns(pl.lit(1, pl.Int8).alias("m"))
    top = top.join(matched_q, on="q_gi", how="left").with_columns(pl.col("m").fill_null(0))
    n_all_q = norm.filter(pl.col("src") != 1).height
    n_matched = matched_q.height
    n_distr = n_all_q - n_matched
    # distractor queries need weight w so that w*D / (M + w*D) = target_share
    w = target_share / (1 - target_share) * n_matched / n_distr
    print(f"train: {n_matched} matched queries, {n_distr} distractors ({n_distr / n_all_q:.3f}); "
          f"target share {target_share:.2f} -> distractor weight {w:.2f}")
    # integer + fractional repetition, deterministic by hash
    k_int = int(np.floor(w))
    frac = w - k_int
    d = top.filter(pl.col("m") == 0)
    extra = d.filter((pl.col("q_gi").cast(pl.Int64) * 2654435761 % 1000) < frac * 1000)
    sim = pl.concat([top] + [d] * (k_int - 1) + [extra]).drop("m")
    # duplicated queries must stay distinct rows for the decoders: give them fresh ids
    sim = sim.with_columns(pl.int_range(pl.len()).cast(pl.Int32).alias("row"))
    sim = sim.with_columns((pl.col("q_gi").cast(pl.Int64) + pl.col("row").cast(pl.Int64) * 100_000_000).alias("q_key"))
    print(f"simulated set: {sim.height} queries, distractor share "
          f"{1 - n_matched / sim.height:.3f}")

    n_orig = top.height

    def score(name, params, pairs):
        # duplicated distractors are *distinct* extra records: give the copies their own ids so
        # each false merge they produce is counted (the originals keep their real id)
        a = (pairs.join(sim.select("q_key", pl.col("q_gi").alias("q_real"), "row"), left_on="q_gi", right_on="q_key")
                  .select(pl.col("q_real").alias("q_gi"), "s1_gi", "row")
                  .join(qmap, on="q_gi").join(smap, on="s1_gi")
                  .with_columns(pl.when(pl.col("row") < n_orig).then(pl.col("q_id"))
                                  .otherwise(pl.col("q_id") + "#" + pl.col("row").cast(pl.String)).alias("q_id")))
        r = macro_f05(a, truth, s1["entity_id"], by)
        print(f"  {name} {params}: F0.5={r['macro_f05']:.5f} singles={r['singleton_acc']:.4f} "
              f"non-singles={r['nonsingleton_f05']:.4f} P={r['pair_precision']:.4f} "
              f"R={r['pair_recall']:.4f} {r['by_group']}", flush=True)
        return r

    sim_top = sim.select(pl.col("q_key").alias("q_gi"), "s1_gi", "best", "second")
    results = []
    with Timer("tune under test-like prior: threshold rule"):
        for t in np.arange(0.40, 0.91, 0.05):
            for mg in (0.0, 0.1):
                r = score("threshold", {"threshold": round(float(t), 2), "margin": mg},
                          decode_threshold(sim_top, float(t), mg))
                results.append((r["macro_f05"], "threshold", {"threshold": round(float(t), 2), "margin": mg}, r))
    with Timer("tune under test-like prior: expected-F rule"):
        for shift in (-2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
            r = score("expected_f", {"shift": shift}, decode_expected_f(sim_top, shift))
            results.append((r["macro_f05"], "expected_f", {"shift": shift}, r))
    results.sort(key=lambda x: -x[0])
    best = results[0]
    print("BEST under test-like prior:", best[1], best[2], best[0])
    dec = {"method": best[1], **best[2], "sim_macro_f05": best[0], "target_distractor_share": target_share,
           "sim_by_country": best[3]["by_group"]}
    with open(WORK / "decision_testprior.json", "w") as f:
        json.dump(dec, f, indent=1)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.40)
