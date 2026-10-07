"""Macro F-0.5 exactly as defined by the challenge.

Per Source-1 entity:  F = 1.25 P R / (0.25 P + R);  an entity with no true
matches scores 1 if nothing is predicted and 0 otherwise; an entity with true
matches scores 0 if nothing is predicted.  The score is the mean over *all*
Source-1 entities of the evaluation set.
"""
from __future__ import annotations

import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series,
              by: pl.DataFrame | None = None) -> dict:
    """pred / truth: long frames with columns (s1_id, q_id).  s1_ids: all S1 ids.
    by: optional (s1_id, group) frame to also report per-group scores."""
    p = pred.select("s1_id", "q_id").unique().with_columns(pl.lit(1).alias("_p"))
    t = truth.select("s1_id", "q_id").unique().with_columns(pl.lit(1).alias("_t"))
    j = p.join(t, on=["s1_id", "q_id"], how="full", coalesce=True)
    per = j.group_by("s1_id").agg(
        pl.col("_p").sum().alias("n_pred"), pl.col("_t").sum().alias("n_true"),
        (pl.col("_p").is_not_null() & pl.col("_t").is_not_null()).sum().alias("tp"))
    base = pl.DataFrame({"s1_id": s1_ids}).join(per, on="s1_id", how="left").fill_null(0)
    P = pl.col("tp") / pl.col("n_pred")
    R = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
           .when((pl.col("n_pred") == 0) | (pl.col("n_true") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * P * R / (0.25 * P + R)))
    base = base.with_columns(f.alias("f05"))
    tot = base.select(pl.col("tp").sum(), pl.col("n_pred").sum(), pl.col("n_true").sum()).row(0)
    out = {
        "macro_f05": base["f05"].mean(),
        "singleton_acc": base.filter(pl.col("n_true") == 0)["f05"].mean(),
        "nonsingleton_f05": base.filter(pl.col("n_true") > 0)["f05"].mean(),
        "pair_precision": tot[0] / max(tot[1], 1),
        "pair_recall": tot[0] / max(tot[2], 1),
    }
    if by is not None:
        g = base.join(by, on="s1_id").group_by("group").agg(pl.col("f05").mean()).sort("group")
        out["by_group"] = dict(g.iter_rows())
    return out
