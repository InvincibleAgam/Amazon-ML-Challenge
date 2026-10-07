"""Experiment: does the stage-2 matcher gain from more boosting rounds?

LightGBM has no epochs; the equivalent is the number of trees.  The curve up to
the current setting (1x = 1000 / 1500 trees for the two ensemble members) is
read off the fitted models; beyond it the models are trained further, one
multiple at a time.  Every point is scored out-of-fold with the exact metric,
under the training prior and under the simulated test prior of tune_prior.py.

    python exp_trees.py [max_multiple]        # default 2  (= 2000 / 3000 trees)

Writes only to work/exp_trees/; the pipeline's models and decisions are untouched.
"""
from __future__ import annotations

import glob
import sys

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

import models
import stage1
from common import N_JOBS, WORK, Timer
from evaluate import macro_f05
from features import STAGE2_FEATURES, load_norm
from stage2 import decode_expected_f, decode_threshold, reduce_argmax

OUT = WORK / "exp_trees"
STAGES = ("stage2", "stage2b")
BASE = [models.PARAMS[s]["n_estimators"] for s in STAGES]     # trees per member at 1x
TARGET_SHARE = 0.40
TRAIN_RULES = [("expected_f", {"shift": 0.0})] + \
              [("threshold", {"threshold": t, "margin": 0.1}) for t in (0.60, 0.65, 0.70)]
TEST_RULES = [("threshold", {"threshold": t, "margin": 0.1}) for t in (0.65, 0.70, 0.75, 0.80)]


def apply_rule(top: pl.DataFrame, method: str, params: dict) -> pl.DataFrame:
    if method == "expected_f":
        return decode_expected_f(top, params["shift"])
    return decode_threshold(top, params["threshold"], params["margin"])


class Evaluator:
    """Exact macro F0.5 under the training prior and under the simulated test prior."""

    def __init__(self, norm: pl.DataFrame, pos: pl.DataFrame):
        ids = norm.select("entity_id", "src", "country").with_row_index("gi")
        self.s1 = ids.filter(pl.col("src") == 1)
        gid = ids.select(pl.col("gi").cast(pl.Int32), "entity_id")
        self.qmap = gid.rename({"gi": "q_gi", "entity_id": "q_id"})
        self.smap = gid.rename({"gi": "s1_gi", "entity_id": "s1_id"})
        self.truth = pos.join(self.qmap, on="q_gi").join(self.smap, on="s1_gi")
        self.by = self.s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("group"))
        self.matched_q = pos.select("q_gi").unique().with_columns(pl.lit(1, pl.Int8).alias("m"))
        n_all_q = norm.filter(pl.col("src") != 1).height
        n_matched = self.matched_q.height
        self.w = TARGET_SHARE / (1 - TARGET_SHARE) * n_matched / (n_all_q - n_matched)

    def train_prior(self, top, method, params) -> dict:
        a = apply_rule(top, method, params).join(self.qmap, on="q_gi").join(self.smap, on="s1_gi")
        return macro_f05(a, self.truth, self.s1["entity_id"], self.by)

    def test_prior(self, top, method, params) -> dict:
        # same simulation as tune_prior.py: distractor records repeated up to the target share
        k_int = int(np.floor(self.w))
        frac = self.w - k_int
        t = top.join(self.matched_q, on="q_gi", how="left").with_columns(pl.col("m").fill_null(0))
        d = t.filter(pl.col("m") == 0)
        extra = d.filter((pl.col("q_gi").cast(pl.Int64) * 2654435761 % 1000) < frac * 1000)
        sim = pl.concat([t] + [d] * (k_int - 1) + [extra]).drop("m")
        sim = sim.with_columns(pl.int_range(pl.len()).cast(pl.Int32).alias("row"))
        sim = sim.with_columns((pl.col("q_gi").cast(pl.Int64)
                                + pl.col("row").cast(pl.Int64) * 100_000_000).alias("q_key"))
        n_orig = t.height
        sim_top = sim.select(pl.col("q_key").alias("q_gi"), "s1_gi", "best", "second")
        a = (apply_rule(sim_top, method, params)
             .join(sim.select("q_key", pl.col("q_gi").alias("q_real"), "row"), left_on="q_gi", right_on="q_key")
             .select(pl.col("q_real").alias("q_gi"), "s1_gi", "row")
             .join(self.qmap, on="q_gi").join(self.smap, on="s1_gi")
             .with_columns(pl.when(pl.col("row") < n_orig).then(pl.col("q_id"))
                             .otherwise(pl.col("q_id") + "#" + pl.col("row").cast(pl.String)).alias("q_id")))
        return macro_f05(a, self.truth, self.s1["entity_id"], self.by)


def booster(m) -> lgb.Booster:
    return m.booster_ if hasattr(m, "booster_") else m


def add_segment(fitted, X, qf, raw, lo: float, hi: float):
    """Adds the trees of the multiple range (lo, hi] to the running out-of-fold raw scores."""
    for f in (0, 1):
        idx = np.flatnonzero(qf == f)
        xs = X[idx]
        for i, m in enumerate(fitted[1 - f]):
            s, e = int(round(lo * BASE[i])), int(round(hi * BASE[i]))
            raw[i][idx] += booster(m).predict(xs, start_iteration=s, num_iteration=e - s,
                                              raw_score=True, num_threads=N_JOBS)
        del xs


def extend(fitted, X, y, qf, mult: int):
    """Trains every member for one more multiple of its base number of trees."""
    out = []
    for f in (0, 1):
        m = qf == f
        Xf, yf = X[m], y[m]
        members = []
        for i, stage in enumerate(STAGES):
            params = {"random_state": 0, **models.PARAMS[stage]}
            params["random_state"] += 100 * mult      # fresh row / column samples for the new trees
            clf = lgb.LGBMClassifier(**params, n_jobs=N_JOBS, verbose=-1)
            with Timer(f"exp: fold {f} {stage} -> {mult}x"):
                clf.fit(Xf, yf, init_model=booster(fitted[f][i]))
            print(f"    trees now: {clf.booster_.num_trees()}", flush=True)
            members.append(clf)
        out.append(members)
        del Xf, yf
    return out


def main(max_mult: int = 2):
    OUT.mkdir(exist_ok=True)
    norm = load_norm("train")
    pos, fold = stage1.train_labels(norm)
    ev = Evaluator(norm, pos)
    del norm
    with Timer("exp: load stage-2 features"):
        frames = [pl.read_parquet(p, columns=["q_gi", "s1_gi"] + STAGE2_FEATURES)
                  for p in sorted(glob.glob(str(WORK / "train_feat" / "*.parquet")))]
        data = pl.concat(frames).join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(
            pl.col("y").fill_null(0))
        del frames
        keys = data.select("q_gi", "s1_gi")
        qf = fold[data["q_gi"].to_numpy()]
        X = models.to_matrix(data, STAGE2_FEATURES)
        y = data["y"].to_numpy()
        del data
    print(f"  {len(y)} pairs, {y.mean():.3f} positive; base trees {BASE}", flush=True)

    raw = [np.zeros(len(y), dtype=np.float64) for _ in STAGES]
    rows = []

    def checkpoint(mult: float):
        p2 = np.mean([1.0 / (1.0 + np.exp(-r)) for r in raw], axis=0).astype(np.float32)
        sc = keys.with_columns(pl.Series("p2", p2))
        sc.write_parquet(OUT / f"train_scores_{mult:.2f}x.parquet")
        if abs(mult - 1.0) < 1e-9:       # must reproduce the pipeline's own out-of-fold scores
            ref = pl.read_parquet(WORK / "train_scores.parquet").select("q_gi", "s1_gi", pl.col("p2").alias("ref"))
            diff = sc.join(ref, on=["q_gi", "s1_gi"]).select((pl.col("p2") - pl.col("ref")).abs().max()).item()
            print(f"  check at 1x: max |p2 - pipeline p2| = {diff:.2e}", flush=True)
        top = reduce_argmax(sc)
        row = {"multiple": mult, "trees": "/".join(str(int(round(mult * b))) for b in BASE),
               "logloss": log_loss(y, np.clip(p2.astype(np.float64), 1e-7, 1 - 1e-7)),
               "auc": roc_auc_score(y, p2)}
        for method, params in TRAIN_RULES:
            r = ev.train_prior(top, method, params)
            name = "train_ef" if method == "expected_f" else f"train_t{params['threshold']:.2f}"
            row[name] = r["macro_f05"]
        for method, params in TEST_RULES:
            r = ev.test_prior(top, method, params)
            row[f"test_t{params['threshold']:.2f}"] = r["macro_f05"]
            if params["threshold"] == 0.70:
                row["test_t0.70_P"], row["test_t0.70_R"] = r["pair_precision"], r["pair_recall"]
        rows.append(row)
        print("  POINT " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in row.items()), flush=True)
        pl.DataFrame(rows).write_csv(OUT / "curve.csv")

    fitted = joblib.load(WORK / "stage2_models.joblib")
    lo = 0.0
    for hi in (0.25, 0.5, 0.75, 1.0):
        with Timer(f"exp: score trees up to {hi}x"):
            add_segment(fitted, X, qf, raw, lo, hi)
        checkpoint(hi)
        lo = hi

    for mult in range(2, max_mult + 1):
        fitted = extend(fitted, X, y, qf, mult)
        joblib.dump(fitted, OUT / f"stage2_models_{mult}x.joblib")
        for hi in (mult - 0.5, float(mult)):
            with Timer(f"exp: score trees up to {hi}x"):
                add_segment(fitted, X, qf, raw, lo, hi)
            checkpoint(hi)
            lo = hi

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    pl.Config.set_float_precision(5)
    print("\n== learning curve (out-of-fold, full training split)")
    print(pl.DataFrame(rows))
    print("EXP DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
