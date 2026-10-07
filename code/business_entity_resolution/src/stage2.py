"""Step 5: final matching model, decoding and decision-rule selection.

  features <split> : collective + full pairwise features on the pruned candidates
  train            : 2-fold cross-fitted LightGBM on the training split
  predict <split>  : p2 for every pruned pair (out-of-fold on train, fold-average on test)
  tune             : choose the decoding rule maximising macro F0.5 on out-of-fold train scores

Decoding.  Every Source-2/3 record is first reduced to its best-scoring Source-1
candidate (records match at most one entity).  Two rules are compared:
  * threshold : keep the record if p >= t and it beats the runner-up by a margin
  * expected_f: for every Source-1 entity, order its claimants by p and keep the
                prefix that maximises the *expected* F0.5 of that entity under
                independent Bernoulli(p) outcomes (the empty prefix is worth
                the probability that the entity is a singleton).  This adapts
                the effective threshold to how many confident claimants an
                entity already has, which is what the macro metric rewards.
"""
from __future__ import annotations

import glob
import json
import sys

import joblib
import numpy as np
import polars as pl

import models
import stage1
from common import WORK, Timer
from evaluate import macro_f05
from features import STAGE2_FEATURES, collective_features, load_norm, pair_features

SUB = 1_000_000
SAMPLE_PCT = 100   # pruned set is ~11M pairs: the final model can use all of them
K_MAX = 12
ROUNDS = 2         # every member is trained for ROUNDS x its n_estimators (curve: exp_trees.py)


def build_features(split: str):
    norm = load_norm(split)
    files = sorted(glob.glob(str(WORK / f"{split}_pruned" / "*.parquet")))
    with Timer(f"stage2: collective features {split}"):
        base = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi", "p1", "q_is_s3"]) for p in files])
        coll = collective_features(base).drop("p1", "q_is_s3")
        del base
    out_dir = WORK / f"{split}_feat"
    out_dir.mkdir(exist_ok=True)
    with Timer(f"stage2: pair features {split}"):
        for p in files:
            d = pl.read_parquet(p).join(coll, on=["q_gi", "s1_gi"], how="left")
            parts = [pair_features(d.slice(i, SUB), norm, full=True, cheap=False)
                     for i in range(0, d.height, SUB)]
            pl.concat(parts).write_parquet(out_dir / p.split("/")[-1])
            print(f"    {p.split('/')[-1]}: {d.height}", flush=True)


def fit_member(X: np.ndarray, y: np.ndarray, stage: str):
    m = models.fit(X, y, stage)
    for k in range(2, ROUNDS + 1):
        m = models.fit(X, y, stage, init_model=m, seed_offset=100 * k)
    return m


def train():
    norm = load_norm("train")
    pos, fold = stage1.train_labels(norm)
    frames = []
    for p in sorted(glob.glob(str(WORK / "train_feat" / "*.parquet"))):
        d = pl.read_parquet(p, columns=["q_gi", "s1_gi"] + STAGE2_FEATURES)
        frames.append(d.filter(stage1.sample_mask(pct=SAMPLE_PCT)))
    data = pl.concat(frames).join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(
        pl.col("y").fill_null(0))
    del frames
    qf = fold[data["q_gi"].to_numpy()]
    X = models.to_matrix(data, STAGE2_FEATURES)
    y = data["y"].to_numpy()
    del data
    fitted = []
    for f in (0, 1):
        m = qf == f
        with Timer(f"stage2: fit fold {f} on {int(m.sum())} pairs ({y[m].mean():.3f} positive)"):
            fitted.append([fit_member(X[m], y[m], "stage2"), fit_member(X[m], y[m], "stage2b")])
    joblib.dump(fitted, WORK / "stage2_models.joblib")
    imp = sorted(zip(STAGE2_FEATURES, fitted[0][0].feature_importances_), key=lambda x: -x[1])
    print("  feature importance (fold 0):", imp[:15])


def predict(split: str):
    fitted = joblib.load(WORK / "stage2_models.joblib")
    fold = stage1.train_labels(load_norm(split))[1] if split == "train" else None
    cols = ["q_gi", "s1_gi"] + [c for c in STAGE2_FEATURES if c not in ("q_gi", "s1_gi")]
    outs = []
    with Timer(f"stage2: predict {split}"):
        for p in sorted(glob.glob(str(WORK / f"{split}_feat" / "*.parquet"))):
            d = pl.read_parquet(p, columns=cols)
            X = models.to_matrix(d, STAGE2_FEATURES)
            if fold is not None:
                p2 = models.predict_oof(fitted, X, fold[d["q_gi"].to_numpy()])
            else:
                p2 = models.predict(fitted, X)
            outs.append(d.select("q_gi", "s1_gi", "p1").with_columns(pl.Series("p2", p2)))
    sc = pl.concat(outs)
    sc.write_parquet(WORK / f"{split}_scores.parquet")
    return sc


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def reduce_argmax(sc: pl.DataFrame) -> pl.DataFrame:
    """One row per query: its best S1 candidate, that score, and the runner-up score."""
    sc = sc.sort(["q_gi", "p2"], descending=[False, True])
    return sc.group_by("q_gi", maintain_order=True).agg(
        pl.col("s1_gi").first(), pl.col("p2").first().alias("best"),
        pl.col("p2").slice(1, 1).first().fill_null(0.0).alias("second"))


def decode_threshold(top: pl.DataFrame, t: float, margin: float = 0.0) -> pl.DataFrame:
    return top.filter((pl.col("best") >= t) & (pl.col("best") - pl.col("second") >= margin)) \
              .select("q_gi", "s1_gi")


def decode_expected_f(top: pl.DataFrame, shift: float = 0.0, k_max: int = K_MAX) -> pl.DataFrame:
    """Per Source-1 entity, keep the prefix of claimants (sorted by p) with the highest
    expected F0.5.  `shift` is a logit offset applied to the probabilities (calibration)."""
    p = np.clip(top["best"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    if shift:
        p = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))
    df = top.select("q_gi", "s1_gi").with_columns(pl.Series("p", p))
    df = df.sort(["s1_gi", "p"], descending=[False, True])
    df = df.with_columns(pl.int_range(pl.len()).over("s1_gi").cast(pl.Int32).alias("r"))
    g = (df.group_by("s1_gi", maintain_order=True)
           .agg(pl.col("p").sum().alias("tot"), (1 - pl.col("p")).log().sum().alias("log_empty"))
           .with_row_index("si"))
    df = df.join(g.select("s1_gi", "si"), on="s1_gi")
    n = g.height
    P = np.zeros((n, k_max))
    head = df.filter(pl.col("r") < k_max)
    P[head["si"].to_numpy(), head["r"].to_numpy()] = head["p"].to_numpy()
    tot = g["tot"].to_numpy()
    best_f = np.exp(g["log_empty"].to_numpy())      # value of predicting nothing
    best_k = np.zeros(n, dtype=np.int32)
    dist = np.ones((n, 1))                           # distribution of #correct in the prefix
    cum = np.zeros(n)
    for j in range(k_max):
        pj = P[:, j]
        new = np.zeros((n, j + 2))
        new[:, :j + 1] += dist * (1 - pj)[:, None]
        new[:, 1:] += dist * pj[:, None]
        dist, cum, k = new, cum + pj, j + 1
        rest = np.maximum(tot - cum, 0.0)            # expected true matches outside the prefix
        t = np.arange(k + 1)[None, :]
        F = 1.25 * t / (0.25 * (t + rest[:, None]) + k)
        ef = (dist * F).sum(axis=1)
        better = ef > best_f
        best_f[better], best_k[better] = ef[better], k
    keep = pl.DataFrame({"si": np.arange(n, dtype=np.uint32), "k": best_k})
    return df.join(keep, on="si").filter(pl.col("r") < pl.col("k")).select("q_gi", "s1_gi")


def decode(sc: pl.DataFrame, dec: dict) -> pl.DataFrame:
    top = reduce_argmax(sc)
    if dec["method"] == "expected_f":
        return decode_expected_f(top, dec["shift"])
    return decode_threshold(top, dec["threshold"], dec["margin"])


def tune():
    norm = load_norm("train")
    ids = norm.select("entity_id", "src", "country").with_row_index("gi")
    s1 = ids.filter(pl.col("src") == 1)
    pos, _ = stage1.train_labels(norm)
    gid = ids.select(pl.col("gi").cast(pl.Int32), "entity_id")
    qmap, smap = gid.rename({"gi": "q_gi", "entity_id": "q_id"}), gid.rename({"gi": "s1_gi", "entity_id": "s1_id"})
    truth = pos.join(qmap, on="q_gi").join(smap, on="s1_gi")
    by = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("group"))
    top = reduce_argmax(pl.read_parquet(WORK / "train_scores.parquet"))
    results = []

    def score(name, params, pairs):
        a = pairs.join(qmap, on="q_gi").join(smap, on="s1_gi")
        r = macro_f05(a, truth, s1["entity_id"], by)
        print(f"  {name} {params}: F0.5={r['macro_f05']:.5f} singles={r['singleton_acc']:.4f} "
              f"non-singles={r['nonsingleton_f05']:.4f} P={r['pair_precision']:.4f} "
              f"R={r['pair_recall']:.4f} {r['by_group']}", flush=True)
        results.append((r["macro_f05"], name, params, r))

    with Timer("tune: threshold rule"):
        for t in np.arange(0.30, 0.86, 0.05):
            for mg in (0.0, 0.1, 0.2):
                score("threshold", {"threshold": round(float(t), 2), "margin": mg},
                      decode_threshold(top, float(t), mg))
    with Timer("tune: expected-F rule"):
        for shift in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
            score("expected_f", {"shift": shift}, decode_expected_f(top, shift))
    results.sort(key=lambda x: -x[0])
    best = results[0]
    print("BEST", best[1], best[2], best[0])
    dec = {"method": best[1], **best[2], "oof_macro_f05": best[0],
           "oof_by_country": best[3]["by_group"], "oof_singleton_acc": best[3]["singleton_acc"],
           "oof_nonsingleton_f05": best[3]["nonsingleton_f05"]}
    with open(WORK / "decision.json", "w") as f:
        json.dump(dec, f, indent=1)
    with open(WORK / "tune_results.json", "w") as f:
        json.dump([{"f05": r[0], "method": r[1], **r[2], "by_country": r[3]["by_group"]}
                   for r in results], f, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        build_features(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "predict":
        predict(sys.argv[2])
    elif cmd == "tune":
        tune()
