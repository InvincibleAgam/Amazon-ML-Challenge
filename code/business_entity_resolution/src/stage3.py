"""Step 5c: correction model (stage 3).

Stage 2 compares strings; it cannot see HOW two records differ.  The generator builds its
distractors ("siblings") from a Source-1 entity with a few fixed operations: the house number
plus one of a few fixed offsets, a word of the name swapped for another name word, another
legal form.  To a string similarity a sibling looks like a true copy with a mistyped number
or word.  Stage 3 is a LightGBM on top of the stage-2 probability with features that describe
the difference itself:
  * numbers: signed difference between the numbers that do not match, digits changed, ...
  * words:   words left over, swapped in, added, mistyped (swap.describe)
  * letters: which letter of a word was replaced, added or dropped (swap.edit)
  * the record's rank among the candidates by stage-2 probability
It is cross-fitted with the same folds as stages 1 and 2.

    python stage3.py features train|test   # -> work/<split>_feat3.parquet
    python stage3.py train                 # -> work/stage3_models.joblib, work/train_scores3.parquet
    python stage3.py predict test          # -> work/test_scores3.parquet
    python stage3.py tune                  # decoding rule on p3 -> work/decision3.json
"""
from __future__ import annotations

import glob
import json
import sys
from multiprocessing import Pool

import joblib
import numpy as np
import polars as pl

import models
import stage1
import swap
from common import N_JOBS, WORK, Timer
from features import STAGE2_FEATURES

NUM3 = ["n3_q_extra", "n3_c_extra", "num_diff", "num_diff_log", "num_diff_last", "num_same_len", "num_len",
        "num_digits_changed", "num_first_digit_eq", "num_last_digit_eq", "num1_diff"]
WORD3 = ["w_typo", "w_left", "w_generic", "w_syn", "w_corr", "w_swap", "w_swap_names", "w_new", "legal_rel"]
EDIT3 = ["e_dist", "e_code", "e_code2", "e_len", "e_pos", "e_at_start", "e_at_end", "e_sound", "e_skel", "e_digit"]
CTX3 = ["p2", "p2_rank", "p2_gap", "p2_second"]
STAGE3_FEATURES = STAGE2_FEATURES + CTX3 + NUM3 + WORD3 + EDIT3
CATEGORICAL = ["e_code", "e_code2"]
PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=100, reg_lambda=1.0,
              colsample_bytree=0.6, subsample=0.8, subsample_freq=1, random_state=3)
CLIP = 50            # |number difference| above this is pooled


def _numbers(side: str) -> pl.Expr:
    return pl.col(f"{side}_nums").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def number_features(df: pl.DataFrame) -> pl.DataFrame:
    """Signed difference between the numbers that the two addresses do not share."""
    qn, cn = _numbers("q"), _numbers("c")
    df = df.with_columns(qn.list.set_difference(cn).alias("_qx"), cn.list.set_difference(qn).alias("_cx"),
                         qn.list.first().alias("_q1"), cn.list.first().alias("_c1"))
    a, b = pl.col("_qx").list.first(), pl.col("_cx").list.first()
    ai, bi = a.cast(pl.Int64, strict=False), b.cast(pl.Int64, strict=False)
    d = ai - bi
    dl = pl.col("_qx").list.last().cast(pl.Int64, strict=False) - pl.col("_cx").list.last().cast(pl.Int64, strict=False)
    d1 = pl.col("_q1").cast(pl.Int64, strict=False) - pl.col("_c1").cast(pl.Int64, strict=False)

    def clip(e: pl.Expr) -> pl.Expr:
        return pl.when(e.abs() <= CLIP).then(e).otherwise((CLIP + 10) * e.sign()).cast(pl.Float32)

    same_len = a.str.len_chars() == b.str.len_chars()
    changed = pl.sum_horizontal([(a.str.slice(i, 1) != b.str.slice(i, 1)).cast(pl.Int8) for i in range(8)])
    return df.with_columns(
        pl.col("_qx").list.len().cast(pl.Int8).alias("n3_q_extra"), pl.col("_cx").list.len().cast(pl.Int8).alias("n3_c_extra"),
        clip(d).alias("num_diff"), (d.sign() * (d.abs() + 1).log10()).cast(pl.Float32).alias("num_diff_log"),
        clip(dl).alias("num_diff_last"), same_len.cast(pl.Int8).alias("num_same_len"),
        a.str.len_chars().cast(pl.Int8).alias("num_len"),
        pl.when(same_len).then(changed).otherwise(None).cast(pl.Int8).alias("num_digits_changed"),
        (a.str.slice(0, 1) == b.str.slice(0, 1)).cast(pl.Int8).alias("num_first_digit_eq"),
        (a.str.slice(-1) == b.str.slice(-1)).cast(pl.Int8).alias("num_last_digit_eq"),
        clip(d1).alias("num1_diff"),
    ).drop("_qx", "_cx", "_q1", "_c1")


_W: dict = {}


def _init(vocab, generic, synonyms):
    _W.update(vocab=vocab, generic=generic, synonyms=synonyms)


def _describe(rows):
    v, g, s = _W["vocab"], _W["generic"], _W["synonyms"]
    return [swap.describe(qo, co, v.get(cn, {}), g.get(cn, set()), s) + swap.edit(qo, co) for qo, co, cn in rows]


def word_features(df: pl.DataFrame, words, synonyms) -> pl.DataFrame:
    q, c = pl.col("q_core").fill_null("").str.split(" "), pl.col("c_core").fill_null("").str.split(" ")
    df = df.with_columns(q.list.set_difference(c).alias("_qo"), c.list.set_difference(q).alias("_co")).with_row_index("_i")
    todo = df.filter((pl.col("_qo").list.len() > 0) | (pl.col("_co").list.len() > 0))
    rows = list(todo.select("_qo", "_co", "country").iter_rows())
    chunks = [rows[i:i + 50_000] for i in range(0, len(rows), 50_000)]
    with Pool(N_JOBS, initializer=_init, initargs=(words[0], words[1], synonyms)) as pool:
        out = [r for part in pool.map(_describe, chunks) for r in part]
    names = WORD3[:8] + EDIT3
    res = pl.DataFrame(out, schema=names, orient="row").with_columns(pl.all().cast(pl.Int32)).with_columns(todo["_i"])
    ql, cl = pl.col("q_legal").fill_null(""), pl.col("c_legal").fill_null("")
    df = (df.join(res, on="_i", how="left").with_columns(pl.col(names).fill_null(0))
            .with_columns((pl.col("w_swap_names") + 1).log10().cast(pl.Float32).alias("w_swap_names"),
                          pl.when((ql == "") & (cl == "")).then(0).when(ql == "").then(1).when(cl == "").then(2)
                            .when(ql == cl).then(3).otherwise(4).cast(pl.Int8).alias("legal_rel")))
    return df.drop("_qo", "_co", "_i")


def context_features(sc: pl.DataFrame) -> pl.DataFrame:
    sc = sc.with_columns(pl.col("p2").rank("ordinal", descending=True).over("q_gi").cast(pl.Int8).alias("p2_rank"),
                         pl.col("p2").max().over("q_gi").alias("_best"))
    second = sc.filter(pl.col("p2_rank") == 2).select("q_gi", pl.col("p2").alias("p2_second"))
    sc = sc.join(second, on="q_gi", how="left").with_columns(pl.col("p2_second").fill_null(0.0))
    return sc.with_columns((pl.col("p2") - pl.when(pl.col("p2_rank") == 1).then(pl.col("p2_second"))
                            .otherwise(pl.col("_best"))).alias("p2_gap")).drop("_best")


def build(split: str):
    norm = pl.read_parquet(WORK / f"{split}_norm.parquet", columns=swap.NORM_COLS + ["nums"])
    words, synonyms = swap.country_words(norm), swap.load_synonyms()
    rec = norm.select("country", "core", "legal", "nums").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
    del norm
    sc = context_features(pl.read_parquet(WORK / f"{split}_scores.parquet").select("q_gi", "s1_gi", "p2"))
    q = rec.select(pl.col("gi").alias("q_gi"), pl.col("core").alias("q_core"), pl.col("legal").alias("q_legal"),
                   pl.col("nums").alias("q_nums"))
    c = rec.select(pl.col("gi").alias("s1_gi"), pl.col("core").alias("c_core"), pl.col("legal").alias("c_legal"),
                   pl.col("nums").alias("c_nums"), "country")
    with Timer(f"stage3: correction features {split} ({sc.height} pairs)"):
        df = sc.join(q, on="q_gi", how="left").join(c, on="s1_gi", how="left")
        df = word_features(number_features(df), words, synonyms)
        df.select(["q_gi", "s1_gi"] + CTX3 + NUM3 + WORD3 + EDIT3).write_parquet(WORK / f"{split}_feat3.parquet")


def load(split: str) -> pl.DataFrame:
    base = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"] + STAGE2_FEATURES)
                      for p in sorted(glob.glob(str(WORK / f"{split}_feat" / "*.parquet")))])
    return base.join(pl.read_parquet(WORK / f"{split}_feat3.parquet"), on=["q_gi", "s1_gi"], how="left")


def train():
    import lightgbm as lgb
    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id"])
    pos, fold = stage1.train_labels(norm)
    del norm
    data = load("train").join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(pl.col("y").fill_null(0))
    keys = data.select("q_gi", "s1_gi", "p1")
    qf = fold[data["q_gi"].to_numpy()]
    y = data["y"].to_numpy()
    X = models.to_matrix(data, STAGE3_FEATURES)
    del data
    fitted, p3 = [], np.zeros(len(y), dtype=np.float32)
    for f in (0, 1):
        m = qf == f
        with Timer(f"stage3: fit fold {f} on {int(m.sum())} pairs"):
            clf = lgb.LGBMClassifier(**PARAMS, n_jobs=N_JOBS, verbose=-1).fit(
                X[m], y[m], feature_name=STAGE3_FEATURES, categorical_feature=CATEGORICAL)
        fitted.append(clf)
    for f in (0, 1):                                    # out-of-fold: the other fold's model
        m = qf == f
        p3[m] = models.predict(fitted[1 - f], X[m])
    joblib.dump(fitted, WORK / "stage3_models.joblib")
    keys.with_columns(pl.Series("p2", p3)).write_parquet(WORK / "train_scores3.parquet")
    imp = sorted(zip(STAGE3_FEATURES, fitted[0].feature_importances_), key=lambda x: -x[1])
    print("  most used features (fold 0):", [(k, int(v)) for k, v in imp[:25]])
    print("  use of the new features:", [(k, int(v)) for k, v in imp if k in NUM3 + WORD3 + CTX3 + EDIT3])


def predict(split: str):
    fitted = joblib.load(WORK / "stage3_models.joblib")
    data = load(split)
    with Timer(f"stage3: predict {split}"):
        p3 = models.predict(fitted, models.to_matrix(data, STAGE3_FEATURES))
    data.select("q_gi", "s1_gi", "p1").with_columns(pl.Series("p2", p3)).write_parquet(WORK / f"{split}_scores3.parquet")


def tune():
    """Decoding rule on the corrected probability, with and without the word-swap veto, under the
    training prior and the simulated test prior."""
    from exp_trees import Evaluator
    from stage2 import decode_threshold, reduce_argmax

    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id"] + swap.NORM_COLS)
    pos, _ = stage1.train_labels(norm)
    ev = Evaluator(norm, pos)
    words, synonyms = swap.country_words(norm), swap.load_synonyms()
    results = []
    for name, path in (("stage 2", "train_scores.parquet"), ("stage 3", "train_scores3.parquet")):
        sc = pl.read_parquet(WORK / path).select("q_gi", "s1_gi", "p2")
        top = reduce_argmax(sc)
        flagged = swap.flag(top.select("q_gi", "s1_gi"), norm, words, synonyms)
        vetoed = top.join(flagged.filter(pl.col("swap")).select("q_gi", "s1_gi"), on=["q_gi", "s1_gi"], how="anti")
        for veto, tp in ((False, top), (True, vetoed)):
            for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
                for mg in (0.0, 0.1):
                    fn = lambda x, t=t, mg=mg: decode_threshold(x, t, mg)   # noqa: E731
                    a = ev.train_prior(tp, "threshold", {"threshold": t, "margin": mg})
                    b = ev.test_prior(tp, "threshold", {"threshold": t, "margin": mg})
                    results.append({"scores": name, "veto": veto, "threshold": t, "margin": mg,
                                    "train_prior": a["macro_f05"], "test_prior": b["macro_f05"],
                                    "P": b["pair_precision"], "R": b["pair_recall"], "by_country": b["by_group"]})
                    print(f"  {name} veto={veto} t={t:.2f} m={mg}: train prior {a['macro_f05']:.5f}  "
                          f"test prior {b['macro_f05']:.5f}  P {b['pair_precision']:.4f} R {b['pair_recall']:.4f} "
                          f"{b['by_group']}", flush=True)
    best = max((r for r in results if r["scores"] == "stage 3"), key=lambda r: r["test_prior"])
    print("BEST stage 3 under the test-like prior:", best)
    json.dump({"method": "threshold", **best}, open(WORK / "decision3.json", "w"), indent=1)
    json.dump(results, open(WORK / "tune3_results.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        build(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "predict":
        predict(sys.argv[2])
    elif cmd == "tune":
        tune()
