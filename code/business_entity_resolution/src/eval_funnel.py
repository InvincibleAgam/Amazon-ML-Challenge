"""Diagnostic: precision / recall of every pipeline module on the training split.

Traces each of the 7.64 M labelled pairs through the pipeline and reports, per
module, how many true pairs are still alive (recall) and what share of the
pairs kept are true (precision):

    A  retrieval (blocking)            work/train_cand/
    B  stage-1 pruning                 work/train_pruned/
    C  stage-2 matcher + decoding      work/train_scores.parquet + work/decision_testprior.json
    D  stage-3 correction + decoding   work/train_scores3.parquet + work/decision3.json + word-swap veto

All scores are out-of-fold, so no model scores data it was trained on.  The
breakdowns (country, source, address presence, script of the name) refer to D,
the submitted output.

    python eval_funnel.py
"""
from __future__ import annotations

import glob
import json

import polars as pl

import stage1
import swap
from common import WORK
from features import load_norm
from stage2 import decode

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_tbl_cols(20)


def main():
    norm = load_norm("train")
    seg = (norm.select("country", "src", "addr_empty", "name_native").with_row_index("q_gi")
               .with_columns(pl.col("q_gi").cast(pl.Int32)))
    pos, _ = stage1.train_labels(norm)
    key = pos.select("q_gi", "s1_gi")
    truth = key.join(seg, on="q_gi")
    queries = seg.filter(pl.col("src") != 1)

    hit_a, n_a = [], 0
    for p in sorted(glob.glob(str(WORK / "train_cand" / "*.parquet"))):
        c = pl.read_parquet(p, columns=["q_gi", "s1_gi"])
        n_a += c.height
        hit_a.append(c.join(key, on=["q_gi", "s1_gi"]))
    hit_a = pl.concat(hit_a).with_columns(pl.lit(1, pl.Int8).alias("A"))

    pr = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"])
                    for p in glob.glob(str(WORK / "train_pruned" / "*.parquet"))])
    n_b = pr.height
    hit_b = pr.join(key, on=["q_gi", "s1_gi"]).with_columns(pl.lit(1, pl.Int8).alias("B"))

    rule2 = json.load(open(WORK / "decision_testprior.json"))
    dec2 = decode(pl.read_parquet(WORK / "train_scores.parquet").select("q_gi", "s1_gi", "p2"), rule2)
    n_c = dec2.height
    hit_c = dec2.join(key, on=["q_gi", "s1_gi"]).with_columns(pl.lit(1, pl.Int8).alias("C"))
    rule = json.load(open(WORK / "decision3.json"))
    dec = decode(pl.read_parquet(WORK / "train_scores3.parquet").select("q_gi", "s1_gi", "p2"), rule)
    dec = swap.flag(dec, norm).filter(~pl.col("swap")).select("q_gi", "s1_gi")
    n_d = dec.height
    hit_d = dec.join(key, on=["q_gi", "s1_gi"]).with_columns(pl.lit(1, pl.Int8).alias("D"))

    t = (truth.join(hit_a, on=["q_gi", "s1_gi"], how="left").join(hit_b, on=["q_gi", "s1_gi"], how="left")
              .join(hit_c, on=["q_gi", "s1_gi"], how="left").join(hit_d, on=["q_gi", "s1_gi"], how="left")
              .with_columns(pl.col("A", "B", "C", "D").fill_null(0)))
    n = t.height
    a, b, c, d = int(t["A"].sum()), int(t["B"].sum()), int(t["C"].sum()), int(t["D"].sum())
    print(f"true pairs: {n:,}   records (S2+S3): {queries.height:,}   decoding rule: {rule}\n")
    print("== pipeline funnel (C and D both start from B)")
    print(pl.DataFrame({
        "module": ["A retrieval (blocking)", "B stage-1 pruning", "C stage-2 matcher + decoding",
                   "D stage-3 correction + decoding + veto"],
        "pairs_kept": [n_a, n_b, n_c, n_d],
        "true_pairs_kept": [a, b, c, d],
        "recall_%": [round(100 * x / n, 2) for x in (a, b, c, d)],
        "precision_%": [round(100 * a / n_a, 2), round(100 * b / n_b, 2), round(100 * c / n_c, 2),
                        round(100 * d / n_d, 2)],
        "true_pairs_lost_here": [n - a, a - b, b - c, b - d],
    }))

    fp = (dec.join(key.with_columns(pl.lit(1).alias("y")), on=["q_gi", "s1_gi"], how="left")
             .filter(pl.col("y").is_null()).join(seg, on="q_gi"))

    def table(keys, title):
        g = t.group_by(keys).agg(
            pl.len().alias("true_pairs"),
            (100 * pl.col("A").mean()).round(2).alias("A_retrieval_recall%"),
            (100 * pl.col("B").mean()).round(2).alias("B_pruning_recall%"),
            (100 * pl.col("C").mean()).round(2).alias("C_stage2_recall%"),
            (100 * pl.col("D").mean()).round(2).alias("D_final_recall%"),
            pl.col("D").sum().alias("TP"))
        g = (g.join(fp.group_by(keys).agg(pl.len().alias("FP")), on=keys, how="left")
              .with_columns(pl.col("FP").fill_null(0))
              .with_columns((100 * pl.col("TP") / (pl.col("TP") + pl.col("FP"))).round(2).alias("final_precision%"),
                            (100 * pl.col("true_pairs") / n).round(1).alias("share_of_pairs%"))
              .drop("TP").sort(keys))
        print(f"\n== {title}")
        print(g)

    table(["country"], "by country")
    table(["src"], "by source")
    table(["addr_empty"], "by address presence (1 = record has no address)")
    table(["country", "name_native"], "by script of the name (1 = native Indic script)")

    unmatched = queries.join(key.select("q_gi").unique(), on="q_gi", how="anti")
    wrong = dec.join(unmatched.select("q_gi"), on="q_gi").height
    print(f"\n== distractor records (no true match): {unmatched.height:,}; wrongly matched: {wrong:,} "
          f"({100 * wrong / unmatched.height:.2f}%)")
    print(f"   false merges in total: {fp.height:,} = {wrong:,} distractors + "
          f"{fp.height - wrong:,} real records matched to the wrong entity")


if __name__ == "__main__":
    main()
