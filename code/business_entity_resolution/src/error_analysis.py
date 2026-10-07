"""Diagnostic: out-of-fold error analysis on the training split (after `stage2.py tune`)."""
from __future__ import annotations

import glob
import json

import polars as pl

import stage1
from common import WORK
from evaluate import macro_f05
from features import load_norm
from stage2 import decode

pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_tbl_rows(40)


def main():
    dec = json.load(open(WORK / "decision.json"))
    norm = load_norm("train")
    raw = pl.read_parquet(WORK / "train_norm.parquet", columns=["raw_name", "raw_addr"]).with_row_index("gi")
    raw = raw.with_columns(pl.col("gi").cast(pl.Int32))
    ids = norm.select("entity_id", "src", "country").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
    pos, _ = stage1.train_labels(norm)
    sc = pl.read_parquet(WORK / "train_scores.parquet")
    pred = decode(sc, dec).with_columns(pl.lit(1, pl.Int8).alias("pred"))
    pruned = sc.select("q_gi", "s1_gi").unique().with_columns(pl.lit(1, pl.Int8).alias("in_pruned"))
    j = (pos.join(pred, on=["q_gi", "s1_gi"], how="full", coalesce=True)
            .join(pruned, on=["q_gi", "s1_gi"], how="left")
            .join(sc.select("q_gi", "s1_gi", "p2"), on=["q_gi", "s1_gi"], how="left")
            .join(norm.select(pl.col("addr_empty").alias("q_addr_empty"), pl.col("name_native").alias("q_name_native"),
                              pl.col("name_dup").alias("q_name_dup")).with_row_index("q_gi").with_columns(pl.col("q_gi").cast(pl.Int32)), on="q_gi")
            .join(ids.select(pl.col("gi").alias("q_gi"), "country"), on="q_gi"))
    fp = j.filter(pl.col("y").is_null() & (pl.col("pred") == 1))
    fn = j.filter((pl.col("y") == 1) & pl.col("pred").is_null())
    tp = j.filter((pl.col("y") == 1) & (pl.col("pred") == 1))
    print(f"pairs: TP {tp.height}  FP {fp.height}  FN {fn.height}   (pair P={tp.height/(tp.height+fp.height):.4f} "
          f"R={tp.height/(tp.height+fn.height):.4f})")
    print("FN breakdown:", fn.group_by(pl.col("in_pruned").fill_null(0), "q_addr_empty", "q_name_native")
          .len().sort("len", descending=True).head(10))
    print("FN by country:", fn.group_by("country").len())
    print("FP by country / query addr empty / native:", fp.group_by("country", "q_addr_empty", "q_name_native").len().sort("len", descending=True).head(10))
    print("FP p2 distribution:", fp["p2"].describe())
    print("FN (in pruned) p2 distribution:", fn.filter(pl.col("in_pruned") == 1)["p2"].describe())

    def show(df, name, n=25):
        d = (df.sample(min(n, df.height), seed=0)
               .join(raw.rename({"gi": "q_gi", "raw_name": "q_name", "raw_addr": "q_addr"}), on="q_gi")
               .join(raw.rename({"gi": "s1_gi", "raw_name": "s1_name", "raw_addr": "s1_addr"}), on="s1_gi")
               .select("country", "p2", "q_name", "s1_name", "q_addr", "s1_addr"))
        print(f"\n=== {name} sample"); print(d)
        d.write_csv(WORK / f"errors_{name}.tsv", separator="\t")

    show(fp, "fp")
    show(fn.filter(pl.col("in_pruned") == 1), "fn_scored")
    show(fn.filter(pl.col("in_pruned").is_null()), "fn_unblocked")


if __name__ == "__main__":
    main()
