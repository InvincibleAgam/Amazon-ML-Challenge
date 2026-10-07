"""Out-of-fold results of a decoded version on the full training split, for the write-up.

    python final_numbers.py <name>=<train scores parquet>=<threshold> [...]   # -> work/final_numbers.json
"""
from __future__ import annotations

import json
import sys

import polars as pl

import stage1
import swap
from common import WORK
from exp_trees import Evaluator
from stage2 import decode_threshold, reduce_argmax


def main(specs: list[str]):
    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id", "addr_empty"] + swap.NORM_COLS)
    pos, _ = stage1.train_labels(norm)
    ev = Evaluator(norm, pos)
    words, synonyms = swap.country_words(norm), swap.load_synonyms()
    empty = norm.select("addr_empty").with_row_index("q_gi").with_columns(pl.col("q_gi").cast(pl.Int32))
    matched = pos.select("q_gi").unique().with_columns(pl.lit(1).alias("m"))
    out = {}
    for spec in specs:
        name, path, t = spec.split("=")
        t = float(t)
        top = reduce_argmax(pl.read_parquet(path).select("q_gi", "s1_gi", "p2"))
        flagged = swap.flag(top.select("q_gi", "s1_gi"), norm, words, synonyms)
        top = top.join(flagged.filter(pl.col("swap")).select("q_gi", "s1_gi"), on=["q_gi", "s1_gi"], how="anti")
        a = ev.train_prior(top, "threshold", {"threshold": t, "margin": 0.1})
        b = ev.test_prior(top, "threshold", {"threshold": t, "margin": 0.1})
        acc = decode_threshold(top, t, 0.1).join(pos, on=["q_gi", "s1_gi"], how="left")
        fm = acc.filter(pl.col("y").is_null()).join(matched, on="q_gi", how="left")
        tp = acc.filter(pl.col("y").is_not_null()).join(empty, on="q_gi")
        allp = pos.join(empty, on="q_gi")
        out[name] = {"training prior": a, "test-like prior": b, "accepted": acc.height, "false merges": fm.height,
                     "false merges: distractor records": int(fm["m"].is_null().sum()),
                     "false merges: copy of another entity": int(fm["m"].is_not_null().sum()),
                     "recall with address": tp.filter(pl.col("addr_empty") == 0).height / allp.filter(pl.col("addr_empty") == 0).height,
                     "recall without address": tp.filter(pl.col("addr_empty") == 1).height / allp.filter(pl.col("addr_empty") == 1).height}
        print(name, json.dumps(out[name], indent=1, default=float), flush=True)
    json.dump(out, open(WORK / "final_numbers.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
