"""Sibling-group rule (US / India): drop a match whose score is at most 0.98 when its house number, which the
Source-1 record does not have, is shared with records of the same entity that stage 4 rejected with confidence
(best candidate, probability < 0.01) and with no confident match (>= 0.9).

In the training data a business has at most one sibling (a distractor at the Source-1 address with a house
number plus an offset); in the test data siblings come in groups at the same number, which stage 4 never saw.
A match that shares its deviating number with such a group is then usually one of them.

    python sibling_rule.py <matching_results in> <test scores of stage 4> <out dir>
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import polars as pl

import stage4
from common import N_JOBS, WORK, read_tsv

MAX_P = 0.98          # only matches that are not near-certain
COUNTRIES = ["us", "india"]


def groups(scores: str) -> pl.DataFrame:
    """Best-candidate pairs whose deviating number is shared with rejected records and with no confident match."""
    sc = pl.read_parquet(scores).select("q_gi", "s1_gi", pl.col("p2").alias("p"))
    n = (pl.read_parquet(WORK / "test_norm.parquet", columns=["raw_addr", "country"])
           .with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32)))
    raws = n["raw_addr"].to_list()
    with Pool(N_JOBS) as pool:
        parts = pool.map(stage4._numbers, [raws[i:i + 200_000] for i in range(0, len(raws), 200_000)])
    n = n.drop("raw_addr").with_columns(pl.Series("expr", [x for part in parts for x in part]))
    d = (sc.join(n.select(pl.col("gi").alias("q_gi"), pl.col("expr").alias("q_expr")), on="q_gi")
           .join(n.select(pl.col("gi").alias("s1_gi"), pl.col("expr").alias("c_expr"), "country"), on="s1_gi"))
    d = d.with_columns(stage4._key("q_expr", "c_expr", "|").alias("key"),
                       pl.col("p").rank("ordinal", descending=True).over("q_gi").alias("rk"))
    first = pl.col("rk") == 1
    d = d.with_columns((first & (pl.col("p") >= 0.9)).cast(pl.Int32).alias("conf"),
                       (first & (pl.col("p") < 0.01)).cast(pl.Int32).alias("rej"))
    g = d.filter(pl.col("key") != "").group_by("s1_gi", "key").agg(pl.col("conf").sum().alias("g_conf"),
                                                                   pl.col("rej").sum().alias("g_rej"))
    d = d.join(g, on=["s1_gi", "key"], how="left").with_columns(
        (pl.col("g_conf") - pl.col("conf")).alias("o_conf"), (pl.col("g_rej") - pl.col("rej")).alias("o_rej"))
    return d.filter(first & (pl.col("key") != "") & (pl.col("o_rej") >= 1) & (pl.col("o_conf") == 0)
                    & pl.col("country").is_in(COUNTRIES) & (pl.col("p") <= MAX_P)).select("q_gi", "s1_gi")


def main(matching: str, scores: str, out_dir: str):
    ids = pl.read_parquet(WORK / "test_norm.parquet", columns=["entity_id"]).with_row_index("gi").with_columns(
        pl.col("gi").cast(pl.Int32))
    m = read_tsv(matching).with_columns(pl.col("matched_entity_ids").fill_null(""))
    acc = (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .join(ids.rename({"gi": "s1_gi", "entity_id": "source1_entity_id"}), on="source1_entity_id")
            .join(ids.rename({"gi": "q_gi", "entity_id": "matched_entity_ids"}), on="matched_entity_ids"))
    drop = groups(scores)
    keep = acc.join(drop, on=["q_gi", "s1_gi"], how="anti")
    lists = (keep.sort("source1_entity_id", "matched_entity_ids").group_by("source1_entity_id", maintain_order=True)
                 .agg(pl.col("matched_entity_ids").str.join(",").alias("new")))
    out = (m.join(lists, on="source1_entity_id", how="left")
            .with_columns(pl.col("new").fill_null("").alias("matched_entity_ids")).select("source1_entity_id", "matched_entity_ids"))
    assert out["source1_entity_id"].to_list() == m["source1_entity_id"].to_list()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.write_csv(Path(out_dir) / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"sibling-group rule: {acc.height - keep.height} matches dropped; written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
