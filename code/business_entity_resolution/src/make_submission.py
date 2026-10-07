"""Step 6: write output/matching_results.tsv and output/candidate_pairs.tsv for the test split.

candidate_pairs.tsv is the stage-1 pruned set: exactly the pairs the matching
models (stage 2 and its correction, stage 3) run inference over.
"""
from __future__ import annotations

import glob
import json
import os

import polars as pl

import swap
from common import OUT, WORK, Timer
from features import load_norm
from stage2 import decode

SWAP_VETO = os.environ.get("ER_SWAP_VETO", "1") == "1"   # drop matches whose name swaps a word (swap.py)
STAGE3 = os.environ.get("ER_STAGE3", "1") == "1"         # decode the corrected probability (stage3.py)
STAGE4 = STAGE3 and os.environ.get("ER_STAGE4", "1") == "1"   # ... after the collective correction (stage4.py)
LAST = "4" if STAGE4 else "3"


def _write(pairs: pl.DataFrame, s1_ids: pl.Series, col: str, path):
    """pairs: (s1_id, q_id) long format -> one row per S1 id, comma-joined ids (sorted)."""
    lists = (pairs.unique().sort(["s1_id", "q_id"])
                  .group_by("s1_id", maintain_order=True).agg(pl.col("q_id").str.join(",").alias(col)))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
             .join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    which = os.environ.get("ER_DECISION", "")          # "" -> decision.json, "testprior" -> decision_testprior.json
    dec = json.load(open(WORK / (f"decision{LAST}.json" if STAGE3 else f"decision_{which}.json" if which else "decision.json")))
    print("decision rule:", dec)
    norm = load_norm("test")
    gid = norm.select("entity_id", "src", "country").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
    s1_ids = gid.filter(pl.col("src") == 1)["entity_id"]
    q_map = gid.select(pl.col("gi").alias("q_gi"), pl.col("entity_id").alias("q_id"))
    s_map = gid.select(pl.col("gi").alias("s1_gi"), pl.col("entity_id").alias("s1_id"), "country")
    with Timer("write candidate_pairs.tsv"):
        cand = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"])
                          for p in sorted(glob.glob(str(WORK / "test_pruned" / "*.parquet")))])
        cand = cand.join(q_map, on="q_gi").join(s_map, on="s1_gi").select("s1_id", "q_id")
        c = _write(cand, s1_ids, "candidate_entity_ids", OUT / "candidate_pairs.tsv")
        print(f"  {cand.height} candidate pairs, {(c['candidate_entity_ids'] != '').sum()} S1 with candidates")
    with Timer(f"write matching_results.tsv ({dec['method']})"):
        sc = pl.read_parquet(WORK / (f"test_scores{LAST}.parquet" if STAGE3 else "test_scores.parquet"))
        a = decode(sc, dec)
        if SWAP_VETO:
            words = swap.country_words(norm)
            print("  generic words appended after the legal form:", {c: sorted(g) for c, g in words[1].items()})
            f = swap.flag(a, norm, words).join(s_map, on="s1_gi")
            print("  word-swap veto, matches dropped per country:",
                  dict(f.filter(pl.col("swap")).group_by("country").len().sort("country").iter_rows()))
            a = f.filter(~pl.col("swap")).select("q_gi", "s1_gi")
        a = a.join(q_map, on="q_gi").join(s_map, on="s1_gi")
        m = _write(a.select("s1_id", "q_id"), s1_ids, "matched_entity_ids", OUT / "matching_results.tsv")
        print(f"  {a.height} matched pairs, {(m['matched_entity_ids'] != '').sum()} of {len(s1_ids)} S1 matched")
        # sanity: matches per S1 entity by country (compare with train: 3.46 per entity, 5.6% singletons)
        st = (a.group_by("country").agg(pl.len().alias("pairs"), pl.col("s1_id").n_unique().alias("s1_matched"))
                .join(gid.filter(pl.col("src") == 1).group_by("country").agg(pl.len().alias("s1_total")), on="country")
                .with_columns((pl.col("pairs") / pl.col("s1_total")).alias("pairs_per_s1"),
                              (1 - pl.col("s1_matched") / pl.col("s1_total")).alias("singleton_rate")))
        print(st.sort("country"))


if __name__ == "__main__":
    main()
