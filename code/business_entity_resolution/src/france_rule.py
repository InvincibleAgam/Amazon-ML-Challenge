"""France rule: keep a French match only if BOTH versions of stage 4 accept it.

The text models (text_model.py) are trained on the US / India labels.  On the leaderboard they
helped US / India as predicted, but the French matches they ADD were mostly siblings (another
business at the same address), while the French matches they REJECT were mostly errors of the
version without text.  So for France a match needs both votes: the decision of stage 4 without
the text models (its own rule) and the text version's probability >= T_FRANCE.  US and India
keep the text version's decision.

    python france_rule.py <matching_results with text> <matching_results without text>
                          <test scores with text> <t_france> <out dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from common import DATA, WORK, read_tsv
from stage2 import reduce_argmax


def main(with_text: str, without_text: str, scores: str, t_france: float, out_dir: str):
    rd = lambda p: read_tsv(p).with_columns(pl.col("matched_entity_ids").fill_null(""))   # noqa: E731
    a, b = rd(without_text), rd(with_text)
    s1 = read_tsv(DATA / "test" / "test_source1.tsv").select(
        pl.col("entity_id").alias("source1_entity_id"), pl.col("country").str.to_lowercase().alias("c"))
    ids = pl.read_parquet(WORK / "test_norm.parquet", columns=["entity_id"]).with_row_index("gi").with_columns(
        pl.col("gi").cast(pl.Int32))
    top = reduce_argmax(pl.read_parquet(scores).select("q_gi", "s1_gi", "p2"))
    ok = (top.filter((pl.col("best") >= t_france) & (pl.col("best") - pl.col("second") >= 0.1))
             .join(ids.rename({"gi": "q_gi", "entity_id": "q"}), on="q_gi")
             .join(ids.rename({"gi": "s1_gi", "entity_id": "s1"}), on="s1_gi").select("s1", "q"))
    fr = (a.join(s1, on="source1_entity_id").filter(pl.col("c") == "france")
           .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1", "matched_entity_ids": "q"}))
    kept = fr.join(ok, on=["s1", "q"], how="semi")
    lists = (kept.sort("s1", "q").group_by("s1", maintain_order=True).agg(pl.col("q").str.join(","))
                 .rename({"s1": "source1_entity_id", "q": "fr"}))
    out = (b.join(s1, on="source1_entity_id", how="left").join(lists, on="source1_entity_id", how="left")
            .with_columns(pl.when(pl.col("c") == "france").then(pl.col("fr").fill_null(""))
                            .otherwise(pl.col("matched_entity_ids")).alias("matched_entity_ids"))
            .select("source1_entity_id", "matched_entity_ids"))
    assert out["source1_entity_id"].to_list() == b["source1_entity_id"].to_list()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.write_csv(Path(out_dir) / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"France: {fr.height} matches without text, {kept.height} kept at t = {t_france} "
          f"({fr.height - kept.height} dropped); written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
