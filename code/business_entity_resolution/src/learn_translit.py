"""Step 1: learn the Indic-script transliteration dictionaries from train pairs."""
import polars as pl

import translit
from common import WORK, Timer, read_ground_truth, read_split


def main():
    with Timer("learn transliteration"):
        s = read_split("train")
        gt = read_ground_truth()
        other = pl.concat([s[2], s[3]]).filter(
            pl.col("business_name").str.contains(r"[ऀ-෿]")
            | pl.col("business_address").fill_null("").str.contains(r"[ऀ-෿]"))
        pairs = (gt.join(other, left_on="q_id", right_on="entity_id")
                   .join(s[1], left_on="s1_id", right_on="entity_id", suffix="_s1"))
        rows = pairs.select("business_name_s1", "business_address_s1", "country_s1",
                            "business_name", "business_address").iter_rows()
        tr, st = translit.learn(rows)
        translit.save(tr, st, WORK / "translit.json")
        print(f"  {pairs.height} pairs -> {len(tr)} token mappings, {len(st)} state mappings")


if __name__ == "__main__":
    main()
