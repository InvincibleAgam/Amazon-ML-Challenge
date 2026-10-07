"""Export the open pairs (the ones stage 4 re-scores) with their raw text, for the text models
that are trained on a GPU (`text_model.py`).

    python export_pairs.py         # -> work/text/pairs_train.parquet, work/text/pairs_test.parquet
    python export_pairs.py extra   # -> work/text/extra_train.parquet, work/text/extra_test.parquet:
                                   #    the other pairs of the wider zone (score_pairs.py scores them)

Columns: q_gi, s1_gi, text_a (record: name | address), text_b (Source-1 entity), country,
p3 (probability so far), and for train: y (label) and fold (same folds as every other stage).
"""
from __future__ import annotations

import polars as pl

import stage1
from common import WORK

OUT = WORK / "text"


def export(split: str):
    keys = pl.read_parquet(WORK / f"{split}_feat4.parquet", columns=["q_gi", "s1_gi", "p3"])
    rec = (pl.read_parquet(WORK / f"{split}_norm.parquet", columns=["entity_id", "country", "raw_name", "raw_addr"])
             .with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32)))
    text = (pl.col("raw_name").fill_null("") + " | " + pl.col("raw_addr").fill_null("")).alias("text")
    rec = rec.with_columns(text)
    d = (keys.join(rec.select(pl.col("gi").alias("q_gi"), pl.col("text").alias("text_a")), on="q_gi", how="left")
             .join(rec.select(pl.col("gi").alias("s1_gi"), pl.col("text").alias("text_b"), "country"), on="s1_gi", how="left"))
    if split == "train":
        pos, fold = stage1.train_labels(rec.select("entity_id"))
        d = d.join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(pl.col("y").fill_null(0))
        d = d.with_columns(pl.Series("fold", fold[d["q_gi"].to_numpy()]).cast(pl.Int8))
    OUT.mkdir(exist_ok=True)
    d.write_parquet(OUT / f"pairs_{split}.parquet")
    print(split, d.height, "pairs", {c: str(t) for c, t in d.schema.items()})
    print(d.head(3))


def export_extra(split: str, lo: float = 0.001, hi: float = 0.999):
    """All pairs of the records whose best stage-3 probability lies in (lo, hi), minus the open pairs
    exported above (the wider zone of stage 4, ER_ZONE=lo,hi)."""
    sc = pl.read_parquet(WORK / f"{split}_scores3.parquet").select("q_gi", "s1_gi", pl.col("p2").alias("p3"))
    sc = sc.with_columns(pl.col("p3").max().over("q_gi").alias("b")).filter((pl.col("b") > lo) & (pl.col("b") < hi))
    done = pl.read_parquet(OUT / f"pairs_{split}.parquet", columns=["q_gi", "s1_gi"])
    x = sc.join(done, on=["q_gi", "s1_gi"], how="anti").select("q_gi", "s1_gi")
    rec = (pl.read_parquet(WORK / f"{split}_norm.parquet", columns=["entity_id", "raw_name", "raw_addr"])
             .with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32),
                                                (pl.col("raw_name").fill_null("") + " | " + pl.col("raw_addr").fill_null("")).alias("text")))
    x = (x.join(rec.select(pl.col("gi").alias("q_gi"), pl.col("text").alias("text_a")), on="q_gi")
          .join(rec.select(pl.col("gi").alias("s1_gi"), pl.col("text").alias("text_b")), on="s1_gi"))
    if split == "train":
        _, fold = stage1.train_labels(rec.select("entity_id"))
        x = x.with_columns(pl.Series("fold", fold[x["q_gi"].to_numpy()]).cast(pl.Int8))
    x.write_parquet(OUT / f"extra_{split}.parquet")
    print(split, "extra pairs:", x.height)


if __name__ == "__main__":
    import sys
    for sp in ("train", "test"):
        export_extra(sp) if sys.argv[1:] == ["extra"] else export(sp)
