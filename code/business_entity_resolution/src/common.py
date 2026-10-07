"""Paths, IO helpers and shared constants."""
from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA = Path(os.environ.get("ER_DATA", ROOT / "dataset"))
WORK = Path(os.environ.get("ER_WORK", ROOT / "work"))
OUT = Path(os.environ.get("ER_OUT", ROOT / "output"))
N_JOBS = int(os.environ.get("ER_JOBS", os.cpu_count() or 4))
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def read_tsv(path) -> pl.DataFrame:
    # quote_char=None: fields are never quoted, and names may contain '"'
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def read_split(split: str) -> dict[int, pl.DataFrame]:
    return {s: read_tsv(DATA / split / f"{split}_source{s}.tsv") for s in (1, 2, 3)}


def read_ground_truth() -> pl.DataFrame:
    """Long format: one row per (s1 id, matched id)."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"}))


class Timer:
    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {self.msg} ...", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[{time.strftime('%H:%M:%S')}] {self.msg} done in {time.time() - self.t:.1f}s",
              flush=True)
