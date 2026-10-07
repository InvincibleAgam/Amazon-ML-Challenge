"""Step 2: normalise every record of a split and cache it as parquet.

Output: WORK/<split>_norm.parquet with one row per record of all three sources.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import polars as pl

import translit
from common import N_JOBS, WORK, Timer, read_split
from normalize import has_indic, normalize_address, normalize_name

_TR = _ST = None


def _init(path):
    global _TR, _ST
    _TR, _ST = translit.load(path)


def _work(rows):
    out = []
    for eid, name, addr, country in rows:
        n = normalize_name(name, _TR)
        a = normalize_address(addr, country or "", _ST, _TR)
        out.append((
            eid,
            " ".join(n["tokens"]), " ".join(n["core"]), " ".join(n["legal"]),
            " ".join(n["alias"]), n["domain"], n["concat"],
            int(has_indic(name)), int(has_indic(addr)),
            " ".join(a["tokens"]), " ".join(a["strong"]), " ".join(a["nums"]),
            a["state"], a["state_alt"], a["empty"],
        ))
    return out


COLS = ["entity_id", "name", "core", "legal", "alias", "domain", "concat", "name_native",
        "addr_native", "addr", "addr_strong", "nums", "state", "state_alt", "addr_empty"]


def normalize_frame(df: pl.DataFrame, chunk: int = 20000) -> pl.DataFrame:
    rows = list(df.select("entity_id", "business_name", "business_address", "country")
                  .iter_rows())
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    res = []
    with Pool(N_JOBS, initializer=_init, initargs=(str(WORK / "translit.json"),)) as pool:
        for part in pool.imap(_work, chunks, chunksize=1):
            res.extend(part)
    out = pl.DataFrame(res, schema=COLS, orient="row")
    return out.with_columns(pl.col("name_native", "addr_native", "addr_empty").cast(pl.Int8))


def run(split: str):
    with Timer(f"normalise {split}"):
        s = read_split(split)
        parts = []
        for src, df in s.items():
            n = normalize_frame(df)
            n = n.with_columns(pl.lit(src, pl.Int8).alias("src"),
                               df["country"].str.to_lowercase().alias("country"),
                               df["business_name"].alias("raw_name"),
                               df["business_address"].alias("raw_addr"))
            parts.append(n)
            print(f"  source{src}: {n.height} rows", flush=True)
        out = pl.concat(parts)
        out.write_parquet(WORK / f"{split}_norm.parquet")


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp)
