"""Step 3: blocking for a whole split.

Writes WORK/<split>_cand/<country>_<chunk>.parquet with columns
    q_gi, s1_gi  : global row indices into WORK/<split>_norm.parquet
    s_name, s_addr, s_comb, rk_comb, rk_name, rk_addr : retrieval scores / ranks
"""
from __future__ import annotations

import json
import shutil
import sys

import polars as pl

import blocking
from common import WORK, Timer, read_ground_truth

K, K_NAME, K_ADDR = 20, 10, 10
CHUNK = 500_000


def learn_neighbors():
    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id", "country", "state"])
    gt = read_ground_truth()
    pairs = (gt.join(norm.rename({"entity_id": "s1_id", "state": "st1"}), on="s1_id")
               .join(norm.select(pl.col("entity_id").alias("q_id"), pl.col("state").alias("st2")),
                     on="q_id"))
    nb = blocking.learn_state_neighbors(pairs)
    nb = {c: {k: sorted(v) for k, v in d.items()} for c, d in nb.items()}
    with open(WORK / "state_neighbors.json", "w") as f:
        json.dump(nb, f, indent=1)
    return nb


def run(split: str):
    nb_path = WORK / "state_neighbors.json"
    nb = json.load(open(nb_path)) if nb_path.exists() else learn_neighbors()
    nb = {c: {k: set(v) for k, v in d.items()} for c, d in nb.items()}
    out_dir = WORK / f"{split}_cand"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    cols = ["src", "country", "core", "alias", "domain", "concat", "legal", "addr_strong",
            "nums", "state", "state_alt"]
    lf = pl.scan_parquet(WORK / f"{split}_norm.parquet").select(cols).with_row_index("gi")
    countries = lf.select(pl.col("country").unique()).collect()["country"].sort().to_list()
    total = 0
    for country in countries:
        d = lf.filter(pl.col("country") == country).collect()
        d = d.with_columns(pl.col("gi").cast(pl.Int32))
        s1 = d.filter(pl.col("src") == 1)
        qs = d.filter(pl.col("src") != 1)
        del d
        if s1.height == 0 or qs.height == 0:
            continue
        with Timer(f"[{split}] index {country}: {s1.height} S1, {qs.height} queries"):
            idx = blocking.BlockIndex(s1, nb.get(country))
            s1_gi = s1["gi"].to_numpy()
            for ci, start in enumerate(range(0, qs.height, CHUNK)):
                q = qs.slice(start, CHUNK)
                cand = idx.query(q, K, K_NAME, K_ADDR)
                q_gi = q["gi"].to_numpy()
                cand = cand.select(
                    pl.Series("q_gi", q_gi[cand["q_row"].to_numpy()]),
                    pl.Series("s1_gi", s1_gi[cand["s1_row"].to_numpy()]),
                    "s_name", "s_addr", "s_comb", "rk_comb", "rk_name", "rk_addr")
                cand.write_parquet(out_dir / f"{country}_{ci:03d}.parquet")
                total += cand.height
                print(f"    chunk {ci}: {q.height} queries -> {cand.height} pairs", flush=True)
                del cand
            del idx, s1, qs
    print(f"[{split}] total candidate pairs: {total}")


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp)
