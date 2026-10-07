"""Diagnostic: blocking recall on a sample of training queries."""
import sys
import polars as pl
from common import WORK, Timer, read_ground_truth
import blocking

frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
kw = dict(k=int(sys.argv[2]) if len(sys.argv) > 2 else 20,
          k_name=int(sys.argv[3]) if len(sys.argv) > 3 else 10,
          k_addr=int(sys.argv[4]) if len(sys.argv) > 4 else 10)
norm = pl.read_parquet(WORK / "train_norm.parquet")
gt = read_ground_truth()
st = norm.select("entity_id", "country", "state")
pairs = (gt.join(st.rename({"entity_id": "s1_id", "state": "st1"}), on="s1_id")
           .join(st.select(pl.col("entity_id").alias("q_id"), pl.col("state").alias("st2")), on="q_id"))
nb = blocking.learn_state_neighbors(pairs)
print("neighbors", nb)
for country in ["us", "india"]:
    s1 = norm.filter((pl.col("src") == 1) & (pl.col("country") == country))
    q = norm.filter((pl.col("src") != 1) & (pl.col("country") == country)).sample(fraction=frac, seed=0)
    with Timer(f"block {country} {q.height} queries"):
        cand = blocking.block_country(s1, q, nb.get(country), **kw)
    cand = cand.join(q.select(pl.int_range(pl.len()).alias("q_row"), pl.col("entity_id").alias("q_id")), on="q_row") \
               .join(s1.select(pl.int_range(pl.len()).alias("s1_row"), pl.col("entity_id").alias("s1_id")), on="s1_row")
    truth = gt.join(q.select(pl.col("entity_id").alias("q_id"), "addr_empty"), on="q_id")
    hit = truth.join(cand, on=["q_id", "s1_id"], how="left")
    print(country, "queries", q.height, "with truth", truth.height, "cands/query %.1f" % (cand.height / q.height))
    print("  union recall %.4f   (addr present %.4f, addr empty %.4f, empty share %.3f)" % (
        hit["rk_comb"].is_not_null().mean(),
        hit.filter(pl.col("addr_empty") == 0)["rk_comb"].is_not_null().mean(),
        hit.filter(pl.col("addr_empty") == 1)["rk_comb"].is_not_null().mean(),
        hit["addr_empty"].mean()))
    for col in ["rk_comb", "rk_name", "rk_addr"]:
        r = hit[col].fill_null(9999).to_numpy()
        print("  ", col, " ".join(f"@{k}:{(r <= k).mean():.4f}" for k in [1, 2, 3, 5, 10, 20, 30]))
    miss = hit.filter(pl.col("rk_comb").is_null()).head(3000).join(norm.select(pl.col("entity_id").alias("q_id"), pl.col("raw_name").alias("q_name"), pl.col("raw_addr").alias("q_addr")), on="q_id") \
              .join(norm.select(pl.col("entity_id").alias("s1_id"), "raw_name", "raw_addr"), on="s1_id")
    miss.select("q_name", "raw_name", pl.col("q_addr").fill_null("<EMPTY>"), "raw_addr").write_csv(WORK / f"block_miss_{country}.tsv", separator="\t")
