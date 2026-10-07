"""Candidate generation (blocking).

Every Source-2 / Source-3 record ("query") is embedded as a sparse IDF-weighted
vector over typed features and compared with Source-1 records of the same
country using a multithreaded sparse top-N matrix product.

Feature families (prefix):
    n:  core name tokens (+ DBA alias tokens, + domain stem)
    k:  consonant skeleton of each name token     (typos / transliteration)
    f:  4-char prefix of each name token          (typos at the end of words)
    p:  6-char prefix of the concatenated name   (domain names, merged words)
    l:  legal-form tokens (down-weighted)        (tie-breaking between namesakes)
    a:  "strong" address tokens
    #:  house / door numbers

The name block and the address block are L2-normalised separately and mixed
with fixed weights, so the retrieval score is a convex combination of a name
cosine and an address cosine.  Three rankings are unioned:
  * top-k by the combined score
  * top-k_name by the name-only score
  * top-k_addr by the address-only score

Scaling: Source-1 is partitioned by state / region.  A query with a known state
is searched only in its own state partition plus "neighbour" states learned
from training pairs (e.g. Telangana <-> Andhra Pradesh, Delhi <-> Haryana);
queries without a state are searched against the whole country.  IDF is global
per country so scores are comparable across partitions.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from common import N_JOBS

_VOWELS = re.compile(r"[aeiouyhw]")
_DOUBLE = re.compile(r"(.)\1+")
LEGAL_WEIGHT = 0.35


def skeleton(tok: str) -> str:
    if len(tok) <= 2:
        return tok
    s = tok[0] + _VOWELS.sub("", tok[1:])
    return _DOUBLE.sub(r"\1", s)


def _explode(df: pl.DataFrame, col: str, prefix: str, max_items: int | None = None):
    e = (df.select("row", pl.col(col).str.split(" ").alias("t"))
           .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != "")))
    if max_items is not None:
        e = e.with_columns(pl.int_range(pl.len()).over("row").alias("pos")) \
             .filter(pl.col("pos") < max_items).drop("pos")
    return e.select("row", (pl.lit(prefix) + pl.col("t")).alias("f"))


def record_features(df: pl.DataFrame) -> pl.DataFrame:
    """df must have a 'row' column. Returns (row, f, block, mult)."""
    name_toks = pl.concat([
        _explode(df, "core", "n:"),
        _explode(df, "alias", "n:"),
        df.filter(pl.col("domain") != "").select("row", (pl.lit("n:") + pl.col("domain")).alias("f")),
    ])
    toks = name_toks.with_columns(pl.col("f").str.slice(2).alias("t"))
    uniq = toks.select("t").unique()
    skel = uniq.with_columns(pl.col("t").map_elements(skeleton, return_dtype=pl.String).alias("k"))
    sk = toks.join(skel, on="t").select("row", (pl.lit("k:") + pl.col("k")).alias("f"))
    # skeleton without the first letter: robust to a corrupted initial ("lnvestment")
    sk2 = (toks.filter(pl.col("t").str.len_chars() >= 5).join(skel, on="t")
               .select("row", (pl.lit("j:") + pl.col("k").str.slice(1)).alias("f")))
    pf = (toks.filter(pl.col("t").str.len_chars() >= 5)
              .select("row", (pl.lit("f:") + pl.col("t").str.slice(0, 4)).alias("f")))
    pref = pl.concat([
        df.filter(pl.col("concat").str.len_chars() >= 6)
          .select("row", (pl.lit("p:") + pl.col("concat").str.slice(0, 6)).alias("f")),
        df.filter(pl.col("domain").str.len_chars() >= 6)
          .select("row", (pl.lit("p:") + pl.col("domain").str.slice(0, 6)).alias("f")),
    ])
    legal = _explode(df, "legal", "l:")
    # initials: every multi-token name gets "i:<initials>"; a single short token is also
    # emitted as "i:<token>" so an initials-only record ("CAA") can retrieve its entity
    core_l = pl.col("core").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    ini = pl.concat([
        df.filter(core_l.list.len() >= 2).select(
            "row", (pl.lit("i:") + core_l.list.eval(pl.element().str.slice(0, 1)).list.join("")).alias("f")),
        df.filter((core_l.list.len() == 1) & pl.col("core").str.len_chars().is_between(2, 5)).select(
            "row", (pl.lit("i:") + pl.col("core")).alias("f")),
    ])
    name = pl.concat([name_toks, sk, sk2, pf, pref, legal, ini]).with_columns(pl.lit(0, pl.Int8).alias("block"))
    addr = pl.concat([
        _explode(df, "addr_strong", "a:"),
        _explode(df, "nums", "#:", max_items=3),
    ]).with_columns(pl.lit(1, pl.Int8).alias("block"))
    out = pl.concat([name, addr]).unique(["row", "f"])
    return out.with_columns(pl.when(pl.col("f").str.starts_with("l:")).then(LEGAL_WEIGHT)
                              .otherwise(1.0).alias("mult"))


# Length-normalisation exponent per block for the Source-1 side ("alpha").  The
# query side uses 2 - alpha, so that a pair scores  shared / (|q|^(2-a) * |c|^a):
# alpha = 1 is the cosine; alpha < 1 penalises long Source-1 addresses less, which
# helps short / partial query addresses ("A 80, East Delhi").
ALPHA = {0: 1.0, 1: 0.5}


def _matrix(feats: pl.DataFrame, vocab: pl.DataFrame, n_rows: int, block: int | None,
            w_name: float, w_addr: float, side: str) -> sp.csr_matrix:
    """block=None -> combined vector; 0 -> name only; 1 -> address only.
    side = 'q' (query) or 'c' (Source-1 candidate)."""
    f = feats.join(vocab, on="f")  # drops features unseen in S1 / too frequent
    if block is not None:
        f = f.filter(pl.col("block") == block)
    f = f.with_columns((pl.col("idf") * pl.col("mult")).alias("w"))
    f = f.with_columns((pl.col("w") ** 2).sum().over(["row", "block"]).sqrt().alias("nrm"))
    a = pl.when(pl.col("block") == 0).then(ALPHA[0]).otherwise(ALPHA[1])
    expo = a if side == "c" else 2.0 - a
    scale = (pl.lit(1.0) if block is not None
             else pl.when(pl.col("block") == 0).then(w_name).otherwise(w_addr))
    f = f.with_columns((pl.col("w") / pl.col("nrm").pow(expo) * scale).cast(pl.Float32).alias("v"))
    m = sp.csr_matrix((f["v"].to_numpy(), (f["row"].to_numpy(), f["col"].to_numpy())),
                      shape=(n_rows, vocab.height), dtype=np.float32)
    m.sort_indices()
    return m


def _topn(Q: sp.csr_matrix, ST: sp.csr_matrix, k: int, chunk: int = 100_000):
    rows, cols, vals = [], [], []
    for i in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[i:i + chunk], ST, top_n=k, threshold=1e-6, n_threads=N_JOBS).tocoo()
        rows.append(C.row.astype(np.int64) + i)
        cols.append(C.col.astype(np.int64))
        vals.append(C.data)
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ra: np.ndarray, rb: np.ndarray,
                 chunk: int = 2_000_000) -> np.ndarray:
    out = np.empty(len(ra), dtype=np.float32)
    for i in range(0, len(ra), chunk):
        a = A[ra[i:i + chunk]]
        b = B[rb[i:i + chunk]]
        out[i:i + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


class BlockIndex:
    """Source-1 side of the blocking for one country (built once, queried in chunks)."""

    def __init__(self, s1: pl.DataFrame, neighbors: dict | None = None, max_df: int = 50000,
                 w_name: float = 0.6, w_addr: float = 0.4):
        self.neighbors = neighbors or {}
        self.w_name, self.w_addr = w_name, w_addr
        s1 = s1.with_columns(pl.int_range(pl.len()).alias("row"))
        fs = record_features(s1)
        self.n1 = n1 = s1.height
        self.vocab = (fs.group_by("f").agg(pl.len().alias("df"))
                        .filter(pl.col("df") <= max_df)
                        .with_columns((np.log((n1 + 1) / (pl.col("df") + 1)) + 1.0).alias("idf"))
                        .with_columns(pl.int_range(pl.len()).alias("col"))
                        .select("f", "idf", "col"))
        # name-only and address-only matrices; the combined one is their weighted
        # sum (disjoint feature columns), built per partition on demand
        self.S = {tag: _matrix(fs, self.vocab, n1, blk, w_name, w_addr, "c")
                  for tag, blk in (("n", 0), ("a", 1))}
        del fs
        self.s1_state = s1["state"].to_numpy()
        self.states = sorted(set(self.s1_state.tolist()) - {""})
        self.part = {st: np.flatnonzero(self.s1_state == st) for st in self.states}

    def _partT(self, tag: str, idx: np.ndarray | None) -> sp.csr_matrix:
        if tag == "c":
            m = self.w_name * self.S["n"] + self.w_addr * self.S["a"]
        else:
            m = self.S[tag]
        if idx is not None:
            m = m[idx]
        return m.T.tocsr()

    def query(self, q: pl.DataFrame, k: int = 20, k_name: int = 10,
              k_addr: int = 10) -> pl.DataFrame:
        """Candidate pairs (q_row, s1_row, s_name, s_addr, s_comb, rk_comb, rk_name, rk_addr);
        q_row indexes `q`, s1_row indexes the Source-1 frame given at construction."""
        q = q.with_columns(pl.int_range(pl.len()).alias("row"))
        fq = record_features(q)
        Q = {tag: _matrix(fq, self.vocab, q.height, blk, self.w_name, self.w_addr, "q")
             for tag, blk in (("n", 0), ("a", 1))}
        del fq
        Q["c"] = (self.w_name * Q["n"] + self.w_addr * Q["a"]).tocsr()
        q_state = q["state"].to_numpy()
        q_alt = q["state_alt"].to_numpy()
        known = np.isin(q_state, self.states)
        plans = []  # (query rows, partition key or None for all of Source 1)
        for st in self.states:
            search_from = list({st} | {a for a, nb in self.neighbors.items() if st in nb})
            q_idx = np.flatnonzero((np.isin(q_state, search_from) | np.isin(q_alt, search_from))
                                   & known)
            if len(q_idx):
                plans.append((q_idx, st))
        q_unknown = np.flatnonzero(~known)
        if len(q_unknown):
            plans.append((q_unknown, None))

        kk = {"c": k, "n": k_name, "a": k_addr}
        parts = []
        for tag in ("c", "n", "a"):
            if kk[tag] <= 0:
                continue
            for q_idx, st in plans:
                s_idx = None if st is None else self.part[st]
                ST = self._partT(tag, s_idx)
                r, c, _ = _topn(Q[tag][q_idx], ST, kk[tag])
                del ST
                s_map = c if s_idx is None else s_idx[c]
                parts.append(pl.DataFrame({"q_row": q_idx[r], "s1_row": s_map}))
        cand = pl.concat(parts).unique(["q_row", "s1_row"])
        qr, sr = cand["q_row"].to_numpy(), cand["s1_row"].to_numpy()
        sn = _rowwise_dot(Q["n"], self.S["n"], qr, sr)
        sa = _rowwise_dot(Q["a"], self.S["a"], qr, sr)
        cand = cand.with_columns(pl.Series("s_name", sn), pl.Series("s_addr", sa))
        cand = cand.with_columns(
            (self.w_name * pl.col("s_name") + self.w_addr * pl.col("s_addr")).alias("s_comb"))
        cand = cand.with_columns(
            pl.col("s_comb").rank("ordinal", descending=True).over("q_row").cast(pl.Int16).alias("rk_comb"),
            pl.col("s_name").rank("ordinal", descending=True).over("q_row").cast(pl.Int16).alias("rk_name"),
            pl.col("s_addr").rank("ordinal", descending=True).over("q_row").cast(pl.Int16).alias("rk_addr"),
        )
        # queries searched in several partitions: keep the best k per ranking overall
        return cand.filter((pl.col("rk_comb") <= k) | (pl.col("rk_name") <= k_name)
                           | (pl.col("rk_addr") <= k_addr))


def block_country(s1: pl.DataFrame, q: pl.DataFrame, neighbors: dict | None = None,
                  k: int = 20, k_name: int = 10, k_addr: int = 10, **kw) -> pl.DataFrame:
    return BlockIndex(s1, neighbors, **kw).query(q, k, k_name, k_addr)


def learn_state_neighbors(pairs: pl.DataFrame, min_count: int = 200, min_share: float = 0.002):
    """pairs: (country, st1, st2) for matched training pairs.
    Returns {country: {query_state: set(s1_states)}} for frequent cross-state pairs."""
    tot = pairs.group_by("country", "st2").agg(pl.len().alias("n"))
    x = (pairs.filter((pl.col("st1") != pl.col("st2")) & (pl.col("st1") != "") & (pl.col("st2") != ""))
              .group_by("country", "st2", "st1").agg(pl.len().alias("c"))
              .join(tot, on=["country", "st2"]).filter((pl.col("c") >= min_count)
                                                       & (pl.col("c") / pl.col("n") >= min_share)))
    out: dict = {}
    for c, st2, st1, _, _ in x.iter_rows():
        out.setdefault(c, {}).setdefault(st2, set()).add(st1)
        out.setdefault(c, {}).setdefault(st1, set()).add(st2)
    return out
