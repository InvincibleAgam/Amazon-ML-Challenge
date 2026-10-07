"""Pairwise features for (query record, Source-1 candidate) pairs.

Two tiers:
  * stage 1 ("cheap"): retrieval scores / ranks, per-query context, a few fast
    string similarities.  Used by the pruning model on every blocked pair.
  * stage 2 ("full"): everything in stage 1 plus heavier fuzzy-token features and
    collective context derived from the stage-1 probabilities.

All features are country-agnostic (no country one-hot), so the model transfers
to countries absent from training (France).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from blocking import skeleton
from common import N_JOBS, WORK

_CHAR_VEC: dict = {}
_CHAR_SRC = {"name": ("char_vec.joblib", "core"), "addr": ("char_vec_addr.joblib", "addr_strong")}


def char_vectorizer(kind: str = "name"):
    """Character 3-4-gram TF-IDF over Source-1 core names / strong address tokens of the
    training split (cached on disk)."""
    if kind not in _CHAR_VEC:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        fname, col = _CHAR_SRC[kind]
        path = WORK / fname
        if path.exists():
            _CHAR_VEC[kind] = joblib.load(path)
        else:
            texts = (pl.read_parquet(WORK / "train_norm.parquet", columns=["src", col])
                       .filter((pl.col("src") == 1) & (pl.col(col) != ""))[col].unique().to_list())
            _CHAR_VEC[kind] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3,
                                              dtype=np.float32, sublinear_tf=True).fit(texts)
            joblib.dump(_CHAR_VEC[kind], path)
    return _CHAR_VEC[kind]


def char_cos(a: list[str], b: list[str], kind: str = "name", chunk: int = 500_000) -> np.ndarray:
    vec = char_vectorizer(kind)
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), chunk):
        A = vec.transform(a[i:i + chunk]); B = vec.transform(b[i:i + chunk])  # rows L2-normalised
        out[i:i + chunk] = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return out

NORM_COLS = ["name", "core", "legal", "alias", "domain", "concat", "name_native", "addr_native",
             "addr_strong", "nums", "state", "state_alt", "addr_empty", "src", "skel", "name_dup",
             "addr_dup"]


def load_norm(split: str) -> pl.DataFrame:
    """Normalised records + consonant-skeleton of the core name + name ambiguity (cached)."""
    path = WORK / f"{split}_norm_feat3.parquet"
    if path.exists():
        return pl.read_parquet(path)
    norm = pl.read_parquet(WORK / f"{split}_norm.parquet")
    toks = norm.select(pl.col("core").str.split(" ").explode().unique().alias("t")).drop_nulls()
    sk = dict(zip(toks["t"].to_list(), [skeleton(t) for t in toks["t"].to_list()]))
    norm = norm.with_columns(
        pl.col("core").str.split(" ").list.eval(pl.element().replace_strict(sk, default=""))
          .list.join(" ").alias("skel"))
    # number of Source-1 entities in the same country with exactly this core name
    dup = (norm.filter(pl.col("src") == 1).group_by("country", "core")
               .agg(pl.len().cast(pl.Int32).alias("name_dup")))
    norm = norm.join(dup, on=["country", "core"], how="left").with_columns(
        pl.col("name_dup").fill_null(0))
    # number of Source-1 entities in the same country with exactly this full normalised address
    # (street tokens + numbers); high in France, where many entities share an address
    adup = (norm.filter((pl.col("src") == 1) & (pl.col("addr_empty") == 0))
                .group_by("country", "addr_strong", "nums").agg(pl.len().cast(pl.Int32).alias("addr_dup")))
    norm = norm.join(adup, on=["country", "addr_strong", "nums"], how="left").with_columns(
        pl.when(pl.col("addr_empty") == 1).then(0).otherwise(pl.col("addr_dup").fill_null(0)).alias("addr_dup"))
    norm = norm.select(["entity_id", "country"] + NORM_COLS)
    norm.write_parquet(path)
    return norm


def _cp(a, b, scorer, **kw) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=N_JOBS, dtype=np.float32, **kw)


def _jacc(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    inter = a.list.set_intersection(b).list.len()
    uni = a.list.set_union(b).list.len()
    return pl.when(uni > 0).then(inter / uni).otherwise(None)


def _split(col: str, side: str) -> pl.Expr:
    return pl.col(f"{side}_{col}").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _num_rel(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Relation between two house/door numbers: -1 missing, 1 equal, 2 one is a prefix of the
    other (truncation noise: 118 vs 1189), 3 suffix (29 vs 329), 4 same length but different
    digits (typically a different building: 6106 vs 6101), 0 anything else."""
    return (pl.when((a == "") | (b == "")).then(-1)
              .when(a == b).then(1)
              .when(a.str.starts_with(b) | b.str.starts_with(a)).then(2)
              .when(a.str.ends_with(b) | b.str.ends_with(a)).then(3)
              .when(a.str.len_chars() == b.str.len_chars()).then(4)
              .otherwise(0)).cast(pl.Int8)


def _num_soft_contain(df: pl.DataFrame) -> pl.Series:
    """Share of the query's numbers that equal / prefix / suffix some candidate number."""
    q, c = _split("nums", "q"), _split("nums", "c")
    e = (df.with_row_index("_i").select("_i", q.alias("qn"), c.alias("cn"))
           .explode("qn").filter(pl.col("qn").is_not_null()).explode("cn")
           .with_columns(_num_rel(pl.col("qn"), pl.col("cn").fill_null("")).is_in([1, 2, 3]).alias("ok"))
           .group_by("_i", "qn").agg(pl.col("ok").max())
           .group_by("_i").agg(pl.col("ok").mean().alias("v")))
    return (pl.DataFrame({"_i": pl.int_range(0, df.height, eager=True).cast(pl.UInt32)})
              .join(e, on="_i", how="left")["v"])


def context_features(cand: pl.DataFrame) -> pl.DataFrame:
    """Per-query competition features from retrieval scores."""
    out = cand.with_columns(pl.len().over("q_gi").cast(pl.Int16).alias("n_cand"))
    for s in ("s_comb", "s_name", "s_addr"):
        mx = pl.col(s).max().over("q_gi")
        # best score among the *other* candidates of the same query
        second = pl.col(s).sort(descending=True).slice(1, 1).first().over("q_gi").fill_null(0.0)
        other_best = pl.when(pl.col(s) >= mx).then(second).otherwise(mx)
        out = out.with_columns((pl.col(s) - other_best).alias(f"{s}_gap"))
    # how many candidates share (almost) the best name score: name ambiguity
    out = out.with_columns(
        (pl.col("s_name") >= pl.col("s_name").max().over("q_gi") - 0.02).sum().over("q_gi")
        .cast(pl.Int16).alias("n_name_ties"))
    return out


def pair_features(cand: pl.DataFrame, norm: pl.DataFrame, full: bool,
                  cheap: bool = True) -> pl.DataFrame:
    """cand: q_gi, s1_gi, retrieval columns (+ context).  Adds the stage-1 ("cheap")
    features if `cheap` and the stage-2 extras if `full`."""
    qi = cand["q_gi"].to_numpy()
    ci = cand["s1_gi"].to_numpy()
    q = norm.select(NORM_COLS)[qi].rename({c: f"q_{c}" for c in NORM_COLS})
    c = norm.select(NORM_COLS)[ci].rename({c: f"c_{c}" for c in NORM_COLS})
    # a frame coming out of stage 1 already carries some q_* columns: keep those
    q = q.drop([col for col in q.columns if col in cand.columns])
    c = c.drop([col for col in c.columns if col in cand.columns])
    added = q.columns + c.columns
    df = pl.concat([cand, q, c], how="horizontal")

    f = {}
    qc, cc = df["q_core"].to_list(), df["c_core"].to_list()
    qx, cx = df["q_concat"].to_list(), df["c_concat"].to_list()
    qa, ca = df["q_addr_strong"].to_list(), df["c_addr_strong"].to_list()
    if cheap:
        f["name_ratio"] = _cp(qc, cc, fuzz.ratio) / 100
        f["name_jw"] = _cp(qx, cx, JaroWinkler.normalized_similarity)
        f["addr_ratio"] = _cp(qa, ca, fuzz.ratio) / 100
    if full:
        f["name_tsort"] = _cp(qc, cc, fuzz.token_sort_ratio) / 100
        f["name_tset"] = _cp(qc, cc, fuzz.token_set_ratio) / 100
        f["name_partial"] = _cp(qx, cx, fuzz.partial_ratio) / 100
        f["name_lev"] = _cp(qx, cx, Levenshtein.normalized_similarity)
        f["addr_tset"] = _cp(qa, ca, fuzz.token_set_ratio) / 100
        f["addr_partial"] = _cp(qa, ca, fuzz.partial_ratio) / 100
        # DBA alias / domain name of the query against the candidate name
        qal = df["q_alias"].to_list()
        f["alias_tset"] = _cp(qal, cc, fuzz.token_set_ratio) / 100
        qd = df["q_domain"].to_list()
        cfull = (df["c_concat"] + df["c_legal"].str.replace_all(" ", "")).to_list()
        f["dom_partial"] = np.maximum(_cp(qd, cx, fuzz.partial_ratio), _cp(qd, cfull, fuzz.partial_ratio)) / 100
        f["dom_ratio"] = np.maximum(_cp(qd, cx, fuzz.ratio), _cp(qd, cfull, fuzz.ratio)) / 100
        f["skel_ratio"] = _cp(df["q_skel"].to_list(), df["c_skel"].to_list(), fuzz.token_set_ratio) / 100
        f["name_char_cos"] = char_cos(qc, cc, "name")
        f["addr_char_cos"] = char_cos(qa, ca, "addr")
        # first numbers: typo-tolerant comparison of house / door numbers
        qn1 = df["q_nums"].str.split(" ").list.first().fill_null("").to_list()
        cn1 = df["c_nums"].str.split(" ").list.first().fill_null("").to_list()
        f["num1_lev"] = _cp(qn1, cn1, Levenshtein.normalized_similarity)
    df = df.with_columns([pl.Series(k, v) for k, v in f.items()])

    has_dom = pl.col("q_domain") != ""
    has_alias = pl.col("q_alias") != ""
    q_nums, c_nums = _split("nums", "q"), _split("nums", "c")
    # initials-only query names ("CAA" for "Coaching Air Amis"): compare the single query token
    # with the initials of the candidate's core tokens (2) / of all its tokens incl. legal (3),
    # or a prefix of the core initials (1); 0 otherwise, -1 when the query is not initials-like
    q_core_l, c_core_l = _split("core", "q"), _split("core", "c")
    q_tok = q_core_l.list.first().fill_null("")
    c_ini = c_core_l.list.eval(pl.element().str.slice(0, 1)).list.join("")
    c_ini_all = pl.col("c_name").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    acro_like = (q_core_l.list.len() == 1) & (q_tok.str.len_chars().is_between(2, 5)) & (c_core_l.list.len() >= 2)
    acro_expr = (pl.when(~acro_like).then(-1)
                   .when(q_tok == c_ini).then(2).when(q_tok == c_ini_all).then(3)
                   .when((c_ini.str.len_chars() > 2) & c_ini.str.starts_with(q_tok)).then(1)
                   .otherwise(0)).cast(pl.Int8).alias("acro_match")
    exprs = [] if not cheap else [acro_expr,
        pl.col("q_addr_empty").cast(pl.Int8),
        pl.col("q_name_native").cast(pl.Int8),
        pl.col("q_addr_native").cast(pl.Int8),
        pl.col("q_name_dup").cast(pl.Int32),
        pl.col("c_name_dup").cast(pl.Int32),
        (pl.col("q_src") == 3).cast(pl.Int8).alias("q_is_s3"),
        has_dom.cast(pl.Int8).alias("q_has_domain"),
        # state agreement: 1 same / 0 different / -1 unknown; alt covers city-named-as-state
        pl.when((pl.col("q_state") == "") | (pl.col("c_state") == "")).then(-1)
          .when((pl.col("q_state") == pl.col("c_state")) | (pl.col("q_state_alt") == pl.col("c_state"))
                | (pl.col("q_state") == pl.col("c_state_alt"))).then(1)
          .otherwise(0).cast(pl.Int8).alias("state_eq"),
        pl.when(pl.col("q_legal") == "").then(-1)
          .when(pl.col("q_legal") == pl.col("c_legal")).then(1)
          .when(pl.col("q_legal").str.split(" ").list.set_intersection(
                pl.col("c_legal").str.split(" ")).list.len() > 0).then(2)
          .otherwise(0).cast(pl.Int8).alias("legal_eq"),
        pl.when(q_nums.list.len() == 0).then(-1).when(c_nums.list.len() == 0).then(-2)
          .when(q_nums.list.first() == c_nums.list.first()).then(1)
          .otherwise(0).cast(pl.Int8).alias("num1_eq"),
        (pl.col("q_core") == pl.col("c_core")).cast(pl.Int8).alias("core_exact"),
    ]
    if full:
        q_core, c_core = _split("core", "q"), _split("core", "c")
        q_ad, c_ad = _split("addr_strong", "q"), _split("addr_strong", "c")
        exprs += [
            _jacc(q_core, c_core).alias("name_jacc"),
            _jacc(q_ad, c_ad).alias("addr_jacc"),
            _jacc(q_nums, c_nums).alias("num_jacc"),
            # containment of the query's tokens in the candidate's (partial query addresses)
            pl.when(q_ad.list.len() > 0).then(
                q_ad.list.set_intersection(c_ad).list.len() / q_ad.list.unique().list.len())
              .otherwise(None).alias("addr_contain"),
            pl.when(q_nums.list.len() > 0).then(
                q_nums.list.set_intersection(c_nums).list.len() / q_nums.list.unique().list.len())
              .otherwise(None).alias("num_contain"),
            pl.when(q_core.list.len() > 0).then(
                q_core.list.set_intersection(c_core).list.len() / q_core.list.unique().list.len())
              .otherwise(None).alias("name_contain"),
            pl.when(c_core.list.len() > 0).then(
                c_core.list.set_intersection(q_core).list.len() / c_core.list.unique().list.len())
              .otherwise(None).alias("name_contain_c"),
            q_core.list.len().cast(pl.Int8).alias("q_n_core"),
            c_core.list.len().cast(pl.Int8).alias("c_n_core"),
            q_ad.list.len().cast(pl.Int8).alias("q_n_addr"),
            c_ad.list.len().cast(pl.Int8).alias("c_n_addr"),
            (q_core.list.first() == c_core.list.first()).cast(pl.Int8).alias("first_tok_eq"),
            has_alias.cast(pl.Int8).alias("q_has_alias"),
            pl.col("c_addr_dup").cast(pl.Int32),
            pl.col("q_addr_dup").cast(pl.Int32),
            _num_rel(q_nums.list.first().fill_null(""), c_nums.list.first().fill_null("")).alias("num_first_rel"),
            _num_rel(q_nums.list.last().fill_null(""), c_nums.list.last().fill_null("")).alias("num_last_rel"),
        ]
    df = df.with_columns(exprs)
    if full:
        df = df.with_columns(_num_soft_contain(df).alias("num_soft_contain"))
    keep = {"q_addr_empty", "q_name_native", "q_addr_native", "q_name_dup", "c_name_dup",
            "c_addr_dup", "q_addr_dup"}
    return df.drop([col for col in added if col not in keep])


STAGE1_FEATURES = [
    "s_name", "s_addr", "s_comb", "rk_comb", "rk_name", "rk_addr", "n_cand",
    "s_comb_gap", "s_name_gap", "s_addr_gap", "n_name_ties",
    "name_ratio", "name_jw", "addr_ratio",
    "q_addr_empty", "q_name_native", "q_addr_native", "q_name_dup", "c_name_dup", "q_is_s3",
    "q_has_domain", "state_eq", "legal_eq", "num1_eq", "core_exact", "acro_match",
]
STAGE2_EXTRA = [
    "name_tsort", "name_tset", "name_partial", "name_lev", "addr_tset", "addr_partial",
    "alias_tset", "dom_partial", "dom_ratio", "skel_ratio", "num1_lev",
    "name_jacc", "addr_jacc", "num_jacc", "addr_contain", "num_contain", "name_contain",
    "name_contain_c", "q_n_core", "c_n_core", "q_n_addr", "c_n_addr", "first_tok_eq",
    "q_has_alias", "num_first_rel", "num_last_rel", "num_soft_contain", "c_addr_dup", "q_addr_dup",
    "name_char_cos", "addr_char_cos", "c_claims_s2", "c_claims_s3",
    # collective context from stage-1 probabilities (see collective_features)
    "p1", "p1_rank", "p1_gap", "p1_qsum", "c_claims", "c_claim_rank", "c_p1_max_other",
]
COLLECTIVE_FEATURES = ["p1_rank", "p1_gap", "p1_qsum", "c_claims", "c_claim_rank", "c_p1_max_other",
                       "c_claims_s2", "c_claims_s3"]
STAGE2_FEATURES = STAGE1_FEATURES + STAGE2_EXTRA
# ER_FEATURES=nocoll trains / scores stage 2 without the cross-record ("collective") features
if __import__("os").environ.get("ER_FEATURES") == "nocoll":
    STAGE2_FEATURES = [f for f in STAGE2_FEATURES if f not in COLLECTIVE_FEATURES]


def collective_features(df: pl.DataFrame) -> pl.DataFrame:
    """Context from stage-1 probabilities `p1` on the pruned candidate set.

    Query side : rank / gap of this candidate among the query's candidates.
    S1 side    : how many queries 'claim' this S1 (it is their best candidate),
                 and how this query ranks among them.
    """
    mx = pl.col("p1").max().over("q_gi")
    second = pl.col("p1").sort(descending=True).slice(1, 1).first().over("q_gi").fill_null(0.0)
    df = df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q_gi").cast(pl.Int16).alias("p1_rank"),
        (pl.col("p1") - pl.when(pl.col("p1") >= mx).then(second).otherwise(mx)).alias("p1_gap"),
        pl.col("p1").sum().over("q_gi").alias("p1_qsum"),
    )
    is_claim = (pl.col("p1_rank") == 1) & (pl.col("p1") >= 0.5)
    df = df.with_columns(is_claim.alias("_claim"))
    if "q_is_s3" in df.columns:   # claims split by source (the generator caps them at 5 / 6 per entity)
        df = df.with_columns(
            (pl.col("_claim") & (pl.col("q_is_s3") == 0)).sum().over("s1_gi").cast(pl.Int16).alias("c_claims_s2"),
            (pl.col("_claim") & (pl.col("q_is_s3") == 1)).sum().over("s1_gi").cast(pl.Int16).alias("c_claims_s3"))
    df = df.with_columns(
        pl.col("_claim").sum().over("s1_gi").cast(pl.Int16).alias("c_claims"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_gi").cast(pl.Int16).alias("c_claim_rank"),
    )
    # best p1 of *other* queries on this S1
    mxc = pl.col("p1").max().over("s1_gi")
    secc = pl.col("p1").sort(descending=True).slice(1, 1).first().over("s1_gi").fill_null(0.0)
    df = df.with_columns(pl.when(pl.col("p1") >= mxc).then(secc).otherwise(mxc).alias("c_p1_max_other"))
    return df.drop("_claim")
