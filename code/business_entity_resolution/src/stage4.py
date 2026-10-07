"""Step 5d: collective correction (stage 4).

Stages 1-3 score one (record, Source-1 entity) pair at a time.  The records of one business are
not independent, though: the generator gives a business a few variants of its address (another
house number, a range "305-307", "305 1/2", another name for the city) and several of its copies
carry the same variant.  A copy with an unusual variant looks like the "sibling" distractor of
the business (same name, house number plus an offset) when it is judged alone, and is certain
when another copy with the same variant is already matched with confidence.

Stage 4 is a LightGBM on top of the stage-3 probability, fitted only on the records that stage 3
leaves open (best probability between ZONE[0] and ZONE[1]); all other records keep their stage-3
score.  Its features, next to those of stages 2 and 3:
  * support: how many OTHER records that are matched to the same Source-1 entity with
    confidence (stage-3 probability >= CONF) deviate from the Source-1 record in the same way
    (same other number, same other address word, same address, same name), and how many
    confident matches the entity has at all.  Only confident matches count, so that records
    which are themselves in doubt cannot vouch for each other;
  * spelling: similarity of the raw names (punctuation and the spelling of the legal form
    survive in copies), edits of two-letter words, letters shared by garbled words, which word
    was added or dropped, how two numbers differ (digit dropped, inserted, replaced).
It is cross-fitted with the same folds as stages 1-3.

    python stage4.py words                 # frequent added / dropped words -> work/stage4_words.json
    python stage4.py features train|test   # -> work/<split>_feat4.parquet
    python stage4.py train                 # -> work/stage4_models.joblib, work/train_scores4.parquet
    python stage4.py predict test          # -> work/test_scores4.parquet
    python stage4.py tune                  # decoding rule on p4 -> work/decision4.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

import joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import models
import stage1
import swap
from common import N_JOBS, ROOT, WORK, Timer
from features import STAGE2_FEATURES
from normalize import LEGAL_CANON
from stage3 import CATEGORICAL as CATEGORICAL3
from stage3 import STAGE3_FEATURES

PASS = int(os.environ.get("ER_PASS", "1"))   # 1: on the stage-3 scores; 2: once more, on the scores of pass 1
PREV, NOW = f"scores{2 + PASS}", f"scores{3 + PASS}"         # scores3 -> scores4, scores4 -> scores5
FEAT, MODEL = f"feat{3 + PASS}", f"stage{3 + PASS}_models.joblib"
# records whose best probability so far lies in between are re-scored (ER_ZONE="0.001,0.999" widens it)
ZONE = tuple(float(x) for x in os.environ.get("ER_ZONE", "0.02,0.98").split(","))
CONF = 0.9               # stage-3 probability from which a match vouches for other records
N_WORDS = 300            # added / dropped words that get an id of their own
MIN_WORD = 100           # ... if they occur at least this often among the training pairs
OTHER = N_WORDS + 1

CTX4 = ["p3", "p3_rank", "p3_gap", "p3_second"]
SUP4 = ["sup_num", "sup_tok", "sup_addr", "sup_name", "sup_other", "e_conf", "e_conf_src", "e_conf_empty",
        "src_n", "src_agree", "src_dis", "oth_n", "oth_agree"]
PAIR4 = ["raw_ratio", "raw_gap", "legal_style", "punct_same", "se_dist", "se_code", "g_bag", "g_firstlast",
         "g_lendiff", "q_name_num", "q_dup_word", "w_added_id", "w_dropped_id", "w_added_pos", "nr_type",
         "nr_lev", "nr_len_q", "nr_len_c", "q_extra_first", "q_variant", "c_raw_n"]
# probabilities of the text models (text_model.py, trained on a GPU), e.g. ER_TEXT=A,X,C; read from
# <ER_TEXT_DIR>/text_scores_<split>_<tag>.parquet.  Without them stage 4 runs on its own features.
TEXT = [t for t in os.environ.get("ER_TEXT", "").split(",") if t]
TEXT_DIR = Path(os.environ.get("ER_TEXT_DIR", WORK / "text"))
# per text model: its probability, the margin over the record's best other candidate, and how many
# other records the text model gives the same entity with confidence
TEXT4 = [f"{c}_{t}" for t in TEXT for c in ("pt", "ptgap", "ptclaims")]
STAGE4_FEATURES = STAGE3_FEATURES + CTX4 + SUP4 + PAIR4 + TEXT4
CATEGORICAL = CATEGORICAL3 + ["se_code", "w_added_id", "w_dropped_id", "nr_type"]
PARAMS = dict(n_estimators=1200, learning_rate=0.03, num_leaves=63, min_child_samples=50, reg_lambda=2.0,
              colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=4, cat_smooth=20,
              min_data_per_group=100)
PARAMS.update(json.loads(os.environ.get("ER_S4_PARAMS", "{}")))     # e.g. a larger model once text features exist

NAME_NUM_RE = re.compile(r"#\s*\d+|\b\d{6,}\b|\b[il1]d\s*[:#]?\s*\d+", re.I)
RANGE_RE = re.compile(r"\b(\d+)\s*-\s*(\d+)\b")
HALF_RE = re.compile(r"\b(\d+)\s+1/2\b")
LETTER_RE = re.compile(r"\b(\d+)\s*-?\s*[a-d]\b", re.I)
WORD_RE = re.compile(r"[a-z.]+")
NUMBER_RE = re.compile(r"(\d+)\s?-\s?(\d+)|(\d+)\s+1/2|(\d+)-?([a-d])\b|(\d+)")
MAX_RANGE = 20           # "305-307" is one house number if the two differ by at most this
BOX_RE = re.compile(r"\b(?:p\.?\s?o\.?\s*box|post\s*box|pmb)\s*#?\s*\d+")   # mail boxes are noise, not a house


# ------------------------------------------------------------------------------------------
# support between the records of one Source-1 entity
# ------------------------------------------------------------------------------------------
def _strip(n: str) -> str:
    return n.lstrip("0") or "0"


def numbers(raw: str | None) -> str:
    """The numbers of a raw address, with a range (305-307), a half (305 1/2) and a letter
    (305b) kept as ONE number: they are variants of the house number, and 307 alone is another
    house.  Returned as a space-free token list joined by "|"."""
    out = []
    for m in NUMBER_RE.finditer(BOX_RE.sub(" ", (raw or "").lower())):
        if m.group(1):
            a, b = _strip(m.group(1)), _strip(m.group(2))
            out += [a + "-" + b] if 0 < int(b) - int(a) <= MAX_RANGE else [a, b]
        elif m.group(3):
            out.append(_strip(m.group(3)) + "+1/2")
        elif m.group(4):
            out.append(_strip(m.group(4)) + m.group(5))
        else:
            out.append(_strip(m.group(6)))
    return "|".join(out)


def _numbers(raws: list) -> list:
    return [numbers(r) for r in raws]


def _tokens(col: str) -> pl.Expr:
    return pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _key(own: str, other: str, sep: str = " ") -> pl.Expr:
    """What the record has and the Source-1 record has not, as a sorted string ("" = nothing)."""
    def toks(col):
        return pl.col(col).fill_null("").str.split(sep).list.eval(pl.element().filter(pl.element() != ""))
    return toks(own).list.set_difference(toks(other)).list.sort().list.join(" ")


def support(split: str) -> pl.DataFrame:
    """One row per scored pair: the support features (SUP4) and the stage-3 context (CTX4)."""
    sc = pl.read_parquet(WORK / f"{split}_{PREV}.parquet").select("q_gi", "s1_gi", pl.col("p2").alias("p3"))
    rec = (pl.read_parquet(WORK / f"{split}_norm.parquet",
                           columns=["src", "core", "addr", "addr_strong", "nums", "addr_empty", "raw_addr"])
             .with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32)))
    raws = rec["raw_addr"].to_list()
    with Pool(N_JOBS) as pool:
        parts = pool.map(_numbers, [raws[i:i + 200_000] for i in range(0, len(raws), 200_000)])
    rec = rec.drop("raw_addr").with_columns(pl.Series("expr", [x for part in parts for x in part]))
    del raws, parts
    q = rec.select(pl.col("gi").alias("q_gi"), pl.col("src").alias("q_src"), pl.col("core").alias("q_core"),
                   pl.col("addr_strong").alias("q_strong"), pl.col("nums").alias("q_nums"),
                   pl.col("expr").alias("q_expr"), pl.col("addr_empty").alias("q_empty"))
    c = rec.select(pl.col("gi").alias("s1_gi"), pl.col("core").alias("c_core"), pl.col("addr").alias("c_addr"),
                   pl.col("expr").alias("c_expr"))
    del rec
    d = sc.join(q, on="q_gi", how="left").join(c, on="s1_gi", how="left")
    d = d.with_columns(
        pl.col("p3").rank("ordinal", descending=True).over("q_gi").cast(pl.Int8).alias("p3_rank"),
        pl.col("p3").max().over("q_gi").alias("_best"),
        _key("q_expr", "c_expr", "|").alias("k_num"), _key("q_strong", "c_addr").alias("k_tok"),
        pl.when(pl.col("q_empty") == 0)
          .then(_tokens("q_strong").list.sort().list.join(" ") + "|" + _tokens("q_nums").list.sort().list.join(" "))
          .otherwise(pl.lit("")).alias("k_addr"),
        pl.when(pl.col("q_core").fill_null("") != pl.col("c_core").fill_null(""))
          .then(pl.col("q_core").fill_null("")).otherwise(pl.lit("")).alias("k_name"))
    second = d.filter(pl.col("p3_rank") == 2).select("q_gi", pl.col("p3").alias("p3_second"))
    d = d.join(second, on="q_gi", how="left").with_columns(pl.col("p3_second").fill_null(0.0))
    d = d.with_columns(
        (pl.col("p3") - pl.when(pl.col("p3_rank") == 1).then(pl.col("p3_second")).otherwise(pl.col("_best"))).alias("p3_gap"),
        ((pl.col("p3_rank") == 1) & (pl.col("p3") >= CONF)).cast(pl.Int32).alias("_conf"),
        (pl.col("q_expr").fill_null("") != "").alias("_num"))
    d = d.with_columns((pl.col("_conf") * pl.col("_num")).cast(pl.Int32).alias("_confn"))

    def others(keys: list[str], name: str, flag: str = "_conf", cond: pl.Expr | None = None) -> pl.DataFrame:
        """Records with the flag that share `keys`, the record itself not counted."""
        src = d if cond is None else d.filter(cond)
        g = src.group_by(keys).agg(pl.col(flag).sum().alias("_g"))
        return g.rename({"_g": name})

    for key, name in (("k_num", "sup_num"), ("k_tok", "sup_tok"), ("k_addr", "sup_addr"), ("k_name", "sup_name")):
        g = others(["s1_gi", key], name, cond=pl.col(key) != "")
        d = d.join(g, on=["s1_gi", key], how="left").with_columns(
            pl.when(pl.col(key) == "").then(None).otherwise(pl.col(name) - pl.col("_conf")).cast(pl.Int16).alias(name))
    # records at the same address that are matched with confidence to ANOTHER Source-1 entity
    g = others(["k_addr"], "_all", cond=pl.col("k_addr") != "")
    own = others(["s1_gi", "k_addr"], "_own", cond=pl.col("k_addr") != "")
    d = d.join(g, on="k_addr", how="left").join(own, on=["s1_gi", "k_addr"], how="left").with_columns(
        pl.when(pl.col("k_addr") == "").then(None).otherwise(pl.col("_all") - pl.col("_own")).cast(pl.Int16).alias("sup_other"))
    # confident matches of the entity: all, from the record's source, without address; open records
    d = d.join(others(["s1_gi"], "e_conf"), on="s1_gi", how="left")
    d = d.join(others(["s1_gi", "q_src"], "e_conf_src"), on=["s1_gi", "q_src"], how="left")
    d = d.join(others(["s1_gi"], "e_conf_empty", cond=pl.col("q_empty") == 1), on="s1_gi", how="left")
    d = d.with_columns(
        (pl.col("e_conf") - pl.col("_conf")).cast(pl.Int16), (pl.col("e_conf_src") - pl.col("_conf")).cast(pl.Int16),
        (pl.col("e_conf_empty").fill_null(0) - pl.col("_conf") * (pl.col("q_empty") == 1)).cast(pl.Int16))
    # the house number is a property of (entity, source): confident matches of the entity that
    # carry a number, from the record's source and from the other source, and how many of them
    # carry the same number as the record
    for keys, name in ((["s1_gi", "q_src"], "_src"), (["s1_gi", "q_src", "k_num"], "_src_same"),
                       (["s1_gi"], "_all_n"), (["s1_gi", "k_num"], "_all_same")):
        d = d.join(others(keys, name, flag="_confn"), on=keys, how="left")
    own = pl.col("_confn")
    d = d.with_columns(
        pl.when(pl.col("_num")).then(pl.col("_src") - own).cast(pl.Int16).alias("src_n"),
        pl.when(pl.col("_num")).then(pl.col("_src_same") - own).cast(pl.Int16).alias("src_agree"),
        pl.when(pl.col("_num")).then(pl.col("_src") - pl.col("_src_same")).cast(pl.Int16).alias("src_dis"),
        pl.when(pl.col("_num")).then(pl.col("_all_n") - pl.col("_src")).cast(pl.Int16).alias("oth_n"),
        pl.when(pl.col("_num")).then(pl.col("_all_same") - pl.col("_src_same")).cast(pl.Int16).alias("oth_agree"))
    # open records: stage 3 is undecided, or sure of a match that contradicts the number of the
    # entity in this source, or sure of a non-match that carries that number
    first = pl.col("p3_rank") == 1
    open_ = (first & (pl.col("p3") > ZONE[0]) & (pl.col("p3") < ZONE[1])
             | first & (pl.col("p3") >= ZONE[1]) & (pl.col("src_dis") > 0) & (pl.col("src_agree") == 0)
             | first & (pl.col("p3") <= ZONE[0]) & (pl.col("src_agree") > 0) & (pl.col("k_num") != ""))
    d = d.with_columns(open_.fill_null(False).max().over("q_gi").alias("zone"))
    return d.select(["q_gi", "s1_gi", "zone"] + CTX4 + SUP4)


# ------------------------------------------------------------------------------------------
# spelling features of one pair
# ------------------------------------------------------------------------------------------
def _clean(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def _legal_style(raw: str) -> str:
    """The legal form as it is written: "pvt. ltd.", "private limited", "l.l.c."."""
    return " ".join(t for t in WORD_RE.findall(raw) if t.replace(".", "") in LEGAL_CANON)


def _unmatched(q_only: list, c_only: list) -> tuple[list, list]:
    """(record words that no Source-1 word explains, Source-1 words left over)."""
    added = [x for x in q_only if not any(swap._same(x, y) for y in c_only)]
    left = [y for y in c_only if not any(swap._same(x, y) for x in q_only)]
    return added, left


def _short_edit(q_only: list, c_only: list) -> tuple[int, int]:
    """One letter replaced in a word of two letters (GI -> Gl): (distance, edit code)."""
    for x in q_only:
        for y in c_only:
            if len(x) == 2 and len(y) == 2 and Levenshtein.distance(x, y) == 1:
                i = 0 if x[0] != y[0] else 1
                return 1, 1600 + 40 * swap._char(y[i]) + swap._char(x[i])
    return 0, 0


def _garble(added: list, left: list) -> tuple[int, int, int]:
    """Best pair of unexplained words with the same initial: share of letters in common
    (any order), same last letter, difference in length."""
    best = (0, 0, 0)
    for x in added:
        for y in left:
            if x[0] != y[0]:
                continue
            common = sum((Counter(x) & Counter(y)).values())
            bag = int(round(200 * common / (len(x) + len(y))))
            if bag > best[0]:
                best = (bag, int(x[-1] == y[-1]), len(x) - len(y))
    return best


def _number_relation(q_only: list, c_only: list) -> tuple[int, int, int, int]:
    """How the closest pair of numbers that do not match differs: (type, edit distance, digits of
    the record's number, digits of the Source-1 number).  Type: 1 last digit dropped, 2 first
    digit dropped, 3 other digit dropped, 4 digit added, 5 one digit replaced, 6 two digits
    replaced, 7 anything else, 0 no such pair."""
    if not q_only or not c_only:
        return 0, -1, len(q_only[0]) if q_only else 0, len(c_only[0]) if c_only else 0
    lev, a, b = min((Levenshtein.distance(a, b), a, b) for a in q_only for b in c_only)
    if len(a) == len(b) - 1 and lev == 1:
        kind = 1 if b.startswith(a) else 2 if b.endswith(a) else 3
    elif len(a) == len(b) + 1 and lev == 1:
        kind = 4
    elif len(a) == len(b) and lev <= 2:
        kind = 5 if lev == 1 else 6
    else:
        kind = 7
    return kind, lev, len(a), len(b)


def _variant(raw_addr: str, c_nums: set) -> int:
    """The record writes a Source-1 number as a range (305-307: 1), a half (305 1/2: 2) or with
    a letter (305b: 3)."""
    for m in RANGE_RE.finditer(raw_addr):
        if m.group(1).lstrip("0") in c_nums and 0 < int(m.group(2)) - int(m.group(1)) <= 10:
            return 1
    for m in HALF_RE.finditer(raw_addr):
        if m.group(1).lstrip("0") in c_nums:
            return 2
    for m in LETTER_RE.finditer(raw_addr):
        if m.group(1).lstrip("0") in c_nums:
            return 3
    return 0


_W: dict = {}


def _init(added: dict, dropped: dict):
    _W.update(added=added, dropped=dropped)


def _pair(row) -> tuple:
    q_raw, c_raw, q_core, c_core, q_name, q_nums, c_nums, q_addr = row
    qr, cr = _clean(q_raw), _clean(c_raw)
    qs, cs = _legal_style(qr), _legal_style(cr)
    style = -1 if not qs or not cs else int(qs == cs)
    punct = sum(int((ch in qr) == (ch in cr)) for ch in ",&.'-")
    qw, cw = (q_core or "").split(), (c_core or "").split()
    q_only, c_only = [x for x in qw if x not in cw], [y for y in cw if y not in qw]
    added, left = _unmatched(q_only, c_only)
    se = _short_edit(q_only, c_only)
    ga = _garble(added, left)
    names = (q_name or "").split()
    qn, cn = (q_nums or "").split(), (c_nums or "").split()
    qx, cx = [a for a in qn if a not in cn], [b for b in cn if b not in qn]
    nr = _number_relation(qx, cx)
    return (int(round(fuzz.ratio(qr, cr))), style, punct, se[0], se[1], ga[0], ga[1], ga[2],
            int(bool(NAME_NUM_RE.search(qr))), int(len(set(names)) < len(names)),
            _W["added"].get(added[0], OTHER) if added else 0, _W["dropped"].get(left[0], OTHER) if left else 0,
            qw.index(added[0]) - len(qw) if added else 0,
            nr[0], nr[1], nr[2], nr[3], int(bool(qx) and bool(qn) and qn[0] == qx[0] and not cx),
            _variant((q_addr or "").lower(), set(cn)))


def _pairs(rows) -> list:
    return [_pair(r) for r in rows]


PAIR_RAW = ["raw_ratio", "legal_style", "punct_same", "se_dist", "se_code", "g_bag", "g_firstlast", "g_lendiff",
            "q_name_num", "q_dup_word", "w_added_id", "w_dropped_id", "w_added_pos", "nr_type", "nr_lev",
            "nr_len_q", "nr_len_c", "q_extra_first", "q_variant"]


def _texts(split: str, keys: pl.DataFrame) -> pl.DataFrame:
    rec = (pl.read_parquet(WORK / f"{split}_norm.parquet",
                           columns=["country", "src", "raw_name", "raw_addr", "name", "core", "nums"])
             .with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32)))
    s1 = rec.filter(pl.col("src") == 1).with_columns(pl.col("raw_name").str.to_lowercase().alias("_r"))
    raw_n = s1.group_by("country", "_r").agg(pl.len().cast(pl.Int32).alias("c_raw_n"))
    c = (s1.join(raw_n, on=["country", "_r"], how="left")
           .select(pl.col("gi").alias("s1_gi"), pl.col("raw_name").alias("c_raw"), pl.col("core").alias("c_core"),
                   pl.col("nums").alias("c_nums"), "c_raw_n"))
    q = rec.select(pl.col("gi").alias("q_gi"), pl.col("raw_name").alias("q_raw"), pl.col("core").alias("q_core"),
                   pl.col("name").alias("q_name"), pl.col("nums").alias("q_nums"), pl.col("raw_addr").alias("q_addr"))
    return keys.join(q, on="q_gi", how="left").join(c, on="s1_gi", how="left")


def pair_features(split: str, keys: pl.DataFrame) -> pl.DataFrame:
    words = json.load(open(WORK / "stage4_words.json"))
    added = {w: i + 1 for i, w in enumerate(words["added"])}
    dropped = {w: i + 1 for i, w in enumerate(words["dropped"])}
    t = _texts(split, keys)
    rows = list(t.select("q_raw", "c_raw", "q_core", "c_core", "q_name", "q_nums", "c_nums", "q_addr").iter_rows())
    chunks = [rows[i:i + 20_000] for i in range(0, len(rows), 20_000)]
    with Pool(N_JOBS, initializer=_init, initargs=(added, dropped)) as pool:
        out = [r for part in pool.map(_pairs, chunks) for r in part]
    res = pl.DataFrame(out, schema=PAIR_RAW, orient="row").with_columns(pl.all().cast(pl.Int32))
    t = pl.concat([t.select("q_gi", "s1_gi", "c_raw_n"), res], how="horizontal")
    best_other = (pl.col("raw_ratio").sum().over("q_gi") - pl.col("raw_ratio")) / (pl.len().over("q_gi") - 1)
    return t.with_columns(pl.when(pl.len().over("q_gi") > 1).then(pl.col("raw_ratio") - best_other)
                            .otherwise(None).cast(pl.Float32).alias("raw_gap"))


def learn_words():
    """The words that records add to or drop from the Source-1 name most often (training pairs
    that stage 3 does not reject outright).  No labels are used."""
    sc = pl.read_parquet(WORK / "train_scores3.parquet").select("q_gi", "s1_gi", pl.col("p2").alias("p3"))
    rec = (pl.read_parquet(WORK / "train_norm.parquet", columns=["core", "name_native"])
             .with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32)))
    d = (sc.filter(pl.col("p3") > ZONE[0])
           .join(rec.select(pl.col("gi").alias("q_gi"), pl.col("core").alias("q"), "name_native"), on="q_gi")
           .join(rec.select(pl.col("gi").alias("s1_gi"), pl.col("core").alias("c")), on="s1_gi")
           .filter((pl.col("name_native") == 0) & (pl.col("q") != pl.col("c"))))
    added, dropped = Counter(), Counter()
    for qc, cc in d.select("q", "c").iter_rows():
        qw, cw = (qc or "").split(), (cc or "").split()
        a, left = _unmatched([x for x in qw if x not in cw], [y for y in cw if y not in qw])
        if a:
            added[a[0]] += 1
        if left:
            dropped[left[0]] += 1
    out = {"added": [w for w, n in added.most_common(N_WORDS) if n >= MIN_WORD],
           "dropped": [w for w, n in dropped.most_common(N_WORDS) if n >= MIN_WORD]}
    print(f"  added words with an id: {len(out['added'])}, e.g. {out['added'][:40]}")
    print(f"  dropped words with an id: {len(out['dropped'])}, e.g. {out['dropped'][:40]}")
    json.dump(out, open(WORK / "stage4_words.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------
# features, model, decoding rule
# ------------------------------------------------------------------------------------------
def build(split: str):
    with Timer(f"stage4: support between records, {split}"):
        sup = support(split)
        zone = sup.filter(pl.col("zone")).drop("zone")
        print(f"  {sup.height} scored pairs; open records {zone['q_gi'].n_unique()} with {zone.height} pairs")
        del sup
    with Timer(f"stage4: spelling features {split}"):
        pf = pair_features(split, zone.select("q_gi", "s1_gi"))
    with Timer(f"stage4: join the features of stages 2 and 3 ({split})"):
        keys = zone.select("q_gi").unique()
        base = pl.concat([pl.read_parquet(p, columns=["q_gi", "s1_gi"] + STAGE2_FEATURES).join(keys, on="q_gi", how="semi")
                          for p in sorted(glob.glob(str(WORK / f"{split}_feat" / "*.parquet")))])
        f3 = [c for c in STAGE3_FEATURES if c not in STAGE2_FEATURES]
        feat3 = pl.read_parquet(WORK / f"{split}_feat3.parquet", columns=["q_gi", "s1_gi"] + f3).join(keys, on="q_gi", how="semi")
        out = (zone.join(base, on=["q_gi", "s1_gi"], how="left").join(feat3, on=["q_gi", "s1_gi"], how="left")
                   .join(pf, on=["q_gi", "s1_gi"], how="left"))
        for t in TEXT:
            pt = pl.read_parquet(TEXT_DIR / f"text_scores_{split}_{t}.parquet").select(
                pl.col("q_gi").cast(pl.Int32), pl.col("s1_gi").cast(pl.Int32), pl.col("pt").cast(pl.Float32).alias(f"pt_{t}"))
            out = out.join(pt, on=["q_gi", "s1_gi"], how="left")
            print(f"  text model {t}: {out[f'pt_{t}'].null_count()} of {out.height} open pairs without a score")
            c = pl.col(f"pt_{t}")
            best = c.max().over("q_gi")
            second = c.sort(descending=True).slice(1, 1).first().over("q_gi")
            out = out.with_columns(
                (c - pl.when(c >= best).then(second).otherwise(best)).fill_null(c).cast(pl.Float32).alias(f"ptgap_{t}"),
                ((c >= 0.9).sum().over("s1_gi") - (c >= 0.9).cast(pl.UInt32)).cast(pl.Int16).alias(f"ptclaims_{t}"))
        out.select(["q_gi", "s1_gi"] + STAGE4_FEATURES).write_parquet(WORK / f"{split}_{FEAT}.parquet")


def _merge(split: str, keys: pl.DataFrame, p4: np.ndarray):
    """The scores so far with the open records replaced by their new probability."""
    new = keys.with_columns(pl.Series("p4", p4))
    sc = pl.read_parquet(WORK / f"{split}_{PREV}.parquet")
    sc = sc.join(new, on=["q_gi", "s1_gi"], how="left").with_columns(
        pl.coalesce("p4", "p2").alias("p2")).drop("p4")
    sc.write_parquet(WORK / f"{split}_{NOW}.parquet")


def train():
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id"])
    pos, fold = stage1.train_labels(norm)
    del norm
    data = pl.read_parquet(WORK / f"train_{FEAT}.parquet").join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(
        pl.col("y").fill_null(0))
    keys = data.select("q_gi", "s1_gi")
    qf = fold[data["q_gi"].to_numpy()]
    y = data["y"].to_numpy()
    p3 = data["p3"].to_numpy()
    X = models.to_matrix(data, STAGE4_FEATURES)
    del data
    fitted, p4 = [], np.zeros(len(y), dtype=np.float32)
    for f in (0, 1):
        m = qf == f
        with Timer(f"stage4: fit fold {f} on {int(m.sum())} pairs"):
            clf = lgb.LGBMClassifier(**PARAMS, n_jobs=N_JOBS, verbose=-1).fit(
                X[m], y[m], feature_name=STAGE4_FEATURES, categorical_feature=CATEGORICAL)
        fitted.append(clf)
    for f in (0, 1):                                    # out-of-fold: the other fold's model
        m = qf == f
        p4[m] = models.predict(fitted[1 - f], X[m])
    print(f"  open pairs {len(y)}, {y.mean():.3f} true;  log loss before {log_loss(y, np.clip(p3, 1e-6, 1 - 1e-6)):.4f}"
          f" -> after {log_loss(y, np.clip(p4, 1e-6, 1 - 1e-6)):.4f}")
    joblib.dump(fitted, WORK / MODEL)
    _merge("train", keys, p4)
    imp = sorted(zip(STAGE4_FEATURES, fitted[0].booster_.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(v for _, v in imp)
    print("  gain by feature (fold 0, % of total):", [(k, round(100 * v / tot, 2)) for k, v in imp[:30]])
    print("  gain of the new features:", [(k, round(100 * v / tot, 2)) for k, v in imp if k in SUP4 + PAIR4 + TEXT4])


def predict(split: str):
    fitted = joblib.load(WORK / MODEL)
    data = pl.read_parquet(WORK / f"{split}_{FEAT}.parquet")
    with Timer(f"stage4: predict {split}"):
        p4 = models.predict(fitted, models.to_matrix(data, STAGE4_FEATURES))
    _merge(split, data.select("q_gi", "s1_gi"), p4)


def tune():
    """Decoding rule on the stage-4 probability (word-swap veto applied), under the training
    prior and the simulated test prior; stage 3 is scored the same way for comparison."""
    from exp_trees import Evaluator
    from stage2 import reduce_argmax

    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id"] + swap.NORM_COLS)
    pos, _ = stage1.train_labels(norm)
    ev = Evaluator(norm, pos)
    words, synonyms = swap.country_words(norm), swap.load_synonyms()
    results = []
    for name, path in (("stage 3", f"train_{PREV}.parquet"), ("stage 4", f"train_{NOW}.parquet")):
        top = reduce_argmax(pl.read_parquet(WORK / path).select("q_gi", "s1_gi", "p2"))
        flagged = swap.flag(top.select("q_gi", "s1_gi"), norm, words, synonyms)
        top = top.join(flagged.filter(pl.col("swap")).select("q_gi", "s1_gi"), on=["q_gi", "s1_gi"], how="anti")
        for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
            for mg in (0.0, 0.1):
                if name == "stage 3" and (t, mg) != (0.75, 0.1):
                    continue
                a = ev.train_prior(top, "threshold", {"threshold": t, "margin": mg})
                b = ev.test_prior(top, "threshold", {"threshold": t, "margin": mg})
                results.append({"scores": name, "veto": True, "threshold": t, "margin": mg,
                                "train_prior": a["macro_f05"], "test_prior": b["macro_f05"],
                                "P": b["pair_precision"], "R": b["pair_recall"], "by_country": b["by_group"]})
                print(f"  {name} t={t:.2f} m={mg}: train prior {a['macro_f05']:.5f}  test prior {b['macro_f05']:.5f}  "
                      f"P {b['pair_precision']:.4f} R {b['pair_recall']:.4f} {b['by_group']}", flush=True)
    best = max((r for r in results if r["scores"] == "stage 4"), key=lambda r: r["test_prior"])
    print("BEST stage 4 under the test-like prior:", best)
    json.dump({"method": "threshold", **best}, open(WORK / f"decision{3 + PASS}.json", "w"), indent=1)
    json.dump(results, open(WORK / f"tune{3 + PASS}_results.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "words":
        learn_words()
    elif cmd == "features":
        build(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "predict":
        predict(sys.argv[2])
    elif cmd == "tune":
        tune()
