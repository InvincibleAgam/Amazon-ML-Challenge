"""Word-swap veto: a record whose name REPLACES a word of the Source-1 name by another full
name word is a different business, even at the same address.

The generator builds "sibling" distractors by keeping the address and most of the name and
swapping one word for another word of the name vocabulary (Pryor Gulf Gravity -> Morris Gulf
Gravity).  The string-similarity features cannot tell a swapped word from a mistyped one; in
the training countries the swapped word is usually the first word, which the model learned by
position, so in a country where the swapped word sits elsewhere the model accepts the sibling.
This module detects the swap itself, independent of position and country.

A record word counts as swapped in when
  * the Source-1 name has a word left that no record word spells, mistypes or shortens,
  * the record word has at least MIN_LEN letters and occurs in at least MIN_NAMES Source-1
    names of the country (a real word of the name vocabulary, not a typo or an invented name),
  * it is neither a typo nor a shortening of a Source-1 word, nor a spelling "correction" of
    the word left over (same initial and similar letters: corner -> comer),
  * it is not a generic word that records append after the legal form (services, center /
    fils, groupe ...; found per country without labels),
  * the pair (word left over, record word) is not a synonym pair learned from the training
    labels (systems -> technologies).
Names in a non-Latin script are left alone: their words come from the transliteration.

On the accepted training pairs the veto removes 10 % of all false merges at a precision of
99 % (see `python swap.py`).

    python swap.py learn    # synonym pairs from the training labels -> work/swap_synonyms.json
    python swap.py          # effect of the veto on the training split (exact metric)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from common import WORK

MIN_LEN = 5           # letters of the swapped-in word
MIN_NAMES = 50        # Source-1 names of the country that contain it
TYPO_RATIO = 75       # rapidfuzz ratio from which two words count as the same word mistyped
CORRECTION_RATIO = 60  # ... or, with the same initial, as a spelling correction (corner -> comer)
GENERIC_SHARE = 0.25  # a word is generic if it is appended at >= this share of the top word's rate
SYNONYM_MIN = 20      # true training pairs needed for a learned synonym pair
NORM_COLS = ["country", "src", "name", "core", "legal", "name_native"]


def _shortening(x: str, y: str) -> bool:
    """x shortens y: same first letter and the letters of x appear in order in y."""
    if len(x) >= len(y) or x[0] != y[0]:
        return False
    it = iter(y)
    return all(ch in it for ch in x)


def _same(x: str, y: str) -> bool:
    return fuzz.ratio(x, y) >= TYPO_RATIO or _shortening(x, y) or _shortening(y, x)


def country_words(norm: pl.DataFrame) -> tuple[dict, dict]:
    """(vocabulary, generic) per country.  vocabulary[country][word] = Source-1 names holding
    the word; generic[country] = words that Source-2/3 names append after the legal form."""
    s1 = (norm.filter(pl.col("src") == 1)
              .select("country", pl.col("core").str.split(" ").list.unique().alias("w")).explode("w")
              .filter(pl.col("w").is_not_null() & (pl.col("w") != "")).group_by("country", "w").len())
    vocab: dict = {}
    for c, w, n in s1.iter_rows():
        vocab.setdefault(c, {})[w] = n
    d = (norm.filter((pl.col("src") != 1) & (pl.col("legal") != ""))
             .select("country", pl.col("name").str.split(" ").alias("t"), pl.col("legal").str.split(" ").alias("lg"),
                     pl.col("core").str.split(" ").alias("cr"))
             .with_columns(pl.col("t").list.get(-1, null_on_oob=True).alias("last"),
                           pl.col("t").list.get(-2, null_on_oob=True).alias("prev"))
             .filter(pl.col("last").is_not_null() & pl.col("prev").is_not_null()
                     & pl.col("cr").list.contains(pl.col("last")) & pl.col("lg").list.contains(pl.col("prev")))
             .group_by("country", "last").len())
    generic: dict = {}
    for c in d["country"].unique().to_list():
        g = d.filter(pl.col("country") == c)
        generic[c] = set(g.filter(pl.col("len") >= GENERIC_SHARE * g["len"].max())["last"].to_list())
    return vocab, generic


def swapped_words(q_only: list, c_only: list, vocab: dict, generic: set, synonyms: set) -> list[tuple[str, str]]:
    """(word left over in the Source-1 name, word swapped in by the record) pairs."""
    q_only = [x for x in (q_only or []) if x]
    c_only = [y for y in (c_only or []) if y]
    left = [y for y in c_only if not any(_same(x, y) for x in q_only)]
    if not left:
        return []
    out = []
    for x in q_only:
        if len(x) < MIN_LEN or x in generic or vocab.get(x, 0) < MIN_NAMES:
            continue
        if any(_same(x, y) for y in c_only):
            continue
        if any(x[0] == y[0] and fuzz.ratio(x, y) >= CORRECTION_RATIO for y in left):
            continue
        if any((y, x) in synonyms for y in left):
            continue
        out.append((left[0], x))
    return out


def describe(q_only: list, c_only: list, vocab: dict, generic: set, synonyms: set) -> tuple:
    """How the words of two names differ, as counts: (Source-1 words the record mistypes or
    shortens, Source-1 words left over, generic words added, synonyms, spelling corrections,
    swapped-in words, Source-1 names holding the most common swapped-in word, other new words)."""
    q_only = [x for x in (q_only or []) if x]
    c_only = [y for y in (c_only or []) if y]
    left = [y for y in c_only if not any(_same(x, y) for x in q_only)]
    n_gen = n_syn = n_corr = n_swap = n_new = top = 0
    for x in q_only:
        if any(_same(x, y) for y in c_only):
            continue
        if x in generic:
            n_gen += 1
        elif any((y, x) in synonyms for y in left):
            n_syn += 1
        elif any(x[0] == y[0] and fuzz.ratio(x, y) >= CORRECTION_RATIO for y in left):
            n_corr += 1
        elif left and len(x) >= MIN_LEN and vocab.get(x, 0) >= MIN_NAMES:
            n_swap += 1
            top = max(top, vocab.get(x, 0))
        else:
            n_new += 1
    return len(c_only) - len(left), len(left), n_gen, n_syn, n_corr, n_swap, top, n_new


def _char(ch: str) -> int:
    if "a" <= ch <= "z":
        return ord(ch) - 96
    return 27 + int(ch) if ch.isdigit() else 37


def _sound(w: str) -> str:
    """Rough sound of a word: letters that are pronounced alike are merged."""
    for a, b in (("ph", "f"), ("ck", "k"), ("qu", "kw"), ("x", "ks"), ("ce", "se"), ("ci", "si"), ("cy", "sy"),
                 ("c", "k"), ("q", "k"), ("z", "s"), ("y", "i")):
        w = w.replace(a, b)
    return "".join(ch for i, ch in enumerate(w) if i == 0 or ch != w[i - 1])


def edit(q_only: list, c_only: list) -> tuple:
    """The closest pair of words that differ by one or two letters, described by the edit that
    turns the Source-1 word into the record's word: (distance, edit code, second edit code,
    length of the Source-1 word, position, at start, at end, same sound, same consonants,
    a digit is involved).  Edit code = operation, letter removed, letter put in.  The generator
    respells a word to make a sibling (c -> k, s -> z, a vowel for another) and mistypes it in
    true copies (g -> 6, i -> l); a string similarity scores both the same."""
    best = None
    for x in (q_only or []):
        for y in (c_only or []):
            if not x or not y or max(len(x), len(y)) < 3:
                continue
            d = Levenshtein.distance(x, y, score_cutoff=2)
            if 0 < d <= 2 and (best is None or (d, -len(y)) < best[0]):
                best = ((d, -len(y)), x, y, d)
    if best is None:
        return (0,) * 10
    _, x, y, d = best
    codes = []
    ops = Levenshtein.editops(y, x)
    for o in ops[:2]:
        if o.tag == "replace":
            codes.append(1600 + 40 * _char(y[o.src_pos]) + _char(x[o.dest_pos]))
        elif o.tag == "insert":
            codes.append(3200 + _char(x[o.dest_pos]))
        else:
            codes.append(4800 + 40 * _char(y[o.src_pos]))
    codes += [0] * (2 - len(codes))
    pos = ops[0].src_pos
    vowels = str.maketrans("", "", "aeiou")
    sx, sy = _sound(x), _sound(y)
    return (d, codes[0], codes[1], len(y), pos, int(pos == 0), int(pos >= len(y) - 1), int(sx == sy),
            int(sx.translate(vowels) == sy.translate(vowels)), int(any(ch.isdigit() for ch in x + y)))


def load_synonyms() -> set:
    p = WORK / "swap_synonyms.json"
    return {tuple(s) for s in json.load(open(p))} if p.exists() else set()


def _swaps(pairs: pl.DataFrame, norm: pl.DataFrame, words, synonyms) -> tuple[pl.DataFrame, list]:
    vocab, generic = words or country_words(norm)
    rec = norm.select("country", "core", "name_native").with_row_index("gi").with_columns(pl.col("gi").cast(pl.Int32))
    q = rec.select(pl.col("gi").alias("q_gi"), pl.col("core").str.split(" ").alias("q"), pl.col("name_native").alias("q_native"))
    c = rec.select(pl.col("gi").alias("s1_gi"), pl.col("core").str.split(" ").alias("c"), "country")
    d = (pairs.select("q_gi", "s1_gi").join(q, on="q_gi", how="left").join(c, on="s1_gi", how="left")
              .with_columns(pl.col("q").list.set_difference(pl.col("c")).alias("q_only"),
                            pl.col("c").list.set_difference(pl.col("q")).alias("c_only")).with_row_index("_i"))
    todo = d.filter((pl.col("q_native") == 0) & (pl.col("q_only").list.len() > 0) & (pl.col("c_only").list.len() > 0))
    found = [(i, swapped_words(qo, co, vocab.get(cn, {}), generic.get(cn, set()), synonyms))
             for i, qo, co, cn in todo.select("_i", "q_only", "c_only", "country").iter_rows()]
    return d, [(i, s) for i, s in found if s]


def flag(pairs: pl.DataFrame, norm: pl.DataFrame, words=None, synonyms: set | None = None) -> pl.DataFrame:
    """pairs: (q_gi, s1_gi, ...).  norm: all records of the split in file order (row = gi) with
    NORM_COLS.  Returns pairs with a boolean column `swap`."""
    d, found = _swaps(pairs, norm, words, load_synonyms() if synonyms is None else synonyms)
    swap = np.zeros(d.height, dtype=bool)
    swap[np.asarray([i for i, _ in found], dtype=np.int64)] = True
    return pairs.with_columns(pl.Series("swap", swap))


def _train():
    import stage1
    norm = pl.read_parquet(WORK / "train_norm.parquet", columns=["entity_id"] + NORM_COLS)
    pos, _ = stage1.train_labels(norm)
    return norm, pos


def learn():
    """Synonym pairs: word swaps that the training labels show to be the same business."""
    norm, pos = _train()
    _, found = _swaps(pos, norm, None, set())
    n: dict = {}
    for _, s in found:
        for p in set(s):
            n[p] = n.get(p, 0) + 1
    syn = sorted(p for p, k in n.items() if k >= SYNONYM_MIN)
    print(f"{len(found)} true training pairs carry a swapped word; synonym pairs (>= {SYNONYM_MIN} pairs):")
    for p in sorted(syn, key=lambda p: -n[p]):
        print(f"  {p[0]} -> {p[1]}: {n[p]}")
    json.dump(syn, open(WORK / "swap_synonyms.json", "w"), indent=1)


def main():
    from evaluate import macro_f05
    from stage2 import decode

    norm, pos = _train()
    ids = norm.select("entity_id", "src", "country").with_row_index("gi")
    s1 = ids.filter(pl.col("src") == 1)
    gid = ids.select(pl.col("gi").cast(pl.Int32), "entity_id")
    qmap, smap = gid.rename({"gi": "q_gi", "entity_id": "q_id"}), gid.rename({"gi": "s1_gi", "entity_id": "s1_id"})
    truth = pos.join(qmap, on="q_gi").join(smap, on="s1_gi")
    by = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("group"))
    words = country_words(norm)
    print("generic words appended after the legal form:", {c: sorted(g) for c, g in words[1].items()})
    print("synonym pairs:", sorted(load_synonyms()))
    rule = json.load(open(WORK / "decision_testprior.json"))
    acc = decode(pl.read_parquet(WORK / "train_scores.parquet").select("q_gi", "s1_gi", "p2"), rule)
    f = flag(acc, norm, words).join(pos, on=["q_gi", "s1_gi"], how="left").with_columns(pl.col("y").fill_null(0))
    sw = f.filter(pl.col("swap"))
    print(f"accepted pairs {f.height}; flagged as swap {sw.height} ({100 * sw.height / f.height:.3f} %), "
          f"of which true matches {int(sw['y'].sum())} ({100 * sw['y'].mean():.2f} %)")
    print(f"false merges in total {int((1 - f['y']).sum())}; removed by the veto {int((1 - sw['y']).sum())} "
          f"({100 * (1 - sw['y']).sum() / (1 - f['y']).sum():.1f} %)")
    for name, pairs in (("rule alone", acc), ("rule + swap veto", f.filter(~pl.col("swap")))):
        r = macro_f05(pairs.join(qmap, on="q_gi").join(smap, on="s1_gi"), truth, s1["entity_id"], by)
        print(f"  {name:18s} macro F0.5 = {r['macro_f05']:.5f}  singles {r['singleton_acc']:.4f}  "
              f"P {r['pair_precision']:.4f}  R {r['pair_recall']:.4f}  {r['by_group']}")


if __name__ == "__main__":
    learn() if sys.argv[1:] == ["learn"] else main()
