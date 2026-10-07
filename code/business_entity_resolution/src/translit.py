"""Learn Indic-script -> Latin mappings from the *training* ground truth.

Source 2 / Source 3 often render an Indian business name (and the state in the
address) in a native script (Devanagari, Tamil, Kannada, ...), while Source 1
is always Latin.  Because the native rendering is a word-by-word
transliteration, aligning the tokens of matched training pairs gives a
high-precision token dictionary.  Unknown tokens fall back to `unidecode`.
No external data is used.
"""
from __future__ import annotations

import collections
import json
import re

from normalize import (INDIC_RE, LEGAL_CANON, _canon_state, _name_tokens, has_indic,
                       to_ascii)

_STRIP = ".,()[]"


def _latin_tokens(name: str) -> list[str]:
    toks = _name_tokens(to_ascii(name).lower())
    return [LEGAL_CANON.get(t, t) for t in toks]


def learn(pairs, min_count: int = 2, min_share: float = 0.5):
    """pairs: iterable of (s1_name, s1_addr, country, other_name, other_addr)."""
    tok_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    comp_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for s1_name, s1_addr, country, o_name, o_addr in pairs:
        cty = (country or "").lower()
        if o_name and has_indic(o_name):
            nat = [t.strip(_STRIP) for t in o_name.split()]
            nat = [t for t in nat if t]
            lat = _latin_tokens(s1_name or "")
            if len(nat) == len(lat):
                for a, b in zip(nat, lat):
                    if INDIC_RE.search(a) and b:
                        tok_counts[a][b] += 1
        if o_addr and has_indic(o_addr) and s1_addr:
            s1_state = ""
            for c in s1_addr.split(","):
                st = _canon_state(c, cty, None)
                if st:
                    s1_state = st
                    break
            if s1_state:
                for c in o_addr.split(","):
                    c = c.strip()
                    if INDIC_RE.search(c):
                        comp_counts[c][s1_state] += 1
    translit = {}
    for a, cnt in tok_counts.items():
        b, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if n >= min_count and n / tot >= min_share:
            translit[a] = b
    states = {}
    for c, cnt in comp_counts.items():
        b, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if n >= min_count and n / tot >= 0.8:
            states[c] = b
    return translit, states


def save(translit: dict, states: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"translit": translit, "states": states}, f, ensure_ascii=False)


def load(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["translit"], d["states"]
