"""Text normalisation for business names and addresses.

Everything here is deterministic, country-aware where a country has known
conventions (US / India / France), and falls back to generic behaviour for any
other country label so the pipeline treats `country` as an open set.
"""
from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# Administrative regions (state / region) -> canonical code
# --------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "guam": "gu", "virgin islands": "vi",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "ladakh": "la", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "andaman and nicobar islands": "an",
    "andaman & nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
IN_CODE_ALIASES = {"ct": "cg", "or": "od", "ts": "tg", "ut": "uk", "uk": "uk", "dl": "dl"}

# Metropolitan French departments -> region (general geographic knowledge, used
# only to reconcile "department" vs "region" spellings of the same area).
FR_REGIONS = {
    "auvergne-rhone-alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire",
                             "haute-loire", "puy-de-dome", "rhone", "savoie", "haute-savoie"],
    "bourgogne-franche-comte": ["cote-d'or", "doubs", "jura", "nievre", "haute-saone",
                                "saone-et-loire", "yonne", "territoire de belfort"],
    "bretagne": ["cotes-d'armor", "finistere", "ille-et-vilaine", "morbihan"],
    "centre-val de loire": ["cher", "eure-et-loir", "indre", "indre-et-loire", "loir-et-cher",
                            "loiret"],
    "corse": ["corse-du-sud", "haute-corse"],
    "grand est": ["ardennes", "aube", "marne", "haute-marne", "meurthe-et-moselle", "meuse",
                  "moselle", "bas-rhin", "haut-rhin", "vosges"],
    "hauts-de-france": ["aisne", "nord", "oise", "pas-de-calais", "somme"],
    "ile-de-france": ["paris", "seine-et-marne", "yvelines", "essonne", "hauts-de-seine",
                      "seine-saint-denis", "val-de-marne", "val-d'oise"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine-maritime"],
    "nouvelle-aquitaine": ["charente", "charente-maritime", "correze", "creuse", "dordogne",
                           "gironde", "landes", "lot-et-garonne", "pyrenees-atlantiques",
                           "deux-sevres", "vienne", "haute-vienne"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute-garonne", "gers", "herault", "lot",
                  "lozere", "hautes-pyrenees", "pyrenees-orientales", "tarn", "tarn-et-garonne"],
    "pays de la loire": ["loire-atlantique", "maine-et-loire", "mayenne", "sarthe", "vendee"],
    "provence-alpes-cote d'azur": ["alpes-de-haute-provence", "hautes-alpes", "alpes-maritimes",
                                   "bouches-du-rhone", "var", "vaucluse"],
}


def _key(s: str) -> str:
    """Canonical lookup key for an admin-region string: ascii, lower, no punctuation."""
    s = unidecode(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _build_state_maps():
    maps = {"us": {}, "india": {}, "france": {}}
    for name, code in US_STATES.items():
        maps["us"][_key(name)] = code
        maps["us"][code] = code
    for name, code in IN_STATES.items():
        maps["india"][_key(name)] = code
        maps["india"][code] = code
    for alias, code in IN_CODE_ALIASES.items():
        maps["india"][alias] = code
    for region, deps in FR_REGIONS.items():
        rk = _key(region)
        maps["france"][rk] = rk
        for d in deps:
            maps["france"][_key(d)] = rk
    maps["france"]["idf"] = _key("ile-de-france")
    maps["france"]["paca"] = _key("provence-alpes-cote d'azur")
    return maps


STATE_MAPS = _build_state_maps()

# State names that are also common city / county names: lowest priority when an
# address contains several state-like components.
AMBIGUOUS_STATE_NAMES = {"washington", "newyork", "delhi", "newdelhi", "nord", "paris",
                         "vienne", "loire", "marne", "manche", "eure", "cher", "indre", "lot",
                         "var", "jura", "rhone", "ain", "orne", "gard", "tarn", "aude"}

# Tokens that are *not* states even if they collide with a code in that country
# (they are common address words).  Only relevant for bare 2-letter codes.
_STATE_CODE_BLOCK = {"india": {"ga", "la", "as", "or", "ch", "ct", "ut", "uk", "an", "ld"}}

# --------------------------------------------------------------------------
# Name normalisation
# --------------------------------------------------------------------------
LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co", "compagnie": "co", "cie": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lmtd": "ltd", "ld": "ltd",
    "pvt": "pvt", "private": "pvt", "pvte": "pvt", "prv": "pvt", "prvt": "pvt",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "pa": "pa", "lllp": "lllp", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "ei": "ei", "eirl": "eirl", "scop": "scop", "scp": "scp", "selarl": "selarl",
    "gie": "gie", "scm": "scm", "sca": "sca", "ets": "ets", "etablissements": "ets",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}
LEGAL_TOKENS = set(LEGAL_CANON.values())
NAME_STOP = {"the", "and", "of", "a", "an", "de", "du", "des", "la", "le", "les", "l", "d",
             "et", "en", "au", "aux", "for", "by", "at", "on", "in",
             # honorifics that the noise process prepends to names
             "dr", "mr", "mrs", "ms", "smt", "messrs", "mssrs"}
# identifiers / phone numbers appended to names: "(ID: 98610)", "#12750", "- 8923189383"
NAME_NOISE_RE = re.compile(r"\(?\b[il1]d\s*[:#]?\s*\d+\)?|#\s*\d+|\b\d{6,}\b|\bm\s*/\s*s\b\.?")
DBA_RE = re.compile(r"\b(?:doing business as|d\s*/\s*b\s*/\s*a|dba|also known as|aka|"
                    r"trading as|t\s*/\s*a|formerly known as|fka)\b", re.I)
DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|"
                       r"in|fr|biz|info|co|us|io|shop|store|online)\b", re.I)
DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.\s?)+[a-z]\b\.?|\b(?:[a-z]\.\s?){2,}", re.I)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
                      "@": "a", "$": "s"})
INDIC_RE = re.compile(r"[ऀ-෿]")
HAS_ALPHA_RE = re.compile(r"[a-z]")


def has_indic(s: str | None) -> bool:
    return bool(s) and INDIC_RE.search(s) is not None


def to_ascii(s: str) -> str:
    if s.isascii():
        return s
    return unidecode(unicodedata.normalize("NFC", s))


def _collapse_acronyms(s: str) -> str:
    return DOTTED_ACRONYM_RE.sub(lambda m: re.sub(r"[.\s]", "", m.group(0)) + " ", s)


def _name_tokens(s: str) -> list[str]:
    """ascii-lowercase string -> list of normalised word tokens."""
    s = _collapse_acronyms(s)
    s = s.replace("'s ", "s ").replace("'", "")
    s = s.replace("&", " and ").replace("+", " and ")
    toks = []
    for t in NON_ALNUM_RE.split(s):
        if not t:
            continue
        if HAS_ALPHA_RE.search(t) and not t.isalpha():
            # leet-speak / OCR digits inside words: Onc0logy, 5olutions, DENTA1
            if not re.fullmatch(r"\d+(st|nd|rd|th)", t):
                t = t.translate(LEET)
        toks.append(t)
    return toks


def split_name(raw: str, translit: dict | None = None):
    """Return (primary_name, alias_names, domain_stem) as ascii-lower strings."""
    s = raw.strip()
    if translit and has_indic(s):
        s = transliterate_tokens(s, translit)
    s = to_ascii(s).lower()
    domain = ""
    m = DOMAIN_RE.search(s)
    if m:
        domain = m.group(1).replace("-", "")
        s = (s[:m.start()] + " " + s[m.end():]).strip(" |-")
    s = NAME_NOISE_RE.sub(" ", s)
    if not domain:
        # domain written without the dot: "gnacademycom", "innovativeagrocom"
        w = s.strip(" .-|")
        if " " not in w and len(w) >= 8 and w.endswith("com") and w.isalnum():
            domain, s = w[:-3], ""
    parts = [p for p in re.split(r"\s*\|\s*", s) if p.strip()]
    s = parts[0] if parts else ""
    aliases = parts[1:]
    dba = DBA_RE.split(s)
    if len(dba) > 1:
        s = dba[0]
        aliases += dba[1:]
    return s, aliases, domain


def normalize_name(raw: str | None, translit: dict | None = None) -> dict:
    """Normalise a business name.

    Returns dict with
        tokens  : all normalised tokens (legal forms canonicalised, stopwords removed)
        core    : tokens without legal forms
        legal   : sorted legal-form tokens
        alias   : core tokens of any DBA / alias name
        domain  : domain stem if the name contained a URL
        concat  : core tokens concatenated (for domain comparison)
    """
    if raw is None:
        raw = ""
    primary, aliases, domain = split_name(raw, translit)
    toks = [LEGAL_CANON.get(t, t) for t in _name_tokens(primary)]
    toks = [t for t in toks if t not in NAME_STOP]
    core = [t for t in toks if t not in LEGAL_TOKENS]
    legal = sorted({t for t in toks if t in LEGAL_TOKENS})
    alias = []
    for a in aliases:
        at = [LEGAL_CANON.get(t, t) for t in _name_tokens(a)]
        alias += [t for t in at if t not in NAME_STOP and t not in LEGAL_TOKENS]
    if not core and not alias and domain:
        # name is only a URL: use the domain stem as the (single) core token
        core = [domain]
    if not core and toks:
        core = [t for t in toks]
    return {
        "tokens": toks, "core": core, "legal": legal, "alias": alias, "domain": domain,
        "concat": "".join(core),
    }


# --------------------------------------------------------------------------
# Transliteration of Indic-script tokens (dictionary learned from train pairs)
# --------------------------------------------------------------------------
def transliterate_tokens(s: str, translit: dict) -> str:
    out = []
    for tok in s.split():
        if INDIC_RE.search(tok):
            key = tok.strip(".,()[]")
            out.append(translit.get(key) or unidecode(tok))
        else:
            out.append(tok)
    return " ".join(out)


# --------------------------------------------------------------------------
# Address normalisation
# --------------------------------------------------------------------------
ADDR_CANON = {
    # street types (US / India / generic)
    "street": "st", "st": "st", "str": "st", "stre": "st", "saint": "st", "ste": "ste",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "drv": "dr",
    "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave", "aven": "ave", "avenu": "ave",
    "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "bd": "blvd", "bvd": "blvd",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "crt": "ct", "circle": "cir",
    "cir": "cir", "place": "pl", "pl": "pl", "plz": "plz", "plaza": "plz",
    "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy", "trail": "trl",
    "trl": "trl", "terrace": "terr", "terr": "terr", "square": "sq", "sq": "sq",
    "way": "way", "wy": "way", "expressway": "expy", "expy": "expy", "freeway": "fwy",
    "fwy": "fwy", "turnpike": "tpke", "tpke": "tpke", "crossing": "xing", "xing": "xing",
    "point": "pt", "pt": "pt", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "heights": "hts", "hts": "hts", "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "north": "n", "n": "n", "south": "s", "s": "s", "so": "s", "east": "e", "e": "e",
    "west": "w", "w": "w", "northeast": "ne", "ne": "ne", "northwest": "nw", "nw": "nw",
    "southeast": "se", "se": "se", "southwest": "sw", "sw": "sw",
    "apartment": "apt", "apt": "apt", "suite": "ste", "unit": "unit", "floor": "fl",
    "fl": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "bldng": "bldg",
    "room": "rm", "rm": "rm", "department": "dept", "dept": "dept",
    # India
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "marg", "sector": "sec",
    "sec": "sec", "sect": "sec", "phase": "ph", "ph": "ph", "colony": "colony", "col": "colony",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "behind": "bh", "beh": "bh",
    "ground": "gr", "grd": "gr", "gf": "gf", "cross": "cross", "crs": "cross", "main": "main",
    "block": "blk", "blk": "blk", "layout": "layout", "extension": "extn", "extn": "extn",
    "ext": "extn", "industrial": "indl", "indl": "indl", "ind": "indl", "estate": "est",
    "est": "est", "complex": "cmplx", "cmplx": "cmplx", "tower": "twr", "towers": "twr",
    "twr": "twr", "chowk": "chowk", "bazar": "bazaar", "bazaar": "bazaar", "gali": "gali",
    "village": "vill", "vill": "vill", "vpo": "vill", "post": "po", "po": "po", "tehsil": "teh",
    "teh": "teh", "taluk": "tq", "taluka": "tq", "tq": "tq", "district": "dist", "dist": "dist",
    "distt": "dist",
    # France
    "rue": "rue", "r": "rue", "route": "rte", "rte": "rte", "impasse": "imp", "imp": "imp",
    "allee": "all", "allees": "all", "all": "all", "chemin": "chem", "chem": "chem", "che": "chem",
    "ch": "chem", "bld": "blvd", "boulv": "blvd", "sq": "sq", "pce": "pl", "plc": "pl",
    "quai": "quai", "qu": "quai", "cours": "crs", "faubourg": "fbg", "fbg": "fbg", "fg": "fbg",
    "passage": "pass", "pass": "pass", "sente": "sente", "residence": "res", "res": "res",
    "batiment": "bat", "bat": "bat", "appartement": "apt", "appt": "apt", "bis": "bis",
    "ter": "ter", "sainte": "ste", "lieu-dit": "ld", "lieudit": "ld", "rond-point": "rpt",
    "rpt": "rpt", "promenade": "prom", "prom": "prom", "esplanade": "esp", "esp": "esp",
    "hameau": "ham", "ham": "ham", "zone": "zone", "za": "za", "zi": "zi", "zac": "zac",
    # city aliases
    "bengaluru": "bangalore", "bombay": "mumbai", "gurugram": "gurgaon", "madras": "chennai",
    "calcutta": "kolkata", "mysuru": "mysore", "hubballi": "hubli", "belagavi": "belgaum",
    "kalaburagi": "gulbarga", "vadodara": "baroda", "puducherry": "pondicherry",
    "thiruvananthapuram": "trivandrum", "kochi": "cochin", "prayagraj": "allahabad",
}
# address tokens that carry little identifying information
ADDR_WEAK = {"no", "number", "door", "h", "hno", "house", "plot", "flat", "unit", "apt", "ste",
             "fl", "bldg", "rm", "po", "box", "pob", "nr", "opp", "bh", "and", "the", "of", "de",
             "du", "des", "la", "le", "les", "l", "d", "c", "o", "co", "null", "none", "na",
             "shop", "office", "off", "premises", "gr", "gf", "first", "second", "third",
             "dno", "kh", "khasra", "survey", "sy", "sno", "ward", "wno", "bis", "ter", "et",
             "chez", "mr", "mme", "m", "at", "in", "a", "b", "n", "s", "e", "w"}
ADDR_SPLIT_RE = re.compile(r"[a-z]+|\d+")
ORDINAL_WORDS = {w: str(i + 1) for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
    "twentieth".split())}
ORDINAL_WORD_RE = re.compile(r"\b(?:" + "|".join(ORDINAL_WORDS) + r")\b")
ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th|er|eme|e)\b")
NULL_COMP = {"null", "none", "nan", "n/a", "na", ""}


def _canon_state(comp_raw: str, country: str, native_states: dict | None):
    """If an address component is a state/region return its canonical code, else None."""
    c = comp_raw.strip()
    if native_states and has_indic(c):
        v = native_states.get(c)
        if v:
            return v
    cmap = STATE_MAPS.get(country)
    if cmap is None:
        return None
    k = _key(c)
    if not k:
        return None
    v = cmap.get(k)
    if v is None:
        return None
    if len(k) == 2 and k in _STATE_CODE_BLOCK.get(country, ()):
        # bare ambiguous two-letter code that is also a common word: accept only
        # if it is the whole component in upper case (e.g. "GA" not "ga")
        if not c.isupper():
            return None
    return v


def normalize_address(raw: str | None, country: str, native_states: dict | None = None,
                      translit: dict | None = None) -> dict:
    """Normalise an address.

    Returns dict with
        tokens  : word tokens (canonicalised abbreviations), state removed
        strong  : tokens minus weak/stop tokens
        nums    : numeric tokens (leading zeros stripped), in order of appearance
        state   : canonical state/region code or ''
        empty   : 1 if the address is missing
    """
    country = (country or "").strip().lower()
    if raw is None or raw.strip().lower() in NULL_COMP:
        return {"tokens": [], "strong": [], "nums": [], "state": "", "state_alt": "", "empty": 1}
    comps = [c for c in raw.split(",")]
    found = []  # (is_ambiguous, is_full_name, -position, code, component)
    kept = []
    for i, c in enumerate(comps):
        cs = c.strip()
        if cs.lower() in NULL_COMP:
            continue
        st = _canon_state(cs, country, native_states)
        if st is not None:
            k = _key(cs)
            found.append((k in AMBIGUOUS_STATE_NAMES, len(k) > 3, -i, st, cs))
            continue
        kept.append(cs)
    state = state_alt = ""
    if found:
        # Preference: unambiguous > ambiguous, 2-letter code > full name, later > earlier.
        # A losing state-like component is usually a city named after a state
        # ("Delaware, OH", "Indiana, PA", "Washington, DC"): keep it as address text.
        found.sort()
        state = found[0][3]
        for f in found[1:]:
            if f[3] != state:
                kept.append(f[4])
                if not state_alt:
                    state_alt = f[3]
    s = " , ".join(kept)
    if translit and has_indic(s):
        s = transliterate_tokens(s, translit)
    s = to_ascii(s).lower()
    s = s.replace("n°", " no ").replace("nº", " no ").replace("no.", " no ")
    s = ORDINAL_WORD_RE.sub(lambda m: " " + ORDINAL_WORDS[m.group(0)] + " ", s)
    s = _collapse_acronyms(s)
    s = s.replace("'", " ")
    s = ORDINAL_RE.sub(r"\1 ", s)
    toks, nums = [], []
    for t in ADDR_SPLIT_RE.findall(s):
        if t.isdigit():
            n = t.lstrip("0") or "0"
            nums.append(n)
            continue
        toks.append(ADDR_CANON.get(t, t))
    strong = [t for t in toks if t not in ADDR_WEAK]
    return {"tokens": toks, "strong": strong, "nums": nums, "state": state,
            "state_alt": state_alt, "empty": 0}
