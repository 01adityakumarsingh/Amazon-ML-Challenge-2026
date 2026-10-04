import re
import unicodedata

_SPLIT = re.compile(r"[^\w\u0300-\u036f\u0900-\u0dff]+", re.UNICODE)   
_DIGITS = re.compile(r"\d+")

LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
    "llp", "lp", "plc", "sarl", "sas", "eurl", "sa", "gmbh", "and", "the", "of", "pte", "opc", "ltda",
}

def strip_accents(s: str) -> str:
    d = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in d if not ("\u0300" <= ch <= "\u036f"))

def tokens(s: str) -> list:
    s = strip_accents(s.lower().replace("&", " and ").replace("+", " and "))
    return [t for t in _SPLIT.split(s) if t]

def name_tokens(name: str) -> list:
    return [t for t in tokens(name) if t not in LEGAL]

def numbers(addr: str) -> list:
    out = []
    for m in _DIGITS.findall(addr):
        m = m.lstrip("0")
        if m:
            out.append(m)
    return out

def alpha_tokens(addr: str, min_len: int = 4) -> list:
    return [t for t in tokens(addr) if t.isalpha() and len(t) >= min_len and t != "null"]
