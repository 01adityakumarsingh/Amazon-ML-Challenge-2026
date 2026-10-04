import re
import unicodedata

BLOCKS = {
    "Devanagari": 0x0900, "Bengali": 0x0980, "Gurmukhi": 0x0A00, "Gujarati": 0x0A80,
    "Oriya": 0x0B00, "Tamil": 0x0B80, "Telugu": 0x0C00, "Kannada": 0x0C80, "Malayalam": 0x0D00,
}
_BLOCK_RANGES = [(b, b + 0x7F, b) for b in BLOCKS.values()]

_DEV = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "r",
    0x0F: "e", 0x10: "ai", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x32: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "r",
    0x47: "e", 0x48: "ai", 0x4B: "o", 0x4C: "au",
}
_VOWEL_SIGN_OFFSETS = {0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x47, 0x48, 0x4B, 0x4C}
_VIRAMA_OFFSET = 0x4D
_CONSONANT_OFFSETS = set(range(0x15, 0x3A)) - {0x1D + 1} 


def _to_devanagari_offset(ch: str):
    cp = ord(ch)
    for lo, hi, base in _BLOCK_RANGES:
        if lo <= cp <= hi:
            return cp - base  
    return None


def is_brahmic(s: str) -> bool:
    return any(_to_devanagari_offset(ch) is not None for ch in s)


def transliterate_word(word: str) -> str:
    out = []
    offsets = [_to_devanagari_offset(ch) for ch in word]
    i, n = 0, len(word)
    while i < n:
        off = offsets[i]
        if off is None:
            out.append(word[i].lower()); i += 1; continue
        if off in _CONSONANT_OFFSETS and off in _DEV:
            out.append(_DEV[off])
            j = i + 1
            if j < n and offsets[j] == _VIRAMA_OFFSET:
                i = j + 1; continue  # explicit vowel suppression
            if j < n and offsets[j] in _VOWEL_SIGN_OFFSETS and offsets[j] in _DEV:
                out.append(_DEV[offsets[j]]); i = j + 1; continue
            out.append("a")  # inherent vowel
            i += 1; continue
        if off in _DEV:
            out.append(_DEV[off]); i += 1; continue
        i += 1
    return "".join(out)


_WORD_SPLIT = re.compile(r"\s+")

def transliterate_name(name: str) -> str:
    return " ".join(transliterate_word(w) for w in _WORD_SPLIT.split(name.strip()) if w)
