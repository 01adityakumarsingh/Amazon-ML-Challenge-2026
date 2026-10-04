import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from normalize import name_tokens, tokens, numbers, alpha_tokens, LEGAL
from transliterate import is_brahmic, transliterate_name

STOP_LEGAL = LEGAL - {"and", "the", "of"}

def _nonlatin(s):
    return any(ord(c) > 0x24F for c in s if c.isalpha())

def prep(df: pd.DataFrame) -> dict:
    """Per-record derived fields, aligned with df rows."""
    nt = [tuple(name_tokens(n)) for n in df.business_name.values]
    at = [tuple(alpha_tokens(a, 3)) for a in df.business_address.values]
    
    tl_np4 = []
    for n, t4 in zip(df.business_name.values, nt):
        if is_brahmic(n):
            tl = transliterate_name(n).split()
            tl_np4.append(frozenset(w[:4] for w in tl if len(w) >= 3))
        else:
            tl_np4.append(frozenset(x[:4] for x in t4))
    return {
        "nt": nt,
        "np4": [frozenset(t[:4] for t in x) for x in nt],
        "tl_np4": tl_np4,
        "compact": ["".join(x) for x in nt],
        "legal": [frozenset(t for t in tokens(n) if t in STOP_LEGAL) for n in df.business_name.values],
        "nums": [frozenset(numbers(a)) for a in df.business_address.values],
        "at": [frozenset(x) for x in at],
        "ap4": [frozenset(t[:4] for t in x) for x in at],
        "nonlatin": np.array([_nonlatin(n) for n in df.business_name.values]),
        "name_txt": [" ".join(x) for x in nt],
        "addr_txt": [" ".join(tokens(a)) for a in df.business_address.values],
    }

def _rowcos(A, B, ia, ib, chunk=400_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out

def _jac(a, b):
    if not a or not b:
        return -1.0, -1.0
    i = len(a & b)
    return i / len(a | b), i / min(len(a), len(b))

def fit_vectorizers(p1: dict, samples: list, step: int = 1):
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3, sublinear_tf=True, dtype=np.float32)
    va = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3, sublinear_tf=True, dtype=np.float32)
    vn.fit(p1["name_txt"][::step] + sum((s["name_txt"] for s in samples), []))
    va.fit(p1["addr_txt"][::step] + sum((s["addr_txt"] for s in samples), []))
    return vn, va

FEATURES = [
    "name_cos", "addr_cos", "n_jac", "n_contain", "n_p4_jac", "n_p4_contain", "n_first_eq", "n_first_p4_eq", "n_tl_p4_jac", "n_tl_p4_contain",
    "n_lenratio", "compact_pre_eq", "compact_contain", "legal_conflict", "legal_b_empty",
    "a_num_jac", "a_num_any", "a_num_first", "a_p4_jac", "a_p4_contain", "a_tok_jac",
    "b_addr_empty", "b_nonlatin", "a_nnums", "b_nnums", "b_ntok",
    "score", "nkeys", "rank", "score_gap_b", "score_ratio_b", "cnt_b", "src",
]

def pair_features(p1, pk, X1, Xk, vecs, cand: pd.DataFrame, pos1: pd.Series, posk: pd.Series, src: int):
    """cand columns: s1, other, score, nkeys, rank. pos1/posk map id -> row position."""
    vn, va = vecs
    ia = pos1.reindex(cand.s1.values).values; ib = posk.reindex(cand.other.values).values
    N1n, N1a = X1; Nkn, Nka = Xk
    f = {}
    f["name_cos"] = _rowcos(N1n, Nkn, ia, ib)
    f["addr_cos"] = _rowcos(N1a, Nka, ia, ib)
    cols = {k: np.empty(len(ia), dtype=np.float32) for k in FEATURES[2:26]}
    nt1, nt2 = p1["nt"], pk["nt"]
    for r, (a, b) in enumerate(zip(ia, ib)):
        ta, tb = nt1[a], nt2[b]
        j, c = _jac(frozenset(ta), frozenset(tb)); cols["n_jac"][r] = j; cols["n_contain"][r] = c
        j, c = _jac(p1["np4"][a], pk["np4"][b]); cols["n_p4_jac"][r] = j; cols["n_p4_contain"][r] = c
        j, c = _jac(p1["tl_np4"][a], pk["tl_np4"][b]); cols["n_tl_p4_jac"][r] = j; cols["n_tl_p4_contain"][r] = c
        cols["n_first_eq"][r] = (ta[:1] == tb[:1]) if ta and tb else -1
        cols["n_first_p4_eq"][r] = (ta[0][:4] == tb[0][:4]) if ta and tb else -1
        cols["n_lenratio"][r] = min(len(ta), len(tb)) / max(len(ta), len(tb), 1)
        ca, cb = p1["compact"][a], pk["compact"][b]
        cols["compact_pre_eq"][r] = (ca[:6] == cb[:6]) if len(ca) >= 6 and len(cb) >= 6 else -1
        cols["compact_contain"][r] = (ca in cb or cb in ca) if len(ca) >= 5 and len(cb) >= 5 else -1
        la, lb = p1["legal"][a], pk["legal"][b]
        cols["legal_conflict"][r] = (bool(la) and bool(lb) and not (la & lb))
        cols["legal_b_empty"][r] = not lb
        na, nb = p1["nums"][a], pk["nums"][b]
        j, _ = _jac(na, nb); cols["a_num_jac"][r] = j
        cols["a_num_any"][r] = (len(na & nb) > 0) if na and nb else -1
        cols["a_num_first"][r] = -1
        j, c = _jac(p1["ap4"][a], pk["ap4"][b]); cols["a_p4_jac"][r] = j; cols["a_p4_contain"][r] = c
        j, _ = _jac(p1["at"][a], pk["at"][b]); cols["a_tok_jac"][r] = j
        cols["b_addr_empty"][r] = (not nb) and (not pk["at"][b])
        cols["b_nonlatin"][r] = pk["nonlatin"][b]
        cols["a_nnums"][r] = len(na); cols["b_nnums"][r] = len(nb); cols["b_ntok"][r] = len(tb)
    f.update(cols)
    df = pd.DataFrame(f)
    df["score"] = cand.score.values; df["nkeys"] = cand.nkeys.values; df["rank"] = cand["rank"].values
    mx = cand.groupby("other").score.transform("max").values
    df["score_gap_b"] = cand.score.values - mx
    df["score_ratio_b"] = cand.score.values / np.maximum(mx, 1e-6)
    df["cnt_b"] = cand.groupby("other").s1.transform("size").values
    df["src"] = src
    return df[FEATURES].astype(np.float32)
