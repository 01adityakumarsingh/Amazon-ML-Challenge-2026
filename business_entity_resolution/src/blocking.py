import zlib
import numpy as np
import pandas as pd
from normalize import name_tokens, numbers, alpha_tokens
from transliterate import is_brahmic, transliterate_name

KT_ADDR, KT_NAME1, KT_NAME2, KT_ADDR2, KT_NAME3, KT_NAME_TL = 0, 1, 2, 3, 4, 5

def _h(country: str, *parts) -> int:
    b = "\x1f".join((country,) + parts).encode("utf-8")
    return ((zlib.crc32(b) << 30) ^ zlib.adler32(b) ^ (zlib.crc32(b[::-1]) << 5)) & 0x3FFFFFFFFFFFFFFF

def record_keys(country, name, addr, max_nums=6, max_alpha=8):
    keys = []
    nums = list(dict.fromkeys(numbers(addr)))[:max_nums]
    alph = list(dict.fromkeys(a[:4] for a in alpha_tokens(addr)))[:max_alpha]
    for n in nums:
        for a in alph:
            keys.append((_h(country, "A", n, a), KT_ADDR))
    nt = list(dict.fromkeys(t for t in name_tokens(name) if len(t) >= 3 and t.isalpha()))[:5]
    for t in nt:
        keys.append((_h(country, "N1", t[:5]), KT_NAME1))
    p = [t[:4] for t in nt[:4]]
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            a, b = sorted((p[i], p[j]))
            keys.append((_h(country, "N2", a, b), KT_NAME2))
    ap = list(dict.fromkeys(a[:4] for a in alpha_tokens(addr)))[:6]
    for i in range(len(ap)):
        for j in range(i + 1, len(ap)):
            x, y = sorted((ap[i], ap[j]))
            keys.append((_h(country, "A2", x, y), KT_ADDR2))
    compact = "".join(t for t in name_tokens(name) if t.isalnum())
    if compact.endswith("com"):
        compact = compact[:-3]
    if len(compact) >= 6:
        keys.append((_h(country, "N3", compact[:7]), KT_NAME3))
    if is_brahmic(name):
        tl_tokens = list(dict.fromkeys(t for t in transliterate_name(name).split() if len(t) >= 3))[:5]
        for t in tl_tokens:
            keys.append((_h(country, "N1", t[:5]), KT_NAME1))
            keys.append((_h(country, "NT", t[:5]), KT_NAME_TL))
    return keys

def key_table(df: pd.DataFrame, ids: np.ndarray, block: int = 200_000, log=None) -> pd.DataFrame:
    outs = []
    for s in range(0, len(df), block):
        if log and s > 0:
            log(f"  key_table: {s:,}/{len(df):,} records processed...")
        sub = df.iloc[s:s + block]; sid = ids[s:s + block]
        rid, kh, kt = [], [], []
        for i, (c, n, a) in enumerate(zip(sub.country.values, sub.business_name.values, sub.business_address.values)):
            for k, t in record_keys(c, n, a):
                rid.append(sid[i]); kh.append(k); kt.append(t)
        outs.append(pd.DataFrame({"rid": np.array(rid, dtype=np.int64), "kh": np.array(kh, dtype=np.int64),
                                  "kt": np.array(kt, dtype=np.int8)}).drop_duplicates())
    return pd.concat(outs, ignore_index=True)

def index_s1(k1: pd.DataFrame, max_s1_block: int = 40) -> pd.DataFrame:
    n1 = k1.groupby("kh").size().rename("n1")
    n1 = n1[n1 <= max_s1_block]
    return k1.merge(n1, left_on="kh", right_index=True).sort_values("kh", ignore_index=True)

def candidate_pairs_indexed(k1i: pd.DataFrame, kk: pd.DataFrame, max_pairs_per_key=4000, top_k=10,
                            weights=(3.0, 1.0, 1.5, 1.0, 2.0, 1.5)) -> pd.DataFrame:
    nk = kk.groupby("kh").size().rename("nk")
    n1_by_key = k1i.drop_duplicates("kh").set_index("kh")["n1"]
    good = n1_by_key.reindex(nk.index).mul(nk) <= max_pairs_per_key
    good_keys = nk.index[good.fillna(False).values]
    k1i = k1i[k1i.kh.isin(good_keys)]
    kk = kk[kk.kh.isin(good_keys)]
    m = k1i.merge(kk, on="kh", suffixes=("_s1", "_o"))
    m["w"] = np.array(weights)[m.kt_s1.values]
    g = m.groupby(["rid_s1", "rid_o"], sort=False).agg(score=("w", "sum"), nkeys=("w", "size")).reset_index()
    g = g.sort_values(["rid_o", "score", "rid_s1"], ascending=[True, False, True])
    g["rank"] = g.groupby("rid_o").cumcount()
    g = g[g["rank"] < top_k].rename(columns={"rid_s1": "s1", "rid_o": "other"})
    return g[["s1", "other", "score", "nkeys", "rank"]]

def candidate_pairs(k1: pd.DataFrame, kk: pd.DataFrame, max_s1_block=40, max_pairs_per_key=4000,
                    top_k=10, weights=(3.0, 1.0, 1.5, 1.0, 2.0, 1.5)) -> pd.DataFrame:
    return candidate_pairs_indexed(index_s1(k1, max_s1_block), kk, max_pairs_per_key, top_k, weights)
