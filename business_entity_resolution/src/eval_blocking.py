import sys, time
import numpy as np, pandas as pd
from config import WORK_DIR
from blocking import key_table, candidate_pairs

D = WORK_DIR / "dev"
def rd(k): return pd.read_csv(D / f"dev_source{k}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
def ids(df): return df.entity_id.str.split("-", n=1).str[1].astype(np.int64).values

t = time.time()
s1 = rd(1); s1_ids = ids(s1)
gt = pd.read_csv(D / "dev_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.matched_entity_ids != ""]
gt = gt.assign(tok=gt.matched_entity_ids.str.split(",")).explode("tok")
gt["s1"] = gt.source1_entity_id.str.split("-").str[1].astype(np.int64)
gt["src"] = gt.tok.str[1].astype(int); gt["oth"] = gt.tok.str.split("-").str[1].astype(np.int64)
k1 = key_table(s1, s1_ids); print("S1 keys", len(k1), f"{time.time()-t:.0f}s")
top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for k in (2, 3):
    df = rd(k); oid = ids(df)
    kk = key_table(df, oid)
    cp = candidate_pairs(k1, kk, top_k=top_k)
    cp["pk"] = (cp.s1.values << 31) | cp.other.values
    g = gt[gt.src == k].copy(); g["pk"] = (g.s1.values << 31) | g.oth.values
    g["hit"] = g.pk.isin(set(cp.pk.values))
    ctry = pd.Series(df.country.values, index=oid)
    g["country"] = ctry.reindex(g.oth.values).values
    print(f"\nS{k}: candidates {len(cp):,} ({len(cp)/len(s1):.1f} per S1, {len(cp)/len(df):.1f} per S{k} record) "
          f"recall {g.hit.mean():.4f}  [{time.time()-t:.0f}s]")
    print(g.groupby("country").hit.mean().round(4).to_dict())
    
    hit = cp[cp.pk.isin(set(g.pk.values))]
    print("  true pair rank distribution:", hit["rank"].value_counts(normalize=True).sort_index().head(5).round(3).to_dict())
