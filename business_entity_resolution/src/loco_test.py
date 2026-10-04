import numpy as np, pandas as pd
from config import WORK_DIR, SEED
from features import FEATURES
from train_model import make_model, f05_macro

D = WORK_DIR / "dev"
F = pd.read_pickle(WORK_DIR / "dev_features.pkl")
s1 = pd.read_csv(D / "dev_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1["id"] = s1.entity_id.str.split("-").str[1].astype(np.int64)
ctry = dict(zip(s1.id, s1.country)); F["country"] = F.s1.map(ctry)
gt = pd.read_csv(D / "dev_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
g = gt[gt.matched_entity_ids != ""]; g = g.assign(tok=g.matched_entity_ids.str.split(",")).explode("tok")
gt_sets = {}
for a, t in zip(g.source1_entity_id.str.split("-").str[1].astype(np.int64), g.tok):
    gt_sets.setdefault(a, set()).add((int(t[1]), int(t.split("-")[1])))
gt_sets = {a: frozenset(v) for a, v in gt_sets.items()}
is_val = ((F.s1.values.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(5) == 0
s1_val = s1.id.values[((s1.id.values.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(5) == 0]

def fit(rows):
    rng = np.random.RandomState(SEED)
    keep = rows.y.values | (rng.rand(len(rows)) < 0.3)
    m, _ = make_model(); m.fit(rows.loc[keep, FEATURES], rows.y[keep]); return m

def decide(rows, p):
    R = rows[["s1", "other", "src"]].copy(); R["p"] = p
    R = R.sort_values(["src", "other", "p"], ascending=[True, True, False], ignore_index=True)
    key = (R.src.values.astype(np.int64) << 40) | R.other.values
    first = np.r_[True, key[1:] != key[:-1]]; same = np.r_[key[:-1] == key[1:], False]
    pv = R.p.values; R["margin"] = pv - np.where(first & same, np.r_[pv[1:], 0.0], 0.0)
    return R[first]

def score(best, s1_ids, thr, mg):
    sel = best[(best.p >= thr) & (best.margin >= mg)]; pred = {}
    for a, s, o in zip(sel.s1.values, sel.src.values.astype(int), sel.other.values):
        pred.setdefault(int(a), set()).add((s, int(o)))
    return f05_macro(list(s1_ids), gt_sets, {a: frozenset(v) for a, v in pred.items()})

GRID = [(t, m) for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95) for m in (0.0, 0.1, 0.3)]
def tune(best, s1_ids): return max(GRID, key=lambda tm: score(best, s1_ids, *tm))

def run(train_c, test_c):
    if train_c is None:
        tr = F[~is_val]; te_rows = F[is_val & (F.country == test_c)]; te_ids = s1_val[np.isin(s1_val, s1.id[s1.country == test_c])]
        hv_rows, hv_ids = te_rows, te_ids
    else:
        tr = F[(F.country == train_c) & ~is_val]; hv_rows = F[(F.country == train_c) & is_val]
        hv_ids = s1_val[np.isin(s1_val, s1.id[s1.country == train_c])]
        te_rows = F[F.country == test_c]; te_ids = s1.id[s1.country == test_c].values
    m = fit(tr)
    b_hv = decide(hv_rows, m.predict_proba(hv_rows[FEATURES])[:, 1]); b_te = decide(te_rows, m.predict_proba(te_rows[FEATURES])[:, 1])
    thr_src = tune(b_hv, hv_ids); thr_or = tune(b_te, te_ids)
    return score(b_te, te_ids, *thr_src), thr_src, score(b_te, te_ids, *thr_or), thr_or

print("setting                    F0.5 (thr from train country)   F0.5 (oracle thr)")
for tc, ec in [(None, "US"), (None, "India"), ("US", "India"), ("India", "US")]:
    a, ta, b, tb = run(tc, ec)
    print(f"train={tc or 'US+India':9s} test={ec:6s}   {a:.4f} thr={ta}         {b:.4f} thr={tb}", flush=True)
