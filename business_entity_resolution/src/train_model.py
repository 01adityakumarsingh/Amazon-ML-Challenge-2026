
import gc, os, sys, time, json, pickle
import numpy as np, pandas as pd
from config import WORK_DIR, SEED, CHUNK_ROWS
from data_loader import iter_source
from blocking import key_table, index_s1, candidate_pairs_indexed
from features import prep, fit_vectorizers, pair_features, FEATURES
from decision import best_per_record

SPLIT = os.environ.get("ER_TRAIN_SPLIT", "dev")   
NEG_FRAC = float(os.environ.get("ER_NEG_FRAC", "0.3"))
CHUNK = int(os.environ.get("ER_CHUNK_ROWS", CHUNK_ROWS))

def ids_of(series):
    return series.str.split("-", n=1).str[1].astype(np.int64).values

def is_val_id(a: np.ndarray) -> np.ndarray:
    return ((a.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(5) == 0

def load_ground_truth(split):
    from data_loader import read_ground_truth
    from config import DATA_DIR, GT_PATH
    if split == "dev":
        gt_path = WORK_DIR / "dev" / "dev_ground_truth.tsv"
    else:
        gt_path = GT_PATH
    s1, oth, src, cnt = [], [], [], []
    with open(gt_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            a = int(a.split("-")[1])
            items = [x for x in b.split(",") if x]
            cnt.append(len(items))
            for x in items:
                p, n = x.split("-")
                s1.append(a); src.append(int(p[1])); oth.append(int(n))
    return (np.array(s1, np.int64), np.array(oth, np.int64), np.array(src, np.int8), np.array(cnt, np.int32))

def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=int(os.environ.get("ER_N_ESTIMATORS", 600)),
            learning_rate=float(os.environ.get("ER_LR", 0.05)),
            num_leaves=int(os.environ.get("ER_NUM_LEAVES", 95)),
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=int(os.environ.get("ER_MIN_CHILD", 30)),
            random_state=SEED, verbose=-1), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_leaf_nodes=63,
                                              random_state=SEED), "sklearn-hgb"

def f05_macro(s1_ids, gt_sets, pred_sets):
    sc = []
    for a in s1_ids:
        t, p = gt_sets.get(a, frozenset()), pred_sets.get(a, frozenset())
        if not t and not p: sc.append(1.0); continue
        if not t or not p: sc.append(0.0); continue
        tp = len(t & p); pr, rc = tp / len(p), tp / len(t)
        sc.append(0.0 if tp == 0 else 1.25 * pr * rc / (0.25 * pr + rc))
    return float(np.mean(sc))

def main():
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    log(f"split={SPLIT} neg_frac={NEG_FRAC} chunk_rows={CHUNK}")

    s1_ids, oth_ids, src_ids, cnt = load_ground_truth(SPLIT)
    truth = {(int(a), int(s), int(o)) for a, s, o in zip(s1_ids, src_ids, oth_ids)}
    log(f"ground truth: {len(truth):,} pairs")

    s1_parts = []
    for ch in iter_source(SPLIT, 1, CHUNK):
        s1_parts.append(ch); log(f"S1: loaded {sum(len(p) for p in s1_parts):,} records so far...")
    s1 = pd.concat(s1_parts, ignore_index=True); del s1_parts
    i1 = ids_of(s1.entity_id)
    log(f"S1: {len(s1):,} records loaded, tokenizing (this step is the slow, silent one -- do not interrupt)...")
    p1 = prep(s1); pos1 = pd.Series(np.arange(len(s1)), index=i1)
    log("S1: tokenized, now building the blocking key index...")
    k1i = index_s1(key_table(s1, i1, log=log))
    log(f"S1 prepared: {len(s1):,} records, key index {len(k1i):,} rows")

    samp = {"name_txt": [], "addr_txt": []}
    for k in (2, 3):
        first = next(iter_source(SPLIT, k, min(CHUNK, 200_000))); pk0 = prep(first)
        samp["name_txt"] += pk0["name_txt"][::2]; samp["addr_txt"] += pk0["addr_txt"][::2]
    vecs = fit_vectorizers(p1, [samp], step=max(1, len(s1) // 300_000))
    X1 = (vecs[0].transform(p1["name_txt"]).tocsr(), vecs[1].transform(p1["addr_txt"]).tocsr())
    log("S1 vectorized")

    import resource
    def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  

    rng = np.random.RandomState(SEED)
    feature_paths = []
    n_pos_seen = n_neg_kept = n_pairs_seen = 0
    for k in (2, 3):
        parts = []  
        for ch in iter_source(SPLIT, k, CHUNK):
            ch = ch.reset_index(drop=True); ik = ids_of(ch.entity_id)
            pk = prep(ch); kk = key_table(ch, ik)
            cand = candidate_pairs_indexed(k1i, kk, top_k=10).reset_index(drop=True)
            if len(cand):
                posk = pd.Series(np.arange(len(ch)), index=ik)
                Xk = (vecs[0].transform(pk["name_txt"]).tocsr(), vecs[1].transform(pk["addr_txt"]).tocsr())
                F = pair_features(p1, pk, X1, Xk, vecs, cand, pos1, posk, k)
                y = np.fromiter(((int(a), k, int(o)) in truth for a, o in zip(cand.s1.values, cand.other.values)),
                                bool, len(cand))
                keep = y | (rng.rand(len(cand)) < NEG_FRAC)
                F = F.loc[keep].copy(); F["s1"] = cand.s1.values[keep]; F["other"] = cand.other.values[keep]
                F["y"] = y[keep]
                n_pos_seen += int(y.sum()); n_neg_kept += int(keep.sum() - y.sum()); n_pairs_seen += len(cand)
                parts.append(F)
                del Xk, F, cand, kk, pk
            del ch
            gc.collect()
            if n_pairs_seen and n_pairs_seen % (CHUNK * 4) < CHUNK:
                log(f"S{k}: {n_pairs_seen:,} candidate pairs seen so far, "
                    f"{n_pos_seen:,} positives + {n_neg_kept:,} sampled negatives kept, peak RSS {rss_mb():,.0f} MB")
        
        fpath = WORK_DIR / f"{SPLIT}_features_s{k}.pkl"
        pd.concat(parts, ignore_index=True).to_pickle(fpath)  
        feature_paths.append(fpath)
        del parts; gc.collect()
        log(f"S{k} done: {n_pairs_seen:,} candidate pairs seen total, flushed to {fpath.name}, peak RSS {rss_mb():,.0f} MB")

    F = pd.concat([pd.read_pickle(p) for p in feature_paths], ignore_index=True)
    for p in feature_paths: p.unlink()  
    gc.collect()
    log(f"feature table assembled: {len(F):,} rows ({F.y.mean():.3f} positive), peak RSS {rss_mb():,.0f} MB")
    F.to_pickle(WORK_DIR / f"{SPLIT}_features.pkl")

    val_mask = is_val_id(F.s1.values)
    all_s1 = np.unique(i1)
    val_s1_ids = all_s1[is_val_id(all_s1)]
    tr = F[~val_mask]
    model, name = make_model()
    model.fit(tr[FEATURES], tr.y)
    log(f"trained {name} on {len(tr):,} rows")

    F["p"] = model.predict_proba(F[FEATURES])[:, 1].astype(np.float32)
    gt_sets = {}
    for a, s, o in truth:
        gt_sets.setdefault(a, set()).add((s, o))
    gt_sets = {a: frozenset(v) for a, v in gt_sets.items()}

    best = best_per_record(F.s1.values, F.other.values, F.src.values, F.p.values)
    best = best[best.s1.isin(set(val_s1_ids.tolist()))]
    GRID = [(t, m) for t in (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
                   for m in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)]
    res = []
    for thr, mg in GRID:
        sel = best[(best.p >= thr) & (best.margin >= mg)]
        pred = {}
        for a, s, o in zip(sel.s1.values, sel.src.values.astype(int), sel.other.values):
            pred.setdefault(int(a), set()).add((s, int(o)))
        res.append((f05_macro(val_s1_ids.tolist(), gt_sets, {a: frozenset(v) for a, v in pred.items()}), thr, mg))
    res.sort(reverse=True)
    log(f"top settings: {[(round(s,4),t,m) for s,t,m in res[:5]]}")
    log(f"F0.5 by threshold @margin 0: { {t: round(s,4) for s,t,m in res if m==0.0} }")

    pickle.dump({"model": model, "features": FEATURES, "thr": res[0][1], "margin": res[0][2]},
                open(WORK_DIR / "model.pkl", "wb"))
    json.dump({"threshold": res[0][1], "margin": res[0][2], "val_f05": res[0][0]},
              open(WORK_DIR / "threshold.json", "w"))
    log("saved work/model.pkl")

if __name__ == "__main__":
    main()
