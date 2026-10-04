import argparse, pickle, time
from collections import defaultdict
import numpy as np, pandas as pd
from config import OUT_DIR, WORK_DIR, CHUNK_ROWS
from data_loader import iter_source
from blocking import key_table, index_s1, candidate_pairs_indexed
from features import prep, fit_vectorizers, pair_features
from decision import best_per_record

NB = 32  

def ids_of(series: pd.Series, k: int) -> np.ndarray:
    if not (series.str[:3] == f"S{k}-").all():
        raise ValueError(f"unexpected id prefix in source {k}")
    suf = series.str[3:]
    n = suf.astype(np.int64).values
    if not (n.astype(str) == suf.values).all():
        raise ValueError("ids with leading zeros/non-canonical digits: switch the pipeline to string ids")
    return n

class Sink:
    def __init__(self, d, name):
        self.paths = [d / f"{name}_{b:02d}.bin" for b in range(NB)]
        for p in self.paths: p.write_bytes(b"")
    def add(self, s1, src, other):
        arr = np.column_stack([s1, np.full(len(s1), src), other]).astype(np.int64)
        b = arr[:, 0] % NB
        for v in np.unique(b):
            with open(self.paths[v], "ab") as f: arr[b == v].tofile(f)
    def read(self, b):
        a = np.fromfile(self.paths[b], dtype=np.int64)
        return a.reshape(-1, 3)

def write_lists(sink: Sink, s1_ids: np.ndarray, out_path, header):
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for b in range(NB):
            ids_b = s1_ids[s1_ids % NB == b]
            arr = sink.read(b)
            lists = {}
            if len(arr):
                d = pd.DataFrame(arr, columns=["s1", "src", "oth"]).drop_duplicates().sort_values(["s1", "src", "oth"])
                d["tok"] = "S" + d.src.astype(str) + "-" + d.oth.astype(str)
                lists = d.groupby("s1").tok.agg(",".join).to_dict()
            for a in ids_b:
                f.write(f"S1-{a}\t{lists.get(a, '')}\n"); n_rows += 1
    return n_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test"); ap.add_argument("--model", default=str(WORK_DIR / "model.pkl"))
    ap.add_argument("--thr", type=float); ap.add_argument("--margin", type=float)
    ap.add_argument("--top-k", type=int, default=10); ap.add_argument("--max-s1-block", type=int, default=60)
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    a = ap.parse_args()
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    art = pickle.load(open(a.model, "rb")); model, FEATS = art["model"], art["features"]
    thr = a.thr if a.thr is not None else art.get("thr", 0.9); mg = a.margin if a.margin is not None else art.get("margin", 0.1)
    log(f"model loaded; decision rule: p >= {thr}, margin >= {mg}")
    tmp = WORK_DIR / f"tmp_{a.split}"; tmp.mkdir(parents=True, exist_ok=True)
    
    s1 = pd.concat(list(iter_source(a.split, 1)), ignore_index=True)
    i1 = ids_of(s1.entity_id, 1); assert len(np.unique(i1)) == len(i1), "duplicate S1 ids"
    log(f"S1: {len(s1):,} records; countries: {s1.country.value_counts().to_dict()}")
    p1 = prep(s1); pos1 = pd.Series(np.arange(len(s1)), index=i1)
    k1i = index_s1(key_table(s1, i1), a.max_s1_block)
    log(f"S1 key index: {len(k1i):,} rows")
    
    samp = {"name_txt": [], "addr_txt": []}
    for k in (2, 3):
        first = next(iter_source(a.split, k, 200_000)); pk0 = prep(first)
        samp["name_txt"] += pk0["name_txt"]; samp["addr_txt"] += pk0["addr_txt"]
    vecs = fit_vectorizers(p1, [samp], step=max(1, len(s1) // 300_000))
    X1 = (vecs[0].transform(p1["name_txt"]).tocsr(), vecs[1].transform(p1["addr_txt"]).tocsr())
    log("S1 features ready")

    cand_sink, match_sink = Sink(tmp, "cand"), Sink(tmp, "match")
    stats = defaultdict(lambda: defaultdict(int))
    for k in (2, 3):
        for ch in iter_source(a.split, k, a.chunk_rows):
            ch = ch.reset_index(drop=True); ik = ids_of(ch.entity_id, k)
            pk = prep(ch); kk = key_table(ch, ik)
            cand = candidate_pairs_indexed(k1i, kk, top_k=a.top_k).reset_index(drop=True)
            ctry = pd.Series(ch.country.values, index=ik)
            for c, n in ch.country.value_counts().items(): stats[(k, c)]["records"] += n
            if len(cand):
                posk = pd.Series(np.arange(len(ch)), index=ik)
                Xk = (vecs[0].transform(pk["name_txt"]).tocsr(), vecs[1].transform(pk["addr_txt"]).tocsr())
                F = pair_features(p1, pk, X1, Xk, vecs, cand, pos1, posk, k)
                p = model.predict_proba(F[FEATS])[:, 1]
                best = best_per_record(cand.s1.values, cand.other.values, np.full(len(cand), k), p)
                sel = best[(best.p >= thr) & (best.margin >= mg)]
                cand_sink.add(cand.s1.values, k, cand.other.values)
                match_sink.add(sel.s1.values, k, sel.other.values)
                for c, n in ctry.reindex(best.other.values).value_counts().items(): stats[(k, c)]["with_candidates"] += n
                for c, n in ctry.reindex(sel.other.values).value_counts().items(): stats[(k, c)]["assigned"] += n
            log(f"S{k}: chunk of {len(ch):,} done ({len(cand):,} candidate pairs)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n1 = write_lists(match_sink, i1, OUT_DIR / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids")
    n2 = write_lists(cand_sink, i1, OUT_DIR / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids")
    log(f"wrote {n1:,} / {n2:,} rows")

    print("\nSanity report (training reference: ~74-75% of S2/S3 records matched, 5.6% of S1 without matches)")
    for (k, c), d in sorted(stats.items()):
        print(f"  S{k} {c:8s} records={d['records']:>10,}  with_candidate={d['with_candidates']/d['records']:.3f}  "
              f"assigned={d['assigned']/d['records']:.3f}")
    verify(OUT_DIR, s1, a.split)

def verify(out_dir, s1, split):
    """Cheap local checks mirroring the challenge rules (run the official validator too)."""
    m = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(out_dir / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert set(m.source1_entity_id) == set(s1.entity_id) and not m.source1_entity_id.duplicated().any(), "S1 rows mismatch"
    assert set(c.source1_entity_id) == set(s1.entity_id), "candidate rows mismatch"
    cm = m.set_index("source1_entity_id").matched_entity_ids; cc = c.set_index("source1_entity_id").candidate_entity_ids.reindex(cm.index)
    bad = 0
    for x, y in zip(cm.values, cc.values):
        if x and not set(x.split(",")) <= set(y.split(",")): bad += 1
    lens = cm.str.count(",").add(1).where(cm != "", 0)
    print(f"\nverify OK: {len(m):,} rows; S1 with >=1 match: {(lens > 0).mean():.3f}; mean matches/S1: {lens.mean():.2f}; "
          f"matches not in candidates: {bad}")

if __name__ == "__main__":
    main()
