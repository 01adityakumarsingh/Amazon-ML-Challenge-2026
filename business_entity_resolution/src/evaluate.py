import argparse
import numpy as np, pandas as pd

def load(p):
    d = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return {a: frozenset(x for x in b.split(",") if x) for a, b in zip(d.iloc[:, 0], d.iloc[:, 1])}

def f05(t, p):
    if not t and not p: return 1.0
    if not t or not p: return 0.0
    tp = len(t & p)
    if tp == 0: return 0.0
    pr, rc = tp / len(p), tp / len(t)
    return 1.25 * pr * rc / (0.25 * pr + rc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--holdout", action="store_true", help="only S1 entities in the validation hash bucket used by train_model.py")
    a = ap.parse_args()
    gt, pred = load(a.gt), load(a.pred)
    ids = list(gt)
    if a.holdout:
        ids = [i for i in ids if ((np.uint64(int(i[3:])) * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(5) == 0]
    missing = [i for i in ids if i not in pred]
    assert not missing, f"{len(missing)} S1 entities missing from predictions"
    sc = [f05(gt[i], pred[i]) for i in ids]
    tp = sum(len(gt[i] & pred[i]) for i in ids); npred = sum(len(pred[i]) for i in ids); ntrue = sum(len(gt[i]) for i in ids)
    print(f"entities={len(ids):,}  macro F0.5={np.mean(sc):.4f}  pair precision={tp/max(npred,1):.4f}  pair recall={tp/max(ntrue,1):.4f}")
    single = [i for i in ids if not gt[i]]
    print(f"singletons={len(single):,}  correctly empty={np.mean([not pred[i] for i in single]):.3f}")
