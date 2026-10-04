import numpy as np
import pandas as pd
from config import WORK_DIR, DEV_FRACTION, SEED, CHUNK_ROWS
from data_loader import iter_source, id_to_int, read_ground_truth

def keep_hash(ids: np.ndarray, frac: float, salt: int) -> np.ndarray:
    h = ((ids.astype(np.uint64) + np.uint64(salt)) * np.uint64(11400714819323198485)) >> np.uint64(40)
    return (h % np.uint64(1_000_000)) < np.uint64(int(frac * 1_000_000))

def main():
    out = WORK_DIR / "dev"; out.mkdir(parents=True, exist_ok=True)
    s1, oth, src, cnt = read_ground_truth()
   
    all_s1 = np.unique(s1) 
    matched = {2: np.sort(oth[src == 2]), 3: np.sort(oth[src == 3])}

    dev_s1_ids = set()
    with open(out / "dev_source1.tsv", "w", encoding="utf-8") as fo:
        fo.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for ch in iter_source("train", 1):
            ids = id_to_int(ch.entity_id)
            m = keep_hash(ids, DEV_FRACTION, SEED)
            dev_s1_ids.update(ids[m].tolist())
            ch[m].to_csv(fo, sep="\t", header=False, index=False, quoting=3)
    dev_s1 = np.array(sorted(dev_s1_ids), dtype=np.int64)
    print("dev S1 entities:", len(dev_s1))

    pair_mask = np.isin(s1, dev_s1)
    gt = pd.DataFrame({"s1": s1[pair_mask], "src": src[pair_mask], "oth": oth[pair_mask]})
    gt["tok"] = "S" + gt.src.astype(str) + "-" + gt.oth.astype(str)
    grouped = gt.groupby("s1").tok.agg(",".join)
    with open(out / "dev_ground_truth.tsv", "w", encoding="utf-8") as fo:
        fo.write("source1_entity_id\tmatched_entity_ids\n")
        for a in dev_s1:
            fo.write(f"S1-{a}\t{grouped.get(a, '')}\n")
    dev_matched = {k: np.sort(gt.oth[gt.src == k].values) for k in (2, 3)}

    for k in (2, 3):
        n_keep = n_match = 0
        with open(out / f"dev_source{k}.tsv", "w", encoding="utf-8") as fo:
            fo.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for ch in iter_source("train", k):
                ids = id_to_int(ch.entity_id)
                is_dev_match = np.isin(ids, dev_matched[k])
                is_orphan = ~np.isin(ids, matched[k])
                m = is_dev_match | (is_orphan & keep_hash(ids, DEV_FRACTION, SEED + k))
                n_keep += m.sum(); n_match += is_dev_match.sum()
                ch[m].to_csv(fo, sep="\t", header=False, index=False, quoting=3)
        print(f"dev S{k}: {n_keep:,} records ({n_match:,} matched, {n_keep - n_match:,} orphans)")

if __name__ == "__main__":
    main()
