import numpy as np
import pandas as pd
from config import CHUNK_ROWS, GT_PATH, path

def iter_source(split: str, source: int, chunksize: int = CHUNK_ROWS):
    yield from pd.read_csv(path(split, source), sep="\t", dtype=str, keep_default_na=False,
                           quoting=3, chunksize=chunksize)

def id_to_int(series: pd.Series) -> np.ndarray:
    return series.str.split("-", n=1).str[1].astype(np.int64).values

def read_ground_truth():
    s1, oth, src, cnt = [], [], [], []
    with open(GT_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            a = int(a.split("-")[1])
            items = [x for x in b.split(",") if x]
            cnt.append(len(items))
            for x in items:
                p, n = x.split("-")
                s1.append(a); src.append(int(p[1])); oth.append(int(n))
    return (np.array(s1, dtype=np.int64), np.array(oth, dtype=np.int64),
            np.array(src, dtype=np.int8), np.array(cnt, dtype=np.int32))
