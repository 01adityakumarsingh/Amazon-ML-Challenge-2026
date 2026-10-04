import numpy as np
import pandas as pd

def best_per_record(s1, other, src, p) -> pd.DataFrame:
    R = pd.DataFrame({"s1": s1, "other": other, "src": np.asarray(src, dtype=np.int64), "p": np.asarray(p, dtype=np.float32)})
    R = R.sort_values(["src", "other", "p", "s1"], ascending=[True, True, False, True], ignore_index=True)
    key = (R.src.values << 40) | R.other.values
    first = np.r_[True, key[1:] != key[:-1]]
    same_next = np.r_[key[:-1] == key[1:], False]
    pv = R.p.values
    R["margin"] = pv - np.where(first & same_next, np.r_[pv[1:], 0.0], 0.0)
    return R[first]
