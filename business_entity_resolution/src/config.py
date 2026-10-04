import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", "dataset"))         
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))       
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", ROOT / "output"))

CHUNK_ROWS = int(os.environ.get("ER_CHUNK_ROWS", 200_000))  
SEED = 42
DEV_FRACTION = float(os.environ.get("ER_DEV_FRACTION", 0.07))  

for d in (WORK_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

def path(split: str, source: int) -> Path:
    """split is 'train', 'test' or 'dev' (dev = the small subset built by make_dev_subset.py)."""
    if split == "dev":
        return WORK_DIR / "dev" / f"dev_source{source}.tsv"
    return DATA_DIR / split / f"{split}_source{source}.tsv"

GT_PATH = DATA_DIR / "train" / "train_ground_truth.tsv"
