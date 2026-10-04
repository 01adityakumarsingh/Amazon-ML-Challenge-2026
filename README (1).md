# Business Entity Resolution — Pipeline

Matches Source 2 / Source 3 business records to Source 1 (the deduplicated reference source),
across US, India, and — at test time — France, using only the provided data (no external
lookups, APIs, or geocoding).

**Verified full-scale result:** trained on the entire training set, this pipeline scored
**0.888 macro F0.5** on the actual leaderboard (held-out validation F0.5 was 0.9397 — see
`Documentation_template.md` for the full write-up of that run, including known gaps between
validation and leaderboard performance).

See `Documentation_template.md` for the full methodology write-up (approach, blocking design,
features, results, limitations). This file covers how to run the code.

## 1. Requirements

```bash
pip install -r requirements.txt
pip install lightgbm   # strongly recommended -- see note below
```

Python 3.9+. Core dependencies: numpy, pandas, scikit-learn, scipy. **LightGBM is used for the
classifier when installed; if it isn't, `train_model.py` silently falls back to scikit-learn's
weaker `HistGradientBoostingClassifier`** — check your run's log for `trained lightgbm on ...`
vs. `trained sklearn-hgb on ...` to confirm which one actually ran. On Kaggle, this needs the
notebook's **Internet** setting turned on (Settings panel) for the one-time `pip install`; the
rest of the pipeline needs no network at all.

`rapidfuzz` and `unidecode` are listed for optional/future use but the current pipeline runs on
the standard library plus the core dependencies above.

## 2. Data layout expected

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Point the pipeline at it with an environment variable (default is `./dataset` relative to
wherever you run the scripts from):

```bash
export ER_DATA_DIR=/path/to/dataset
```

## 3. Quick start — the exact configuration that produced the 0.888 result

```bash
cd src

export ER_TRAIN_SPLIT=train      # train directly on dataset/train/ -- no dev-subset copy needed
export ER_NEG_FRAC=0.3           # negative-sampling rate; see Section 7 for memory trade-offs
export ER_CHUNK_ROWS=150000      # rows per chunk when streaming Source 2/3

python train_model.py            # trains on the FULL training set, saves work/model.pkl
python predict.py --split test   # writes output/matching_results.tsv + output/candidate_pairs.tsv
```

This full run took roughly **1h50m for training** and **~2h20m for inference** on a
~30 GB-RAM Kaggle instance. Expect a long, unattended run either way — see Section 7 before
starting one.

### Recommended before a real run: build and check a dev subset first

Before spending hours on a full run, sanity-check the pipeline on a small subset:

```bash
python make_dev_subset.py                  # builds work/dev/ (~7% of S1 entities + their matches)
python train_model.py                      # ER_TRAIN_SPLIT unset/=dev -> trains on the small subset, prints validation macro F0.5
python predict.py --split dev              # smoke-tests the full inference path
python evaluate.py --pred ../output/matching_results.tsv --gt ../work/dev/dev_ground_truth.tsv
```

### Re-scoring with a different threshold WITHOUT re-running the full pipeline

`predict.py` caches every scored candidate to `work/<split>_scored.pkl`. Once that exists, trying
a new decision threshold takes **seconds**, not hours:

```bash
python apply_threshold.py --split test --thr 0.5 --margin 0.15
# or per-country:
python apply_threshold.py --split test --thr-by-country '{"India": [0.5, 0.15], "US": [0.65, 0.3]}'
```

This only re-applies the decision rule to already-scored candidates — it cannot recover recall
that blocking/top-k missed in the original `predict.py` run, only trade precision/recall within
what was already found. `candidate_pairs.tsv` from the original run remains valid and unchanged.

## 4. Environment variables

**"Default" below is what the code falls back to if you set nothing.** "Used for the 0.888 run"
is the actual value set in that run's Kaggle cells — these are two different things, and the
0.888 run overrode several defaults explicitly. Don't assume the default column describes what
was actually run.

| Variable | Default | Used for the 0.888 run | Purpose |
|---|---|---|---|
| `ER_DATA_DIR` | `dataset` | `/kaggle/input/datasets/<user>/<dataset-slug>/dataset` | Folder containing `train/` and `test/` (exact Kaggle path depends on your own uploaded dataset's slug — check `!ls /kaggle/input/` rather than copying this literally) |
| `ER_WORK_DIR` | `<project>/work` | `/kaggle/working/work` | Caches, dev subset, trained model, tmp files |
| `ER_OUT_DIR` | `<project>/output` | `/kaggle/working/output` | Where the two output TSVs are written |
| `ER_CHUNK_ROWS` | `200000` | **`150000`** | Rows per chunk when streaming Source 2/3. **Lower this first if you hit a memory error.** |
| `ER_DEV_FRACTION` | `0.07` | *(not used — irrelevant when `ER_TRAIN_SPLIT=train`)* | Fraction of S1 entities kept in `make_dev_subset.py`'s output |
| `ER_TRAIN_SPLIT` | `dev` | **`train`** | `train` = train directly on the full `dataset/train/` files (no subset copy); `dev` = train on `work/dev/` |
| `ER_NEG_FRAC` | `0.3` | **`0.3`** (explicitly set, matches default) | Fraction of non-matching candidate pairs kept during training. **The main lever for controlling training memory** — see Section 7 |
| `ER_N_ESTIMATORS`, `ER_LR`, `ER_NUM_LEAVES`, `ER_MIN_CHILD` | `600` / `0.05` / `95` / `30` | *(not overridden — defaults used)* | LightGBM hyperparameters |
| `ER_CACHE_FEATURES` | `0` for `train` split, `1` for `dev` | *(not overridden — `0` applied automatically)* | Whether to also write a full combined feature-table pickle (skipped by default at full scale) |

The exact cells used for the 0.888 run, for reference:
```python
# Cell 1
import os, sys
DATA_DIR = "/kaggle/input/datasets/<user>/<dataset-slug>/dataset"   # <- your actual path
SRC_DIR  = "/kaggle/input/datasets/<user>/<code-dataset-slug>/src"  # <- your actual path
WORK_DIR = "/kaggle/working/work"
OUT_DIR  = "/kaggle/working/output"
os.makedirs(WORK_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)
os.environ["ER_DATA_DIR"] = DATA_DIR
os.environ["ER_WORK_DIR"] = WORK_DIR
os.environ["ER_OUT_DIR"]  = OUT_DIR
os.environ["ER_CHUNK_ROWS"] = "150000"

# Cell 2
os.environ['ER_TRAIN_SPLIT'] = 'train'
os.environ['ER_NEG_FRAC'] = '0.3'
os.environ['ER_CHUNK_ROWS'] = '150000'
%cd {SRC_DIR}
!python train_model.py
```

`predict.py` also takes command-line flags for a given run without touching env vars:

```bash
python predict.py --split test \
    --model ../work/model.pkl \
    --thr 0.6 --margin 0.3 \
    --top-k 10 --max-s1-block 60 \
    --chunk-rows 150000
```

`--thr`/`--margin` default to whatever `train_model.py` tuned and saved in `model.pkl` (including
any per-country overrides — see below); pass them explicitly to override for every country
uniformly.

### Per-country decision thresholds

`train_model.py` searches for a per-country threshold/margin in addition to the global one
(skipped for a country if it has fewer than 200 validation entities). `predict.py` applies these
automatically per record's country, falling back to the global setting for any country not in
that dictionary — which is exactly what happens for France, since it has zero training coverage.

## 5. What each script does

| Script | Role |
|---|---|
| `config.py` | Central paths/settings, all overridable via the env vars above |
| `data_loader.py` | Chunked TSV readers; streams `train_ground_truth.tsv` into flat arrays |
| `normalize.py` | Tokenization, accent stripping, legal-suffix list, de-leet digit/letter fix, structured address parsing |
| `transliterate.py` | Approximate Brahmic-script → Latin phonetic transliteration for Indian names |
| `blocking.py` | Hashed multi-key candidate generation, chunk-safe, country as a hard blocking key |
| `features.py` | ~38 pair features (TF-IDF cosine, token overlap, transliteration, structured address matching, blocking context) |
| `decision.py` | Shared one-S1-per-record assignment rule (used by training eval, predict, and apply_threshold) |
| `make_dev_subset.py` | Builds a small, representative subset of the training data for fast iteration |
| `train_model.py` | Streams the full training set in chunks, flushes each source to disk before the next, trains the classifier, grid-searches global and per-country threshold/margin, saves `model.pkl` |
| `apply_threshold.py` | Re-scores a cached `predict.py` run with a new threshold/margin in seconds, no re-inference needed |
| `loco_test.py` | Leave-one-country-out check (train on US, test on India, and vice versa) — a proxy for the unseen France country |
| `predict.py` | Full inference on `train`/`test`/`dev`; writes both required output files; caches scores for `apply_threshold.py`; prints a per-country sanity report |
| `eval_blocking.py` | Standalone blocking-recall measurement, independent of the model |
| `evaluate.py` | Scores a `matching_results.tsv` against any ground-truth file using the exact challenge metric |

## 6. Output format

Both files, one row per Source 1 entity, tab-separated, no quoting:

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

`candidate_pairs.tsv` has the same shape with `candidate_entity_ids` instead, and is guaranteed
to be a superset of the matches in `matching_results.tsv` — `predict.py` asserts this itself
(`verify()` at the end of the run) as a lightweight local check before you run the challenge's
own `utils/validate_submission.py`.

## 7. Memory: what actually crashes this pipeline, and how to avoid it

This pipeline was iterated through several real out-of-memory crashes on ~30 GB Kaggle
instances at full scale (2.2M S1 / ~5M+5.3M S2+S3 training records). What was learned, in order
of how much it matters:

1. **`ER_NEG_FRAC` is the single biggest lever.** At `0.3`, the full training set produces
   roughly 34M feature rows; at `0.4`, roughly 42M. Both of these have been observed to crash a
   ~30 GB machine during the final train/validation split and model-fit step (not during
   candidate generation itself, which is chunked and stays flat). **If you hit a crash right
   after the `"feature table assembled: N rows"` log line, lower `ER_NEG_FRAC` first** —
   try `0.10`–`0.15` for meaningfully more headroom.
2. **Source 2 and Source 3 are each flushed to disk before the other starts processing** —
   this is what keeps memory flat through chunked candidate generation (verified: peak RSS held
   exactly constant through all of Source 3 in testing). This is not something you need to
   configure; it's automatic.
3. **The final feature table is reassembled incrementally** (one flushed file read and merged at
   a time, not all held simultaneously) and **training uses `.loc` directly on a boolean mask**
   rather than holding a separate full copy of the training subset alongside the full table — both
   reduce peak memory around the model-fit step, but do not eliminate the need to keep
   `ER_NEG_FRAC` reasonable at full scale.
4. `predict.py` was already chunk-streaming from the start and has not been observed to leak
   memory across chunks or across Source 2 into Source 3.

**Practical guidance:** run `train_model.py` once with a conservative `ER_NEG_FRAC` (0.10–0.15)
to get a safe, working baseline before experimenting with higher values — a crash partway through
a multi-hour run costs the entire run, since chunked candidate generation must be redone from
scratch for a different `ER_NEG_FRAC` (the sampling happens during that same pass).

Also lower `ER_CHUNK_ROWS` (e.g. to `75000`) if a memory error happens **during** candidate
generation specifically (rather than at final assembly) — this is a secondary, less common cause.

`train_model.py` and `predict.py` both print progress with elapsed time and (in `train_model.py`)
peak RSS at each stage — watch these numbers rather than assuming a long silence means it's
frozen; full-scale S1 preparation alone can take several minutes with no output.

## 8. Fair play

No external databases, APIs, geocoding services, or internet-sourced data were used at any
stage. All normalization dictionaries (legal suffixes, the Devanagari transliteration table)
were hand-authored from general linguistic/Unicode-structure knowledge; all learned components
(TF-IDF vectorizers, the classifier) are fit only on the provided train/test text.
