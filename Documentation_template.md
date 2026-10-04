# Business Entity Resolution — Methodology Document

**Team:** INSIDERS
**Leaderboard score (matching_results.tsv submitted):** 0.888 (macro F0.5)

---

## 1. Overview

The task links noisy Source 2 / Source 3 business records to a deduplicated Source 1 reference,
with each S2/S3 record matching at most one S1 entity (or none). Evaluation is macro-averaged
F0.5 per Source 1 entity — a precision-heavy metric that gives full credit for correctly
predicting "no match" on singletons and penalizes false merges roughly twice as heavily as
missed matches.

Two properties of the training ground truth, verified on the full 7,638,365-pair training set
before any modeling began, shaped every design choice below:

- **Every Source 2/3 record matches at most one Source 1 entity** (zero violations across all
  training pairs) — the task reduces to a per-record top-1 assignment problem, not clustering.
- **`country` agrees on 100% of true matched pairs** — used as a hard equality key during
  blocking, never as a trained feature, so the pipeline handles France (present only in the test
  set, with zero training examples) without any code change.

Roughly 26% of Source 2/3 records and 5.6% of Source 1 entities have no match in the training
data, so correctly predicting "no match" is a first-class objective.

## 2. Methodology / Pipeline

```
raw TSVs → normalization → blocking (candidate generation) → pair feature engineering
         → gradient-boosted classifier → per-record top-1 assignment (single global threshold + margin)
         → matching_results.tsv / candidate_pairs.tsv
```

**Scale-driven engineering.** Source 1 has ~2.2M records; Source 2 and Source 3 have ~5M each in
training, and a comparably large test set (~1.73M S1, with US, India, and France all present).
Source 1 is prepared once and kept resident (tokenization, blocking-key index, TF-IDF vectorizers).
Source 2/3 are **never loaded in full** — they are streamed in configurable-size chunks
(`ER_CHUNK_ROWS`), and each source's sampled training pairs are **flushed to disk before the next
source begins processing**, so memory does not accumulate across S2 into S3 (an earlier version
without this flush would grow unbounded through the second source and crash near the end of
processing). The final feature table is then reassembled incrementally from the two flushed
files rather than requiring three full copies in memory simultaneously.

A smaller `dev` subset (a deterministic hash-based sample of Source 1 entities, all their
matches, and a proportional share of orphan Source 2/3 records, preserving the same match/orphan
ratio as the full data) was used throughout development for fast iteration before every full-scale
run.

### 2.1 Normalization

Standard-library only, no external dictionaries or lookups (per the challenge's fair-play rules):

- Unicode NFKD decomposition strips Latin diacritics only; Indic combining marks are left intact
  so transliterated names are not corrupted.
- Tokenization keeps Latin and Indic Unicode blocks; `&`/`+` normalize to "and".
- A hand-authored, multi-country legal-suffix stopword list (Inc, Ltd, Pvt, GmbH, SARL, SAS, Pte,
  Opc, ...) is stripped from name tokens used for matching, but tracked separately as a
  "legal-conflict" signal (a genuine mismatch, e.g. "Pvt" vs "Ltd", down-weights a match without
  being a hard rule).
- Address numeric tokens are extracted with leading zeros stripped, so numbering-format noise
  doesn't block genuine matches.
- Alphabetic address tokens (length ≥ 4) become candidate street/city/area anchors.

**Cross-script transliteration.** A meaningful share of Indian S2/S3 business names are rendered
in a Brahmic script (Devanagari, Bengali, Telugu, Kannada, Malayalam, etc.) as word-for-word
phonetic transliterations of the same Latin words used in the matching Source 1 record (e.g.
"Private Limited" → "प्राइवेट लिमिटेड"). This is exploited using a property of the Unicode
Standard itself, not an external lookup: the major Brahmic scripts are laid out at parallel
relative offsets within their Unicode blocks (they all descend from Brahmi), so a character in
any of these scripts can be mapped to its Devanagari counterpart by block-relative offset, then
rendered to an approximate Latin phonetic form with a single hand-built Devanagari→Latin
consonant/vowel table. This transliteration is used both as an extra blocking key (transliterated
name tokens, in the same key space as ordinary Latin name tokens) and as a pair feature
(transliteration-aware token-prefix Jaccard/containment between the two records' names).

### 2.2 Candidate generation / blocking

Multi-key hashed blocking, unioned across several weighted key families, all namespaced by
`country` (giving the hard country-equality constraint for free):

| Key family | Built from | Purpose |
|---|---|---|
| Address (number × street/city prefix) | numeric token × alpha-token prefix | Primary signal: house number plus a nearby place-name token |
| Name unigram / bigram | non-legal name-token prefixes, individually and in pairs | Typo- and reorder-tolerant name matching |
| Address, number-free | pairs of alpha address tokens, no number required | Recovers matches where the house number is missing or corrupted |
| Compact-name prefix | concatenated, legal-suffix-stripped name, `.com`-suffix stripped | Recovers glued/domain-style names |
| Transliterated name unigram | approximate Latin transliteration of a Brahmic name | Recovers native-script Indian names against a Latin S1 record |

Overly common keys (touching more S1 entities than a size cap) are dropped before the join to
keep the candidate set bounded and auditable; each S2/S3 record keeps its top-K (10) S1
candidates by summed key weight. Keys are hashed with a fixed CRC32/Adler32 combination rather
than Python's per-process-salted built-in `hash()`, which is required for reproducibility across
separate training/inference runs.

**Measured blocking recall** (on a representative development-scale subset; not re-measured at
full scale before this submission): approximately 98% overall, with a consistent gap between
US (~99%) and India (~97%) — native-script names and heavily truncated/summarized Indian
addresses (retaining only a city name, dropping all intermediate locality tokens) account for
most of the remaining India-side misses. This gap is a known, unresolved limitation of this
submission (see Section 5).

### 2.3 Feature engineering

33 features per (S1, candidate) pair, derived entirely from the normalized text:

- **Text similarity:** character 3-gram TF-IDF cosine similarity, computed separately for name
  and address, with vectorizers fit on the data being processed (so they adapt to any country's
  character statistics, including France, without retraining).
- **Name token overlap:** Jaccard and containment on full tokens and on 4-character token
  prefixes; first-token equality (full and prefix); token-count length ratio.
- **Transliteration-aware overlap:** 4-character prefix Jaccard/containment between the two
  records' name tokens, using the transliterated form for any Brahmic-script name.
- **Compact-name matching:** prefix equality and substring containment on the concatenated,
  legal-suffix-stripped name.
- **Legal-suffix signal:** whether the two records' legal-form tokens conflict, and whether the
  candidate side has none at all.
- **Address matching:** Jaccard on extracted numbers; any-number-in-common flag; Jaccard and
  containment on 4-character address-token prefixes; full alphabetic-token Jaccard.
- **Data-quality flags:** candidate address entirely empty; candidate name contains non-Latin
  characters; raw token/number counts.
- **Blocking-derived context:** the raw blocking score and key count, the pair's rank within its
  record's candidate list, and the score gap/ratio to the runner-up candidate for the same record.

`country` is deliberately excluded as a feature (used only inside blocking), so the trained model
generalizes to a country never seen in training.

### 2.4 Model and decision rule

- **Classifier:** LightGBM (gradient-boosted trees; MIT-licensed, well under the 8B-parameter
  cap), trained on all positive candidate pairs plus a **30% random sample of negative candidate
  pairs** (`ER_NEG_FRAC=0.3`). Negatives are subsampled **during** chunked candidate generation,
  not after, which is what keeps peak memory bounded at full scale. Source 2/3 were streamed in
  150,000-row chunks (`ER_CHUNK_ROWS=150000`).
- **Validation split:** by Source 1 entity (deterministic hash, 20% held out), never by pair, so a
  Source 1 entity's positive and negative pairs never cross the split.
- **Decision rule:** since each S2/S3 record matches at most one S1 entity, the single
  highest-scoring S1 candidate per record is accepted only if both:
  - predicted probability ≥ **0.6**, and
  - margin over the second-best candidate for the same record ≥ **0.3**.

  This is a **single global threshold/margin**, chosen by grid search (13 threshold values ×
  7 margin values = 91 combinations) to directly maximize macro F0.5 on the held-out validation
  split. The chosen setting sits in a stable plateau (validation F0.5 stayed between roughly
  0.936 and 0.939 across thresholds 0.4–0.75), not an isolated spike — evidence the setting is a
  genuine, not overfit, optimum for this run.
- Country-specific threshold tuning (US and India have measurably different optimal operating
  points; see Section 5) was **not** part of this submission's decision rule.

## 3. Results

**Held-out validation (20% of Source 1 entities, full training data, single global threshold):**
macro F0.5 = **0.9397** (threshold 0.6, margin 0.3).

**Actual leaderboard score on the real test set: 0.888.**

The gap between validation (0.9397) and the leaderboard score (0.888) is real and expected —
validation is measured on training-distribution data with, on average, an easier candidate pool
than the full test set (more entities, more near-duplicate names at scale, and a country — France
— with zero training coverage). This gap (~0.05) is substantially smaller than the gap observed
from an earlier, smaller-scale training attempt (a model trained on only ~7% of the training data
scored 0.809 on the same leaderboard against a similarly-measured ~0.98 validation number),
indicating this full-scale model generalizes considerably better, in line with its own validation
estimate.

**Test-set inference sanity check** (per-country candidate coverage and assignment rate, test
set, 1,732,544 Source 1 entities across US/India/France):

| Source/Country | Records | With ≥1 candidate | Assigned (matched) |
|---|---|---|---|
| S2 US | 1,871,330 | 99.4% | 60.4% |
| S2 India | 2,312,565 | 97.7% | 55.3% |
| S2 France | 703,378 | 98.7% | 62.5% |
| S3 US | 1,945,701 | 99.6% | 62.5% |
| S3 India | 2,405,000 | 97.0% | 54.7% |
| S3 France | 731,615 | 98.8% | 64.0% |

For comparison, the training data's own match rate is approximately 74–75% of Source 2/3 records
matched, and 5.6% of Source 1 entities with no match. Every country's assigned rate on the test
set sits well below this baseline (US and France by roughly 12–15 points, India by roughly 20
points), while candidate coverage stays high (97–99.6%) — indicating the model/threshold is
leaving real recall on the table (favoring precision, consistent with F0.5's weighting), with
India's gap being the largest. France's candidate coverage and assignment rate fall between the
US and India figures, not as an outlier in either direction — a reassuring sign that the
country-neutral feature design (no country used as a model feature, only as a blocking key)
generalizes reasonably to a country with zero training examples.

**Structural check:** every entity in `matching_results.tsv` had at least one row (1,732,544 /
1,732,544), zero matched IDs fell outside the corresponding `candidate_pairs.tsv` entry, and
93.8% of entities received at least one match (mean 3.38 matches per matched entity).

## 4. Reproduction

```
cd code/business_entity_resolution
pip install -r requirements.txt
export ER_DATA_DIR=/path/to/dataset      # contains train/ and test/

python src/train_model.py                # ER_TRAIN_SPLIT=train, streams the full training set,
                                          # writes work/model.pkl with the tuned threshold/margin
python src/predict.py --split test       # writes output/matching_results.tsv + candidate_pairs.tsv
```

Exact configuration used for this submission's training run:
```python
os.environ['ER_TRAIN_SPLIT'] = 'train'
os.environ['ER_NEG_FRAC'] = '0.3'
os.environ['ER_CHUNK_ROWS'] = '150000'
```

## 5. Known limitations and next steps

- **India recall is the largest identified gap.** Blocking recall on India-country records trails
  US by several points at development scale, and the test-set assignment-rate gap (~20 points
  below the training baseline, versus ~12–15 for US/France) is consistent with this being the
  dominant remaining weakness. Targeted fixes identified but **not yet included in this
  submission's model**: (a) correcting a transliteration bug where certain Devanagari/Bengali
  nasalization marks were silently dropped, corrupting common words; (b) collapsing long/short
  vowel-length distinctions in transliteration, since casual Romanization is inconsistent about
  this; (c) a de-leeting normalization step for OCR/typo-style digit-for-letter substitutions
  (e.g. "Ga1axy" → "Galaxy"); (d) an additional blocking key pairing a city-level address token
  with the business name's first token, to recover matches where the address was truncated down
  to just city/state.
- **A single global decision threshold was used**, despite evidence (from a leave-one-country-out
  stress test) that US and India have measurably different optimal operating points. Per-country
  threshold tuning was implemented after this submission and is expected to give a small,
  consistent improvement, but was not validated on a full-scale retrain before this write-up.
- **Blocking recall was not re-measured at full scale** before this submission; the ~98%
  figure quoted above is from a smaller development subset and may differ at the full data volume.
- **No external data, models, or APIs were used at any stage**, per the challenge's fair-play
  rules — all normalization dictionaries (legal suffixes, the Devanagari transliteration table)
  were hand-authored from general linguistic/Unicode-structure knowledge, not sourced from an
  external database, and all learned components (TF-IDF vectorizers, the classifier) are trained
  only on the provided train/test text itself.

## 6. Appendix: reconstructing this exact run

This document was written by reconstructing the configuration of the training run that produced
the submitted `matching_results.tsv`, from development logs and prior working sessions, after the
fact. Section 3's validation number (0.9397, threshold 0.6/margin 0.3), Section 2's feature/
blocking description, and Section 2.4's training configuration (`ER_NEG_FRAC=0.3`,
`ER_CHUNK_ROWS=150000`) are all confirmed directly against the run's own logged output and
settings. This run is therefore fully reproducible from the commands in Section 4. Recommended
practice going forward: retain each run's full first log line (which already prints every
relevant `ER_*` setting) alongside its results, so future configurations don't need to be
reconstructed after the fact.
