# LMD-full Preprocessing Plan (Notebook → Custom Script)

Goal: Use a notebook to **analyze LMD-full**, identify the **right preprocessing parameters**, then implement a **custom preprocessing script** inspired by Microsoft’s approach (but tailored to our pipeline).

---

## Phase A — Notebook-driven dataset understanding (choose parameters first)

### A1) Build a dataset manifest (index)
**Task**
- Recursively enumerate all `.mid/.midi` files under `data/raw/lmd_full/`
- Save a stable manifest:

**Output**
- `data/processed/manifests/lmd_full_manifest.csv`

**Suggested columns**
- `path`, `size_bytes`, `folder_prefix`
- optional: `md5` (for dedup/debug)

**Why**
- Reproducible file list + faster reruns without rescanning.

---

### A2) MIDI health check + failure taxonomy
**Task**
- Parse a sample (start with ~1k, then 10k) using a consistent library (e.g., `miditoolkit`)
- Track failure types:
  - unreadable, parse error, empty/blank, malformed meta events, etc.

**Outputs**
- `data/processed/reports/parse_failures.csv`
- `data/processed/reports/parse_success_sample_stats.csv`

**Why**
- Understand noise level + define robust error handling rules.

---

### A3) Collect dataset statistics that influence preprocessing design
On a representative sample (5k–20k files), compute distributions for:

**Core musical statistics**
- duration (seconds; bars if estimated)
- note count, note density
- tempo (mean/median; extreme outliers)
- time signatures frequency (4/4 vs others)
- pitch range usage
- polyphony estimate (max simultaneous notes)
- program/instrument usage (piano-only vs multi-track)

**Timing/quantization indicators**
- onset grid-fit (does 1/16 beat fit well? 1/24? messy?)
- typical resolution / rhythmic complexity

**Outcome**
- Evidence-backed choices for:
  - quantization grid (`pos_resolution`)
  - tempo clipping bounds
  - time signature filtering (if any)
  - truncation / maximum length rules

---

### A4) Write a “Preprocessing Spec” (your target design)
Create a short spec (YAML/Markdown) capturing decisions:

**I/O**
- input root: `data/raw/lmd_full/`
- output roots: `data/processed/...`

**Filtering**
- min notes threshold
- max duration / max bars
- tempo bounds (min/max)
- time signature policy (keep all vs restrict)
- track/instrument policy (keep multi-track vs piano-only)

**Normalization**
- quantization grid
- velocity handling (keep/clip/bucket/drop)
- multi-track handling (merge vs keep tracks)
- sustain pedal handling (ignore vs convert)

**Segmentation**
- full piece vs chunking
- if chunking: window length + overlap

**Deduplication**
- on/off
- if on: define canonical representation for hashing

**Splits**
- split method (leak-resistant + reproducible)

**Why**
- Prevent “coding blind”; the script becomes an implementation of this spec.

---

### A5) Prototype a mini-preprocess inside the notebook (end-to-end)
**Task**
- Implement a minimal pipeline in notebook cells for 50–200 files:
  - parse → clean → quantize → export debug representation

**Manual inspection**
- rhythms preserved?
- tempo sane?
- quantization creates too many collisions?
- truncation too aggressive?

**Outcome**
- Debug assumptions before writing the full script.

---

## Phase B — Implement the custom preprocessing script

### B1) Create a CLI script skeleton
Create:
- `scripts/preprocess_lmd.py`

**Must support**
- config file input (YAML/JSON)
- `--input_root`, `--output_dir`, `--num_workers`
- `--limit` (debug subset)
- robust logging + summary output

---

### B2) Implement pipeline as modular functions (testable)
Recommended internal stages:

1. `load_midi(path) -> midi_obj | error`
2. `extract_notes(midi_obj) -> notes`
3. `compute_stats(notes, midi_obj) -> stats`
4. `filter_piece(stats, config) -> keep/drop + reason`
5. `normalize(notes, config) -> normalized_notes`
6. `encode(normalized_notes, config) -> encoded_sequence`
7. `segment(encoded_sequence, config) -> sequences`
8. `write_outputs(sequences, split, ...)`

**Why**
- Easy unit testing + easy iteration on a single stage.

---

### B3) Deduplication (if enabled)
**Approach**
- Define canonical encoding string *after* normalization/quantization
- Hash it (e.g., SHA1/MD5)
- Track seen hashes → drop duplicates

**Outputs**
- duplicates count + kept canonical instance
- `data/processed/reports/dedup_log.csv` (optional)

---

### B4) Splitting strategy (avoid naive random split)
At minimum:
- deterministic split based on stable hash:
  - file path hash (ok-ish), OR
  - canonical encoding hash (better against content duplicates)

**Outputs**
- `data/splits/train.txt`
- `data/splits/valid.txt`
- `data/splits/test.txt`

**Why**
- Reproducible splits + reduced leakage risk.

---

### B5) Output format aligned with tokenization
Decide output format early:
- 1 sequence per line?
- special tokens needed? (`<s>`, `</s>`)
- vocabulary generation (`dict.txt`)?
- separate files for each split?

**Outputs**
- `data/processed/encoded/train.txt`
- `data/processed/encoded/valid.txt`
- `data/processed/encoded/test.txt`
- `data/processed/encoded/dict.txt` (if applicable)

---

## Phase C — Validation + full run

### C1) Small subset run and compare with notebook expectations
**Task**
- Run script on the same sample set used in notebook
- Compare:
  - kept/drop rates
  - length distributions
  - token distributions
  - failure categories

---

### C2) Full dataset run (with robust logging)
**Must produce**
- `summary.json` (counts, kept/dropped reasons, basic stats)
- `errors.csv` (path, error_type, exception)
- `kept_manifest.csv` (final file list)

---

### C3) Post-run sanity checks
- sequence length distributions (too long/short?)
- token frequency distribution (degenerate tokens?)
- sample decode/round-trip tests (if decoder exists)
- spot-check a few long pieces for structure preservation

---

## Final Artifacts (clean + reproducible)
- `data/processed/manifests/lmd_full_manifest.csv`
- `data/processed/reports/*`
- `data/splits/{train,valid,test}.txt`
- `data/processed/encoded/{train,valid,test}.txt`
- `scripts/preprocess_lmd.py`
- `configs/preprocess_lmd.yaml`

---
