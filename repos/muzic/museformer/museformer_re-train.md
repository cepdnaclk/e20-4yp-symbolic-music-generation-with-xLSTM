# Museformer Re-Training Guide (Custom LMD Dataset)

> **Goal:** Retrain Museformer from scratch on our custom 45,043-file LMD-Full dataset,
> preprocessed with the same 4/4 + 6-track normalization pipeline used in the original paper.

---

## Paths Reference

| Purpose | Path |
|---------|------|
| Museformer repo root | `repos/muzic/museformer/` |
| MIDI source (preprocessed) | `data/museformer_baseline/full/06_pitch_normalize/` |
| **MIDI input for tokenization** | `repos/muzic/museformer/data/midi/` ← copy files here first |
| Split lists (train/valid/test) | `data/meta/{train,valid,test}.txt` |
| Token output dir (to create) | `repos/muzic/museformer/data/token/` |
| Split token data (to create) | `repos/muzic/museformer/data/split/` |
| Binary data (to create) | `repos/muzic/museformer/data-bin/lmd6remi/` |
| Checkpoints (auto-created) | `repos/muzic/museformer/checkpoints/mf-lmd6remi-1/` |

All commands below are run from the **Museformer repo root** unless stated otherwise:

```bash
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/muzic/museformer
```

---

## Task Checklist

### Stage 1 — Environment Setup

Run the following verification commands first. All should pass before continuing.

```bash
# 1.1 — Activate the museformer conda environment
conda activate museformer

# 1.2 — Verify Python version (must be 3.8.x)
python --version

# 1.3 — Verify CUDA version (must be 11.x)
nvcc --version
# or:
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

# 1.4 — Verify fairseq version (must be 0.10.2)
# NOTE: fairseq-train --version does NOT work; use pip show instead
pip show fairseq | grep -E 'Name|Version'
# Expected: Version: 0.10.2

# 1.5 — Verify MidiProcessor is installed
pip show midi-processor | grep -E 'Name|Version'
# If missing: pip install midi-processor

# 1.6 — Verify triton is installed (needed for CUDA FC-Attention kernels)
pip show triton | grep -E 'Name|Version'
# If missing, install from source outside repo root:
# git clone https://github.com/openai/triton.git /tmp/triton
# cd /tmp/triton/python && pip install -e .

# 1.7 — Verify GPUs
nvidia-smi --list-gpus
```

- [ ] **1.1** `conda activate museformer`
- [ ] **1.2** Python == 3.8.x
- [ ] **1.3** CUDA == 11.x, PyTorch CUDA matches
- [ ] **1.4** fairseq == 0.10.2 (via `pip show fairseq`)
- [ ] **1.5** MidiProcessor installed
- [ ] **1.6** triton installed
- [ ] **1.7** At least 1 GPU visible

---

### Stage 2 — MIDI Tokenization
Encode all 45,043 MIDI files into REMIGEN2 token format.

```bash
# Copy your preprocessed MIDIs into the museformer repo first:
# (using rsync to safely handle 45k+ files without hitting shell argument limits)
rsync -a --include="*.mid" --exclude="*" \
  /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/06_pitch_normalize/ \
  data/midi/
# Verify count:
find data/midi -maxdepth 1 -name "*.mid" | wc -l   # should be 45,043

midi_dir=data/midi
token_dir=data/token

mkdir -p $token_dir

mp-batch-encoding $midi_dir $token_dir \
  --encoding-method REMIGEN2 \
  --normalize-pitch-value \
  --remove-empty-bars \
  --ignore-ts \
  --sort-insts 6tracks_cst1
```

**Arguments explained:**
| Flag | Meaning |
|------|---------|
| `--encoding-method REMIGEN2` | REMI-like representation used by Museformer |
| `--normalize-pitch-value` | Transpose songs to C major / A minor |
| `--remove-empty-bars` | Strip leading/trailing silence |
| `--ignore-ts` | Skip time-signature tokens (all data is 4/4) |
| `--sort-insts 6tracks_cst1` | Fix track order: synth, drum, bass, guitar, piano, string |

**Expected output:** one `.txt` token file per MIDI in `data/token/`

- [ ] **2.1** Run encoding command
- [ ] **2.2** Spot-check a few token files (`head -1 data/token/<some_file>.mid.txt`)
- [ ] **2.3** Verify token count matches MIDI count:
  ```bash
  ls data/token/*.mid.txt | wc -l   # should be ~45,043
  ```

---

### Stage 3 — Gather Tokens by Split
Concatenate per-file token files into three flat data files (one per split),
using our custom `train.txt`, `valid.txt`, `test.txt` lists.

```bash
token_dir=data/token
split_dir=data/split
meta_dir=/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/meta

mkdir -p $split_dir

for split in train valid test; do
    python tools/generate_token_data_by_file_list.py \
        $meta_dir/${split}.txt \
        $token_dir \
        $split_dir
done
```

> **Note:** The script looks for `<filename>.mid.txt` in `token_dir` for each
> line in the split list. Ensure filenames match exactly.

**Expected output:** `data/split/{train,valid,test}.data`

- [ ] **3.1** Run the loop above
- [ ] **3.2** Confirm three `.data` files exist:
  ```bash
  ls -lh data/split/
  ```

---

### Stage 4 — Binarize with fairseq-preprocess
Convert the flat token files into fairseq binary format for fast training.

```bash
split_dir=data/split
data_bin_dir=data-bin/lmd6remi

mkdir -p data-bin

fairseq-preprocess \
  --only-source \
  --trainpref $split_dir/train.data \
  --validpref $split_dir/valid.data \
  --testpref  $split_dir/test.data \
  --destdir   $data_bin_dir \
  --srcdict   data/meta/dict.txt \
  --workers   $(nproc)
```

> **Important:** We reuse `data/meta/dict.txt` from the original Museformer repo.
> This is the vocabulary for the REMIGEN2 encoding with 6 fixed tracks.
> Do **not** regenerate it — using the same vocab ensures compatibility
> with the pre-trained weights if you want to fine-tune later.

**Expected output:** binary `*.bin` / `*.idx` files in `data-bin/lmd6remi/`

- [ ] **4.1** Run `fairseq-preprocess`
- [ ] **4.2** Verify output:
  ```bash
  ls data-bin/lmd6remi/
  # Should contain: train.bin, train.idx, valid.bin, valid.idx, test.bin, test.idx, dict.txt
  ```

---

### Stage 5 — Train
Launch Museformer training. The training script is at `ttrain/mf-lmd6remi-1.sh`.

```bash
bash ttrain/mf-lmd6remi-1.sh
```

**Key hyperparameters** (edit `ttrain/mf-lmd6remi-1.sh` if needed):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `PEAK_LR` | `5e-4` | Learning rate — reduce if loss spikes |
| `WARMUP_UPDATES` | `16000` | LR warmup steps |
| `--max-update` | `1000000` | Total training steps |
| `--save-interval-updates` | `5000` | Save checkpoint every N steps |
| `--truncate-train` | `15360` | Max sequence length during training |
| `--truncate-valid` | `10240` | Max sequence length during validation |
| `--batch-size` | `1` | Do not change (Museformer constraint) |
| `UPDATE_FREQ` | `1` | Effective batch = GPUs × UPDATE_FREQ |
| `--num-layers` | `4` | Transformer depth |

**GPUs:** The original paper used 4× V100 32GB. If you have fewer GPUs,
increase `UPDATE_FREQ` to compensate (e.g. 2 GPUs → `UPDATE_FREQ=2`).

**Monitoring training:**
```bash
# Live log
tail -f log/mf-lmd6remi-1.log

# TensorBoard
tensorboard --logdir tb_log/mf-lmd6remi-1 --port 6006
```

- [ ] **5.1** Check GPU count: `nvidia-smi --list-gpus | wc -l`
- [ ] **5.2** Adjust `UPDATE_FREQ` in `ttrain/mf-lmd6remi-1.sh` if needed
- [ ] **5.3** Launch training: `bash ttrain/mf-lmd6remi-1.sh`
- [ ] **5.4** Confirm first checkpoint saves at step 5000
- [ ] **5.5** Monitor loss curve in TensorBoard

---

### Stage 6 — Evaluate (Perplexity)
Once you have a checkpoint, evaluate on the test set:

```bash
bash tval/val__mf-lmd6remi-x.sh 1 checkpoint_best.pt 10240
```

Arguments: `<model_suffix> <checkpoint_filename> <max_seq_len>`

- [ ] **6.1** Run evaluation on `checkpoint_best.pt`
- [ ] **6.2** Record test-set perplexity for comparison with the paper baseline
  (paper reports PPL on LMD with ~29,940 files — expect slightly lower PPL
  with our larger 45,043-file dataset)

---

### Stage 7 — Inference (Optional)
Generate music samples:

```bash
mkdir -p output_log
seed=1
printf '\n\n\n\n\n' | bash tgen/generation__mf-lmd6remi-x.sh 1 checkpoint_best.pt ${seed} \
  | tee output_log/generation.log

# Extract token sequences from the log
python tools/batch_extract_log.py output_log/generation.log output/generation --start_idx 1

# Convert tokens → MIDI
python tools/batch_generate_midis.py \
  --encoding-method REMIGEN2 \
  --input-dir output/generation \
  --output-dir output/generation
```

- [ ] **7.1** Generate 5 samples
- [ ] **7.2** Verify MIDI files exist in `output/generation/`
- [ ] **7.3** Listen to generated samples

---

## Common Issues & Tips

| Issue | Fix |
|-------|-----|
| `mp-batch-encoding` not found | Install MidiProcessor: `pip install midi-processor` |
| fairseq version error | Must use exactly `fairseq==0.10.2` |
| CUDA kernel compile error on first run | Normal — Triton compiles once; wait ~5 min |
| OOM during training | Reduce `--truncate-train` (try `10240` or `8192`) |
| Token file not found for a MIDI | That file failed tokenization; check logs and remove from split list |
| Loss NaN early in training | Reduce `PEAK_LR` to `1e-4` |

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Source | LMD-Full |
| Total files | 45,043 |
| Train | 36,034 (80%) |
| Valid | 4,504 (10%) |
| Test | 4,505 (10%) |
| Split method | Artist-stratified (8,205 unique artists) + random fallback |
| Split seed | 42 |
| Time signature | 4/4 only |
| Tracks | 6 (synth, drum, bass, guitar, piano, string) |
| Pitch normalization | C major / A minor |
| Encoding | REMIGEN2 |
