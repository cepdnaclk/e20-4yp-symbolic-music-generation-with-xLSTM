# MuseFormer Training Guide

This guide outlines the steps to train the MuseFormer model on your custom dataset.

## 1. Environment Setup

Ensure you are in the `museformer` conda environment.
The following dependencies should be installed (checked in `env/museformer_full_export.yaml`):
-   Python 3.8
-   PyTorch
-   Fairseq 0.10.2
-   Triton (User confirmed installation)

**Note**: You also need `MidiProcessor` to encode MIDI files. If not installed, install it:
```bash
pip install git+https://github.com/btyu/MidiProcessor.git
```

## 2. Data Preparation

### A. Place Data
Ensure your MIDI files are in a local directory, e.g., `data/midi`.
If you have already preprocessed them into tokens, verify they are in `data/token` (or a similar directory).

### B. Encode MIDI to Tokens (If starting from MIDI)
If your preprocessing outputs MIDI files, you must encode them into tokens first.

```bash
midi_dir=data/midi  # Your MIDI directory
token_dir=data/token
mkdir -p $token_dir

# Use REMIGEN (version 1) which supports various time signatures/instruments
mp-batch-encoding $midi_dir $token_dir --encoding-method REMIGEN --normalize-pitch-value --remove-empty-bars --sort-insts id
```
*Note: For general use (custom datasets), we remove `--ignore-ts` and change the encoding method to `REMIGEN` as per README, deviating from the LMD-full defaults.*

### C. Generate Metadata Splits
You need `train.txt`, `valid.txt`, and `test.txt` in `data/meta`. These lists define which files belong to which set.
**Note**: The original repository does **not** provide a script to generate these automatically.

I have created a helper script `tools/generate_splits.py` for you to automate this.

**Usage:**
```bash
# Assuming your source files are in data/midi (or wherever your source filenames come from)
# This generates the text files in data/meta
python tools/generate_splits.py data/midi --output_dir data/meta
```
This will randomly split your dataset (default: 80% train, 10% valid, 10% test).

### D. Choose Dictionary
You must use `data/meta/general_use_dict.txt`.
*   **Why**: The standard `dict.txt` only supports the 6 normalized instruments from the specific LMD-full dataset.
*   **General Use**: `general_use_dict.txt` supports all **128 MIDI instruments** (`i-0` to `i-127`) and standard tokens, ensuring compatibility with custom data.

## 3. Preprocess for Training

### A. Gather Tokens
Convert the file lists into consolidated token data files.

```bash
token_dir=data/token
split_dir=data/split
mkdir -p $split_dir

for split in train valid test; do
    # This tool reads the list of files from step 2C and concatenates their token content
    python tools/generate_token_data_by_file_list.py data/meta/${split}.txt $token_dir $split_dir
done
```

### B. Binarize Data
Convert text data to binary format for Fairseq.

```bash
data_bin_dir=data-bin/custom_dataset
mkdir -p data-bin

fairseq-preprocess \
  --only-source \
  --trainpref $split_dir/train.data \
  --validpref $split_dir/valid.data \
  --testpref $split_dir/test.data \
  --destdir $data_bin_dir \
  --srcdict data/meta/general_use_dict.txt \
  --workers 4
```

## 4. Training

Run the training command. You may need to create a new script or modify `ttrain/mf-lmd6remi-1.sh` to point to your new data.

**Command:**
```bash
# Example training command
fairseq-train data-bin/custom_dataset \
    --user-dir museformer \
    --task museformer_language_modeling \
    --arch museformer_lm_base \
    --criterion museformer_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0005 --warmup-updates 4000 --total-num-update 300000 \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --max-tokens 2048 --update-freq 4 \
    --max-update 300000 \
    --log-format simple --log-interval 100 \
    --save-interval-updates 5000 --keep-interval-updates 10 \
    --beat-mask-ts True \
    --sample-break-mode none \
    --fp16
```

**Important Modifications for General Use:**
-   Ensure `--beat-mask-ts True` is set (required for general use dictionary).
-   Adjust `--max-tokens` and `--update-freq` based on your GPU memory.
-   Point `fairseq-train` to your `data-bin/custom_dataset`.

## 5. Summary of Your Next Steps

1.  **Wait for Preprocessing**: Once your separate process finishes, identify where the output is.
2.  **Generate Splits**: Run `python tools/generate_splits.py <your_data_dir>` to create the metadata files.
3.  **Run Pipeline**: Execute Step 3A (Gather Tokens) and Step 3B (Binarize).
4.  **Train**: Launch the training script (Step 4).
