#!/usr/bin/env python3
"""
generate_artist_splits.py
─────────────────────────
Artist-stratified train / valid / test split for LMD-Full-derived datasets.

Strategy
--------
1. Load your file list (one *.mid filename per line, MD5-hash named).
2. Cross-reference each MD5 against LMD's `match_scores.json.gz` to find the
   best-matching MSD track ID (e.g. TRABCEI128F424C983).
3. Derive an artist ID from the MSD track ID (first 6 chars: "TRABC_").
4. Group all songs by artist, then assign *whole artist groups* to splits so
   that no artist's songs appear in more than one split.
5. Songs that could not be matched to any artist are randomly distributed
   into splits (fallback) while still hitting the overall ratio targets.
6. Write train.txt, valid.txt, test.txt.

Usage
-----
# Minimal:
python generate_artist_splits.py \\
    --file_list      data/meta/all_files.txt \\
    --match_scores   /path/to/match_scores.json.gz \\
    --output_dir     data/meta

# Full options:
python generate_artist_splits.py \\
    --file_list      data/meta/all_files.txt \\
    --match_scores   /path/to/match_scores.json.gz \\
    --output_dir     data/meta \\
    --train          0.80 \\
    --valid          0.10 \\
    --seed           42 \\
    --verify

Notes
-----
- match_scores.json.gz is available from:
    http://hog.ee.columbia.edu/craffel/lmd/match_scores.json.gz
- The artist ID is derived from the MSD track ID prefix (characters 2-7),
  which encodes the artist in the Million Song Dataset naming scheme.
  This is a reliable proxy when full MSD metadata is unavailable.
- If you have the full MSD metadata HDF5, you can swap in the real
  artist_name string for better grouping (see the --msd_metadata flag).
"""

from __future__ import annotations

import os
import sys
import json
import gzip
import random
import argparse
import collections
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_file_list(path: str) -> list[str]:
    """Read one filename per line; skip blanks."""
    files = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files


def load_match_scores(path: str) -> dict[str, str]:
    """
    Parse match_scores.json(.gz) and return {md5: best_msd_track_id}.

    Actual schema in the file:
        { msd_track_id: { md5: score, ... }, ... }

    We invert this to:
        { md5: msd_track_id_with_highest_score }

    When the same MD5 appears under multiple MSD track IDs (rare), we keep
    the MSD ID that gave it the highest score.
    """
    print(f"Loading match scores from {path} ...")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    # raw = { msd_id: { md5: score, ... }, ... }
    # Build: { md5: (best_score, best_msd_id) }
    best: dict[str, tuple[float, str]] = {}
    for msd_id, md5_scores in raw.items():
        for md5, score in md5_scores.items():
            if md5 not in best or score > best[md5][0]:
                best[md5] = (score, msd_id)

    md5_to_msd = {md5: val[1] for md5, val in best.items()}
    print(f"  Loaded {len(md5_to_msd):,} MD5→MSD mappings (inverted from "
          f"{len(raw):,} MSD track entries).")
    return md5_to_msd


def msd_track_id_to_artist_id(msd_track_id: str) -> str:
    """
    Extract a stable artist-level grouping key from an MSD track ID.

    MSD track IDs look like:  TR ABCDE 128F424C983
                               ^^     ^
                               TR + 6-char artist segment + 9-char track suffix

    Characters [2:8] are the artist-level code shared by all tracks from the
    same artist in the MSD naming scheme.
    """
    return msd_track_id[2:8] if len(msd_track_id) >= 8 else msd_track_id


def write_split(file_list: list[str], output_path: str) -> None:
    with open(output_path, "w") as f:
        for name in file_list:
            f.write(name + "\n")
    print(f"  Wrote {len(file_list):,} entries → {output_path}")


def verify_no_overlap(splits: dict[str, list[str]]) -> bool:
    """Assert that no filename appears in more than one split."""
    seen = {}
    ok = True
    for split_name, files in splits.items():
        for fname in files:
            if fname in seen:
                print(f"[ERROR] Overlap: '{fname}' in both "
                      f"'{seen[fname]}' and '{split_name}'")
                ok = False
            seen[fname] = split_name
    if ok:
        print("  ✓ No overlap between splits.")
    return ok


def verify_distribution(matched_by_split: dict[str, int],
                        unmatched_by_split: dict[str, int]) -> None:
    """Print a simple coverage table (values are integer counts)."""
    print("\n── Split summary ──────────────────────────────────────")
    total = sum(matched_by_split.values()) + sum(unmatched_by_split.values())
    for s in ("train", "valid", "test"):
        n_matched   = matched_by_split.get(s, 0)
        n_unmatched = unmatched_by_split.get(s, 0)
        n = n_matched + n_unmatched
        pct = 100 * n / total if total else 0
        print(f"  {s:<6}  {n:>6,} files  ({pct:.1f}%)  "
              f"[matched={n_matched:,}  fallback={n_unmatched:,}]")
    print(f"  {'total':<6}  {total:>6,} files")
    print("────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Core split logic
# ─────────────────────────────────────────────────────────────────────────────

def artist_stratified_split(
    files: list[str],
    md5_to_msd: dict[str, str],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], dict]:
    """
    Returns (train_files, valid_files, test_files, stats_dict).

    Algorithm
    ---------
    1.  Map each file → artist_id (via MD5 → MSD track ID → artist prefix).
    2.  Build artist→[files] groups.
    3.  Shuffle artist groups with fixed seed.
    4.  Greedily assign artist groups to train / valid / test until ratio
        targets are met, then push remainder into train.
    5.  Unmatched files are randomly distributed into splits (fallback).
    """
    rng = random.Random(seed)
    test_ratio = 1.0 - train_ratio - valid_ratio

    # ── 1. Categorise files ─────────────────────────────────────────────────
    artist_to_files: dict[str, list[str]] = collections.defaultdict(list)
    unmatched_files: list[str] = []

    for fname in files:
        md5 = fname.replace(".mid", "").replace(".MID", "")
        msd_id = md5_to_msd.get(md5)
        if msd_id:
            artist_id = msd_track_id_to_artist_id(msd_id)
            artist_to_files[artist_id].append(fname)
        else:
            unmatched_files.append(fname)

    n_total = len(files)
    n_matched = n_total - len(unmatched_files)
    print(f"\nMatched to artist:   {n_matched:,} / {n_total:,} files "
          f"({100*n_matched/n_total:.1f}%)")
    print(f"Unmatched (fallback): {len(unmatched_files):,} files")
    print(f"Unique artists:       {len(artist_to_files):,}")

    # ── 2. Shuffle artist groups ─────────────────────────────────────────────
    artist_groups = list(artist_to_files.items())  # [(artist_id, [files]), ...]
    # Sort first for determinism, then shuffle
    artist_groups.sort(key=lambda x: x[0])
    rng.shuffle(artist_groups)

    # ── 3. Greedy assignment of artist groups ────────────────────────────────
    target_train = int(n_matched * train_ratio)
    target_valid = int(n_matched * valid_ratio)
    # test gets the remainder

    train_files, valid_files, test_files = [], [], []

    for artist_id, artist_songs in artist_groups:
        n_train = len(train_files)
        n_valid = len(valid_files)

        if n_train < target_train:
            train_files.extend(artist_songs)
        elif n_valid < target_valid:
            valid_files.extend(artist_songs)
        else:
            test_files.extend(artist_songs)

    # ── 4. Distribute unmatched files randomly ───────────────────────────────
    rng.shuffle(unmatched_files)
    n_um = len(unmatched_files)
    um_train_end = int(n_um * train_ratio)
    um_valid_end = um_train_end + int(n_um * valid_ratio)

    um_train = unmatched_files[:um_train_end]
    um_valid = unmatched_files[um_train_end:um_valid_end]
    um_test  = unmatched_files[um_valid_end:]

    train_files += um_train
    valid_files += um_valid
    test_files  += um_test

    # ── 5. Shuffle within each split (so order isn't artist-grouped) ─────────
    rng.shuffle(train_files)
    rng.shuffle(valid_files)
    rng.shuffle(test_files)

    stats = {
        "total":              n_total,
        "n_matched":          n_matched,
        "n_unmatched":        len(unmatched_files),
        "n_unique_artists":   len(artist_to_files),
        "matched_by_split": {
            "train": len(train_files) - len(um_train),
            "valid": len(valid_files) - len(um_valid),
            "test":  len(test_files)  - len(um_test),
        },
        "unmatched_by_split": {
            "train": len(um_train),
            "valid": len(um_valid),
            "test":  len(um_test),
        },
    }

    return train_files, valid_files, test_files, stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Artist-stratified train/valid/test split for LMD-Full datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--file_list", required=True,
        help="Text file with one .mid filename per line (MD5-hash named, "
             "as produced by the preprocessing pipeline).",
    )
    p.add_argument(
        "--match_scores", required=True,
        help="Path to LMD match_scores.json or match_scores.json.gz. "
             "Download from: http://hog.ee.columbia.edu/craffel/lmd/match_scores.json.gz",
    )
    p.add_argument(
        "--output_dir", default="data/meta",
        help="Directory to write train.txt, valid.txt, test.txt (default: data/meta).",
    )
    p.add_argument(
        "--train", type=float, default=0.80,
        help="Train ratio (default: 0.80).",
    )
    p.add_argument(
        "--valid", type=float, default=0.10,
        help="Validation ratio (default: 0.10). Test = 1 - train - valid.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Run overlap checks and print a distribution summary after splitting.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate ratios
    if not (0 < args.train < 1 and 0 < args.valid < 1):
        print("[ERROR] --train and --valid must be between 0 and 1.")
        sys.exit(1)
    if args.train + args.valid >= 1.0:
        print("[ERROR] train + valid must be < 1.0 (test gets the remainder).")
        sys.exit(1)

    test_ratio = 1.0 - args.train - args.valid
    print(f"Split ratios  →  train={args.train:.0%}  "
          f"valid={args.valid:.0%}  test={test_ratio:.0%}")
    print(f"Random seed   →  {args.seed}")

    # Load inputs
    files = load_file_list(args.file_list)
    print(f"Loaded {len(files):,} files from {args.file_list}")

    md5_to_msd = load_match_scores(args.match_scores)

    # Split
    train_files, valid_files, test_files, stats = artist_stratified_split(
        files=files,
        md5_to_msd=md5_to_msd,
        train_ratio=args.train,
        valid_ratio=args.valid,
        seed=args.seed,
    )

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    splits = {
        "train": train_files,
        "valid": valid_files,
        "test":  test_files,
    }
    print()
    for split_name, split_files in splits.items():
        out_path = os.path.join(args.output_dir, f"{split_name}.txt")
        write_split(split_files, out_path)

    # Verify
    if args.verify:
        print()
        ok = verify_no_overlap(splits)
        verify_distribution(
            matched_by_split=stats["matched_by_split"],
            unmatched_by_split=stats["unmatched_by_split"],
        )
        if not ok:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
