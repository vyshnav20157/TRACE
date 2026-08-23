"""
Build the Memotion "skeleton" JSON that the captioning and training scripts expect.

This mirrors `utils/build_fhm_skeleton.py` and `MAMI/build_mami_skeleton.py`, but for
Memotion 1.0 (SemEval-2020 Task 8). The captioning script (`Memotion/memotion_cap_gen.py`)
and the training scripts both read a single JSON (records orient) with the columns:

    img, text, humour_label, sarcasm_label, offensive_label, split, ivl_caption_unified

Unlike FHM/MAMI there is no single `label` column in the JSON: Memotion Task B is three
independent binary problems over the same memes, so all three labels are stored and the
training script projects the chosen one onto `label` at load time (see
`memotion_common.apply_task_labels`). Captions are therefore generated ONCE and shared by
all three tasks.

Input CSVs (already binarized by the dataset's own eda_binarize.py):
    train_binary.csv, test_binary.csv
with columns:
    image_name, text_ocr, text_corrected, humour_bin, sarcasm_bin, offensive_bin,
    motivational_bin, <*_ord>, overall_sentiment, sentiment_ord

Images live in two folders keyed by split:
    train_meme_images/  -> train_binary.csv filenames
    test_meme_images/   -> test_binary.csv filenames

We store `img` as a split-aware relative path (e.g. "train_meme_images/foo.jpg") so the
downstream `f'{MEMOTION_IMAGE_ROOT}/{img}'` construction resolves with a single root.

Validation split
----------------
Memotion ships only train and test. Early stopping, LR scheduling, and caption selection
all need a held-out set that is NOT the test set, so this builder carves a deterministic
VAL_FRACTION (10%) of train into a `val` split, stratified on the JOINT (humour, sarcasm,
offensive) label combination. A joint stratification means one shared split works for all
three tasks -- each task's marginal positive rate is preserved to within a fraction of a
percent, so humour, offensive, and sarcasm runs are trained on exactly the same rows and
remain comparable to each other.

The `motivational` task and the ordinal / sentiment columns are carried through unused --
Memotion Task A (sentiment) and the 4th Task B dimension are out of scope.

Usage:
    python Memotion/build_memotion_skeleton.py
    python Memotion/build_memotion_skeleton.py --output /path/to/memotion.json
"""

import argparse
import os

import numpy as np
import pandas as pd

from memotion_common import (
    MEMOTION_DATA_PATH,
    MEMOTION_IMAGE_ROOT,
    MEMOTION_TEST_CSV,
    MEMOTION_TRAIN_CSV,
    TASKS,
    TASK_ORDER,
    VAL_FRACTION,
    VAL_SPLIT_SEED,
)

# split name -> image subfolder.
SPLIT_SUBFOLDER = {
    "train": "train_meme_images",
    "val": "train_meme_images",  # val is carved out of train, so same image folder
    "test": "test_meme_images",
}

# Caption fields the downstream scripts read. Start empty; backfilled by the pipeline.
# Only InternVL is used (Gemini was dropped from the Memotion flow).
CAPTION_FIELDS = ["ivl_caption_unified"]

# Carried through for reference but never trained on.
EXTRA_FIELDS = [
    "motivational_bin",
    "humour_ord",
    "sarcasm_ord",
    "offensive_ord",
    "motivational_ord",
    "overall_sentiment",
    "sentiment_ord",
]


def load_csv(path, subfolder, split_name):
    """Read one Memotion binary CSV into the skeleton's column contract."""
    if not os.path.exists(path):
        raise SystemExit(f"Memotion CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"image_name", "text_corrected"} | {cfg["csv_column"] for cfg in TASKS.values()}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing expected column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    # Store a split-aware relative image path so `f'{root}/{img}'` resolves everywhere.
    out["img"] = df["image_name"].apply(lambda f: f"{subfolder}/{f}")

    # `text_corrected` is the human-cleaned OCR and is what we model; fall back to the raw
    # OCR when it is blank, and to empty string when both are (13 such rows in train).
    text = df["text_corrected"].fillna("")
    if "text_ocr" in df.columns:
        text = text.where(text.astype(str).str.strip() != "", df["text_ocr"].fillna(""))
    out["text"] = text.astype(str)

    # All three task labels live side by side; training picks one via --task.
    for task in TASK_ORDER:
        cfg = TASKS[task]
        out[cfg["json_field"]] = df[cfg["csv_column"]].astype(int)

    out["split"] = split_name

    for col in EXTRA_FIELDS:
        if col in df.columns:
            out[col] = df[col]

    print(f"  {split_name:5s}: {len(out):5d} rows from {os.path.basename(path)} (images in {subfolder}/)")
    return out


def carve_val_split(train_df, val_fraction, seed):
    """Move a stratified `val_fraction` of train rows into a 'val' split, in place.

    Stratifies on the joint (humour, sarcasm, offensive) label triple so a single shared
    split preserves every task's positive rate. Rare strata (fewer rows than 1/val_fraction)
    contribute proportionally via rounding, and the sampling is fully determined by `seed`.
    """
    fields = [TASKS[t]["json_field"] for t in TASK_ORDER]
    stratum = train_df[fields].astype(str).agg("-".join, axis=1)

    rng = np.random.RandomState(seed)
    val_index = []
    for key, group in train_df.groupby(stratum, sort=True):
        n_val = int(round(len(group) * val_fraction))
        # Never take a whole stratum, and take at least one row from strata big enough.
        n_val = min(n_val, max(len(group) - 1, 0))
        if n_val <= 0:
            continue
        chosen = rng.choice(group.index.values, size=n_val, replace=False)
        val_index.extend(chosen.tolist())

    train_df = train_df.copy()
    train_df.loc[val_index, "split"] = "val"
    return train_df


def report_split_balance(data):
    """Print per-task positive rates for each split, to verify the stratification held."""
    print("\nPer-task positive rate by split:")
    header = f"  {'split':6s} {'rows':>6s}" + "".join(f"{t:>14s}" for t in TASK_ORDER)
    print(header)
    for split in ["train", "val", "test"]:
        sub = data[data["split"] == split]
        if sub.empty:
            continue
        rates = "".join(
            f"{sub[TASKS[t]['json_field']].mean():>13.3f} " for t in TASK_ORDER
        )
        print(f"  {split:6s} {len(sub):6d}  {rates}")


def verify_images(data, image_root, sample=None):
    """Check that `img` paths resolve under `image_root`. Returns the missing count."""
    paths = data["img"]
    if sample is not None and sample < len(paths):
        paths = paths.sample(n=sample, random_state=0)
    missing = [p for p in paths if not os.path.exists(os.path.join(image_root, p))]
    if missing:
        print(f"\nWARNING: {len(missing)} image path(s) do not resolve under {image_root}.")
        for p in missing[:5]:
            print(f"  missing: {p}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    else:
        print(f"\nAll {len(paths)} checked image paths resolve under {image_root}.")
    return len(missing)


def build_skeleton(train_csv, test_csv, val_fraction, seed):
    train_df = load_csv(train_csv, SPLIT_SUBFOLDER["train"], "train")
    test_df = load_csv(test_csv, SPLIT_SUBFOLDER["test"], "test")

    if val_fraction > 0:
        train_df = carve_val_split(train_df, val_fraction, seed)

    data = pd.concat([train_df, test_df], ignore_index=True)

    # Initialise empty caption fields for the pipeline to backfill.
    for col in CAPTION_FIELDS:
        data[col] = ""

    # Stable, explicit column order: core contract first, then the unused extras.
    label_fields = [TASKS[t]["json_field"] for t in TASK_ORDER]
    ordered = ["img", "text"] + label_fields + ["split"] + CAPTION_FIELDS
    extra = [c for c in data.columns if c not in ordered]
    return data[ordered + extra]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-csv", default=MEMOTION_TRAIN_CSV,
                        help="Memotion binarized train CSV.")
    parser.add_argument("--test-csv", default=MEMOTION_TEST_CSV,
                        help="Memotion binarized test CSV.")
    parser.add_argument("--image-root", default=MEMOTION_IMAGE_ROOT,
                        help="Root that `img` paths are resolved against (for verification).")
    parser.add_argument("--output", default=MEMOTION_DATA_PATH,
                        help="Path to write the skeleton JSON (records orient).")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                        help="Fraction of train held out as val (0 disables the val split).")
    parser.add_argument("--seed", type=int, default=VAL_SPLIT_SEED,
                        help="Seed fixing the val split (keep constant across tasks).")
    parser.add_argument("--verify-images", action="store_true",
                        help="Check that every img path resolves before writing.")
    args = parser.parse_args()

    print(f"Reading Memotion CSVs:\n  train: {args.train_csv}\n  test:  {args.test_csv}")
    data = build_skeleton(args.train_csv, args.test_csv, args.val_fraction, args.seed)

    print(f"\nTotal rows: {len(data)}")
    print("\nSplit distribution:")
    print(data["split"].value_counts().reindex(["train", "val", "test"]).to_string())

    print("\nLabel distribution per task (whole dataset):")
    for task in TASK_ORDER:
        field = TASKS[task]["json_field"]
        counts = data[field].value_counts().sort_index().to_dict()
        print(f"  {task:10s} 0={counts.get(0, 0):5d}  1={counts.get(1, 0):5d}")

    report_split_balance(data)

    if args.verify_images:
        verify_images(data, args.image_root)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # orient='records' matches pd.read_json(...) used by the training scripts.
    data.to_json(args.output, orient="records", indent=2)
    print(f"\nWrote skeleton JSON -> {args.output}")
    print(
        "Caption field (ivl_caption_unified) is empty and ready to backfill via "
        "Memotion/memotion_cap_gen.py."
    )


if __name__ == "__main__":
    main()
