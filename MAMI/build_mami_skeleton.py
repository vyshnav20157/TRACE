"""
Build the MAMI "skeleton" JSON that the captioning and training scripts expect.

This mirrors `utils/build_fhm_skeleton.py`, but for the MAMI (Multimedia Automatic
Misogyny Identification) dataset. The captioning script (`MAMI/mami_cap_gen.py`) and the
training script (`MAMI/train_mami.py`) both read a single JSON (records orient) with the
columns:

    img, text, label, split, ivl_8b_new_caption, gemini_caption

exactly like the FHM flow, so the shared model/dataset/loss code is reused unchanged.
Caption fields start empty and are backfilled by the caption pipeline (InternVL ->
ivl_8b_new_caption on the GPU server, Gemini -> gemini_caption on a local machine).

MAMI ships three TSV files with columns:
    file_name, label, shaming, stereotype, objectification, violence, text
(NOTE: test.tsv's first column header is the typo `mifile_name`; normalized below.)

Images live in two folders keyed by split:
    training_images/  -> train.tsv + validation.tsv filenames
    test_images/      -> test.tsv filenames

We store `img` as a split-aware relative path (e.g. "training_images/8716.jpg") so the
downstream `f'{MAMI_ROOT}/{img}'` construction resolves with a single MAMI root, exactly
as FHM's `img/12345.png` does.

Only the binary "misogynous" `label` is modeled (Task A). The four sub-category columns
(shaming, stereotype, objectification, violence) are preserved in the JSON for reference
but are NOT used by training.

Usage:
    python MAMI/build_mami_skeleton.py \
        --mami-dir /backup/girish_datasets/MAMI \
        --output /backup/girish_datasets/MAMI/mami_captions_complete.json

After running, point the captioning script's `json_path` and the training script's
`data_path` at the produced file.
"""

import argparse
import os

import pandas as pd

from mami_common import MAMI_DATA_PATH

# split name -> (tsv filename, image subfolder). The `split` strings must match exactly
# what train_mami.py filters on.
SPLIT_FILES = {
    "train": ("train.tsv", "training_images"),
    "val": ("validation.tsv", "training_images"),
    "test": ("test.tsv", "test_images"),
}

# Caption fields the downstream scripts read. Start empty; backfilled by the pipeline.
CAPTION_FIELDS = ["ivl_8b_new_caption", "gemini_caption"]

# MAMI sub-category columns -- preserved but not modeled (Task B is out of scope).
SUBLABEL_FIELDS = ["shaming", "stereotype", "objectification", "violence"]


def load_split(mami_dir, split, filename, subfolder):
    path = os.path.join(mami_dir, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- skipping split '{split}'.")
        return None

    df = pd.read_csv(path, sep="\t")

    # test.tsv ships the first column as `mifile_name` (a typo); normalize it.
    if "mifile_name" in df.columns:
        df = df.rename(columns={"mifile_name": "file_name"})

    required = {"file_name", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{filename} is missing expected column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    # Store a split-aware relative image path so `f'{root}/{img}'` resolves everywhere.
    out["img"] = df["file_name"].apply(lambda f: f"{subfolder}/{f}")
    out["text"] = df["text"].fillna("").astype(str)
    out["label"] = df["label"].astype(int)
    out["split"] = split

    # Carry sub-category labels through when present (for future Task B use).
    for col in SUBLABEL_FIELDS:
        if col in df.columns:
            out[col] = df[col].astype(int)

    print(f"  {split:5s}: {len(out):5d} rows from {filename} (images in {subfolder}/)")
    return out


def build_skeleton(mami_dir):
    frames = []
    for split, (filename, subfolder) in SPLIT_FILES.items():
        df = load_split(mami_dir, split, filename, subfolder)
        if df is not None:
            frames.append(df)

    if not frames:
        raise SystemExit(
            f"No MAMI TSV files found in {mami_dir}. "
            f"Expected: {', '.join(fn for fn, _ in SPLIT_FILES.values())}"
        )

    data = pd.concat(frames, ignore_index=True)

    # Initialise empty caption fields for the pipeline to backfill.
    for col in CAPTION_FIELDS:
        data[col] = ""

    # Stable, explicit column order: core FHM contract first, then sub-labels.
    ordered = ["img", "text", "label", "split"] + CAPTION_FIELDS
    extra = [c for c in data.columns if c not in ordered]
    data = data[ordered + extra]

    return data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mami-dir",
        default="/backup/girish_datasets/MAMI",
        help="Directory containing the MAMI TSV files and image folders.",
    )
    parser.add_argument(
        "--output",
        default=MAMI_DATA_PATH,
        help="Path to write the skeleton JSON (records orient). "
        "Point the captioning + training scripts at this file.",
    )
    args = parser.parse_args()

    print(f"Reading MAMI TSVs from: {args.mami_dir}")
    data = build_skeleton(args.mami_dir)

    print(f"\nTotal rows: {len(data)}")
    print("Label distribution (0 = not misogynous, 1 = misogynous):")
    print(data["label"].value_counts().to_string())
    print("\nSplit distribution:")
    print(data["split"].value_counts().to_string())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # orient='records' matches pd.read_json(...) used by the training script.
    data.to_json(args.output, orient="records", indent=2)
    print(f"\nWrote skeleton JSON -> {args.output}")
    print(
        "Caption fields (ivl_8b_new_caption, gemini_caption) are empty and ready to "
        "backfill via MAMI/mami_cap_gen.py."
    )


if __name__ == "__main__":
    main()
