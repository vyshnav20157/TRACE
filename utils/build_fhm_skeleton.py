"""
Build the FHM "skeleton" JSON that the captioning and training scripts expect.

The active captioning entrypoint (`vg_caption_gen.py::process_json_with_missing_captions`)
does NOT create a dataset from raw images -- it reads a JSON that already contains the
`img / text / label / split` rows and only *backfills* the missing caption fields.
Likewise the training scripts (`clip_vitL_14_ft.py`, `clip_xlm_roberta_ft.py`,
`siglip2_ft.py`) read this same JSON and consume the columns:

    img, text, label, split, ivl_8b_new_caption, gemini_caption

This script converts the raw Hateful Memes annotation `.jsonl` files (as distributed by
the Hateful Memes Challenge) into that skeleton JSON, with empty caption fields ready to
be populated by the captioning pipeline (InternVL -> ivl_8b_new_caption, Gemini ->
gemini_caption).

Raw FHM ships these annotation files:
    train.jsonl, dev_seen.jsonl, dev_unseen.jsonl, test_seen.jsonl, test_unseen.jsonl
each line: {"id": ..., "img": "img/12345.png", "label": 0|1, "text": "..."}
(test_*.jsonl may omit "label" in some releases -- handled below.)

Usage:
    python utils/build_fhm_skeleton.py \
        --annotations-dir /path/to/Hateful_Memes_Extended \
        --output /path/to/Hateful_Memes_Extended/ivl_plus_gemini_captions_complete.json

After running, point the captioning script's `json_path` and the training scripts'
`data_path` at the produced file.
"""

import argparse
import json
import os

import pandas as pd

# split name -> annotation filename. These are the split labels the training scripts
# filter on (see clip_vitL_14_ft.py:430-434), so the `split` column must use exactly
# these strings.
SPLIT_FILES = {
    "train": "train.jsonl",
    "dev_seen": "dev_seen.jsonl",
    "dev_unseen": "dev_unseen.jsonl",
    "test_seen": "test_seen.jsonl",
    "test_unseen": "test_unseen.jsonl",
}

# Columns the downstream scripts read. Caption fields start empty and are backfilled by
# the captioning pipeline.
CAPTION_FIELDS = ["ivl_8b_new_caption", "gemini_caption"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_skeleton(annotations_dir):
    frames = []
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(annotations_dir, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found -- skipping split '{split}'.")
            continue
        rows = load_jsonl(path)
        df = pd.DataFrame(rows)
        df["split"] = split
        print(f"  {split:12s}: {len(df):6d} rows from {filename}")
        frames.append(df)

    if not frames:
        raise SystemExit(
            f"No annotation files found in {annotations_dir}. "
            f"Expected one of: {', '.join(SPLIT_FILES.values())}"
        )

    data = pd.concat(frames, ignore_index=True)

    # `img` in the raw FHM jsonl is like "img/12345.png". The dataset loaders build the
    # full path as f'{image_root}/{img}', so we keep `img` exactly as distributed.
    if "img" not in data.columns:
        raise SystemExit("Annotation files have no 'img' column -- unexpected FHM format.")

    # `text` must exist; default to empty string if a row is missing it.
    if "text" not in data.columns:
        data["text"] = ""
    data["text"] = data["text"].fillna("").astype(str)

    # `label` is absent in some public test_* releases. The training scripts cast it to a
    # float tensor, so fill unknown labels with -1 (clearly out-of-range; never treat -1
    # rows as ground truth). dev/train always have labels.
    if "label" not in data.columns:
        data["label"] = -1
    missing_label = data["label"].isna()
    if missing_label.any():
        print(
            f"  NOTE: {int(missing_label.sum())} rows have no label "
            f"(typical for held-out test splits); filled with -1."
        )
    data["label"] = data["label"].fillna(-1).astype(int)

    # Initialise empty caption fields for the pipeline to backfill.
    for col in CAPTION_FIELDS:
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)

    # Keep a stable, explicit column order.
    ordered = ["img", "text", "label", "split"] + CAPTION_FIELDS
    extra = [c for c in data.columns if c not in ordered]
    data = data[ordered + extra]

    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="Directory containing the raw FHM *.jsonl annotation files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the skeleton JSON (records orient). "
        "Point the captioning + training scripts at this file.",
    )
    args = parser.parse_args()

    print(f"Reading FHM annotations from: {args.annotations_dir}")
    data = build_skeleton(args.annotations_dir)

    print(f"\nTotal rows: {len(data)}")
    print(f"Label distribution (==-1 means unlabeled test split):")
    print(data["label"].value_counts().to_string())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # orient='records' matches pd.read_json(...) used by the training scripts.
    data.to_json(args.output, orient="records", indent=2)
    print(f"\nWrote skeleton JSON -> {args.output}")
    print("Caption fields (ivl_8b_new_caption, gemini_caption) are empty and ready to backfill.")


if __name__ == "__main__":
    main()
