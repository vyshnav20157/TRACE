"""
Build the MMSD2.0 "skeleton" JSON (and extract its images) for the captioning and training
scripts.

This mirrors `utils/build_fhm_skeleton.py`, `MAMI/build_mami_skeleton.py`, and
`Memotion/build_memotion_skeleton.py`, but for MMSD2.0 (Qin et al., Findings of ACL 2023) --
multimodal sarcasm detection over Twitter image+text pairs. The captioning script
(`MMSD/mmsd_cap_gen.py`) and the training scripts both read a single JSON (records orient)
with the columns:

    img, text, label, split, id, ivl_caption_task

Unlike MAMI/Memotion, MMSD2.0 is distributed as HuggingFace parquet shards with the images
stored as *bytes inside the table*, not as files on disk:

    mmsd-v2/train-0000{0..3}-of-00004.parquet
    mmsd-v2/validation-00000-of-00001.parquet
    mmsd-v2/test-00000-of-00001.parquet

    schema: image: struct<bytes: binary, path: string>, text: string, label: int64, id: string

The rest of the TRACE pipeline (RAM++ / GroundingDINO / InternVL captioning, and all three
backbones' `MemeDatasetJSON`) is built around opening an image file by path, so this builder
extracts every image ONCE to `MMSD_IMAGE_ROOT/<split>/<id>.jpg` and stores a split-aware
relative `img` path. After that MMSD looks exactly like every other dataset in the repo and
no downstream code needs a parquet code path.

Extraction is resumable and idempotent: an image already on disk is not rewritten (pass
`--overwrite-images` to force). ~2.6 GB of parquet expands to roughly the same volume of
JPEGs, so `--skip-images` is offered for rebuilding just the JSON.

Splits
------
MMSD2.0 ships official train / validation / test splits, so -- unlike Memotion -- nothing is
carved out of train. The parquet `validation` split is stored as `val` to match the split
vocabulary the rest of the repo uses.

Labels
------
A single binary column (1 = sarcastic, 0 = not sarcastic), written straight to the generic
`label` field that the shared `utils/` machinery reads. There is no task registry here: MMSD
is one task, so `apply_task_labels`-style projection is unnecessary.

The tweet `id` is carried through so predictions can be joined back to the original release.

Usage:
    python MMSD/build_mmsd_skeleton.py
    python MMSD/build_mmsd_skeleton.py --verify-images
    python MMSD/build_mmsd_skeleton.py --skip-images        # JSON only, images already out
    python MMSD/build_mmsd_skeleton.py --limit 20           # smoke test
"""

import argparse
import glob
import io
import os

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from mmsd_common import (
    MMSD_DATA_PATH,
    MMSD_IMAGE_ROOT,
    MMSD_PARQUET_DIR,
    MMSD_VERSION,
    PARQUET_SPLITS,
)

# Caption fields the downstream scripts read. Start empty; backfilled by the pipeline.
# Only InternVL is used (Gemini is not part of the MMSD flow, as with Memotion).
CAPTION_FIELDS = ["ivl_caption_task"]

# How many rows to pull out of a parquet shard at a time. The image bytes make rows large
# (~100 KB each), so batching keeps peak memory to a few hundred MB rather than loading a
# whole 500 MB shard at once.
BATCH_SIZE = 256


def shard_paths(parquet_dir, parquet_split):
    """Sorted parquet shards for one split, e.g. train-00000-of-00004.parquet."""
    pattern = os.path.join(parquet_dir, f"{parquet_split}-*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            f"No parquet shards matched {pattern}.\n"
            f"Check MMSD_PARQUET_DIR / MMSD_VERSION in MMSD/mmsd_common.py, or pass "
            f"--parquet-dir."
        )
    return paths


def extract_split(parquet_split, split_name, parquet_dir, image_root, write_images,
                  overwrite, limit=None):
    """Read one split's shards: write out its images and return its skeleton rows.

    Images go to `<image_root>/<split_name>/<id>.jpg`. The stored `img` is the split-aware
    relative path (`"train/12345.jpg"`) so a single root resolves every split downstream.
    """
    out_dir = os.path.join(image_root, split_name)
    if write_images:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    written = 0
    skipped = 0
    paths = shard_paths(parquet_dir, parquet_split)

    for path in paths:
        pf = pq.ParquetFile(path)
        desc = f"  {split_name:5s} {os.path.basename(path)}"
        total = pf.metadata.num_rows
        with tqdm(total=total, desc=desc, unit="row", leave=False) as bar:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE):
                cols = batch.to_pydict()
                for image, text, label, tweet_id in zip(
                    cols["image"], cols["text"], cols["label"], cols["id"]
                ):
                    if limit is not None and len(rows) >= limit:
                        bar.close()
                        return rows, written, skipped

                    # The release names every embedded file `<id>.jpg`; fall back to the id
                    # itself if a shard ever omits the path.
                    filename = os.path.basename(image.get("path") or f"{tweet_id}.jpg")
                    rel_path = f"{split_name}/{filename}"
                    abs_path = os.path.join(image_root, rel_path)

                    if write_images:
                        if overwrite or not os.path.exists(abs_path):
                            # Re-encode through PIL rather than dumping the raw bytes: it
                            # normalizes mode (a handful of frames are not RGB) and
                            # guarantees the file actually decodes, so a corrupt row fails
                            # here rather than 20 GPU-hours into captioning.
                            with Image.open(io.BytesIO(image["bytes"])) as img:
                                img.convert("RGB").save(abs_path, format="JPEG", quality=95)
                            written += 1
                        else:
                            skipped += 1

                    rows.append(
                        {
                            "img": rel_path,
                            "text": "" if text is None else str(text),
                            "label": int(label),
                            "split": split_name,
                            "id": str(tweet_id),
                        }
                    )
                    bar.update(1)

    return rows, written, skipped


def build_skeleton(parquet_dir, image_root, write_images, overwrite, limit=None):
    all_rows = []
    for parquet_split, split_name in PARQUET_SPLITS.items():
        rows, written, skipped = extract_split(
            parquet_split, split_name, parquet_dir, image_root, write_images, overwrite, limit
        )
        summary = f"  {split_name:5s}: {len(rows):6d} rows"
        if write_images:
            summary += f" | images written {written}, already present {skipped}"
        print(summary)
        all_rows.extend(rows)

    data = pd.DataFrame(all_rows)

    # Initialise empty caption fields for the pipeline to backfill.
    for col in CAPTION_FIELDS:
        data[col] = ""

    ordered = ["img", "text", "label", "split", "id"] + CAPTION_FIELDS
    extra = [c for c in data.columns if c not in ordered]
    return data[ordered + extra]


def report_split_balance(data):
    """Print per-split row counts and positive rate."""
    print("\nLabel balance by split:")
    print(f"  {'split':6s} {'rows':>7s} {'pos':>7s} {'neg':>7s} {'pos_rate':>9s}")
    for split in ["train", "val", "test"]:
        sub = data[data["split"] == split]
        if sub.empty:
            continue
        pos = int(sub["label"].sum())
        print(
            f"  {split:6s} {len(sub):7d} {pos:7d} {len(sub) - pos:7d} "
            f"{sub['label'].mean():9.3f}"
        )


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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet-dir", default=MMSD_PARQUET_DIR,
                        help=f"Directory holding the {MMSD_VERSION} parquet shards.")
    parser.add_argument("--image-root", default=MMSD_IMAGE_ROOT,
                        help="Root the images are extracted to and `img` paths resolve against.")
    parser.add_argument("--output", default=MMSD_DATA_PATH,
                        help="Path to write the skeleton JSON (records orient).")
    parser.add_argument("--skip-images", action="store_true",
                        help="Do not extract images; rebuild the JSON only.")
    parser.add_argument("--overwrite-images", action="store_true",
                        help="Re-extract images that are already on disk (default: skip them).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only take the first N rows per split (smoke test).")
    parser.add_argument("--verify-images", action="store_true",
                        help="Check that every img path resolves before writing.")
    args = parser.parse_args()

    print(f"Reading MMSD parquet shards from: {args.parquet_dir}")
    if args.skip_images:
        print("Image extraction disabled (--skip-images).")
    else:
        print(f"Extracting images to: {args.image_root}")

    data = build_skeleton(
        args.parquet_dir,
        args.image_root,
        write_images=not args.skip_images,
        overwrite=args.overwrite_images,
        limit=args.limit,
    )

    print(f"\nTotal rows: {len(data)}")
    print("\nSplit distribution:")
    print(data["split"].value_counts().reindex(["train", "val", "test"]).to_string())

    report_split_balance(data)

    # The tweet id is the join key back to the original release; duplicates would mean two
    # rows share an image path and silently overwrite each other on extraction.
    dupes = int(data["img"].duplicated().sum())
    if dupes:
        print(f"\nWARNING: {dupes} duplicate `img` path(s) -- extracted images may collide.")

    empty_text = int((data["text"].astype(str).str.strip() == "").sum())
    if empty_text:
        print(f"\nNote: {empty_text} row(s) have empty text; the dataset classes fall back "
              f"to a 'No caption' placeholder for those.")

    if args.verify_images:
        verify_images(data, args.image_root)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # orient='records' matches pd.read_json(...) used by the training scripts.
    data.to_json(args.output, orient="records", indent=2)
    print(f"\nWrote skeleton JSON -> {args.output}")
    print(
        "Caption field (ivl_caption_task) is empty and ready to backfill via "
        "MMSD/mmsd_cap_gen.py."
    )


if __name__ == "__main__":
    main()
