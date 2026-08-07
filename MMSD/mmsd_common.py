"""Shared MMSD2.0 configuration and split helpers.

All three MMSD backbone scripts (`clip_xlm_roberta_mmsd.py`, `clip_vitL_14_mmsd.py`,
`siglip2_mmsd.py`) are adaptations of the corresponding FHM/MAMI/Memotion scripts with only
dataset-specific changes. This module centralizes the pieces that differ so those copies
stay in sync:

  * `REPO_ROOT` on sys.path so `from utils...` works when a script is launched as
    `python MMSD/clip_xlm_roberta_mmsd.py` (whose sys.path[0] would otherwise be MMSD/).
  * the MMSD image root and default dataset-JSON path.
  * `split_mmsd_data()` which yields the train/val/test frames.

MMSD vs. Memotion: MMSD2.0 is a *single* binary task (sarcastic / not sarcastic), so there
is no task registry and no `apply_task_labels` projection -- the skeleton writes the
generic `label` column directly and the shared `utils/` machinery reads it unchanged. MMSD
also ships its own official train/validation/test splits, so unlike Memotion nothing has to
be carved out of train.

MMSD vs. MAMI: the images are not files on disk in the distribution -- the HuggingFace
release stores them as bytes inside the parquet shards. `build_mmsd_skeleton.py` extracts
them once to `MMSD_IMAGE_ROOT` so the rest of the pipeline (captioning, the three
backbones) can treat MMSD exactly like every other dataset: a JSON with a relative `img`
path resolved against one image root.
"""

import os
import sys

# Put the repo root on sys.path so the MMSD/ scripts can import the shared utils package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# MMSD2.0 dataset location (the HuggingFace parquet release). We only read from here.
MMSD_DATASET_DIR = "/home/vyshnav/MMSD2.0"

# Which config of the release to build from. The directory under MMSD_DATASET_DIR holding
# the `train-*/validation-*/test-*` parquet shards. MMSD2.0 (the de-biased relabelling from
# Qin et al. 2023) is the one this work benchmarks on; `mmsd-v1` / `mmsd-original` /
# `mmsd-clean` are the other configs shipped in the same release.
MMSD_VERSION = "mmsd-v2"
MMSD_PARQUET_DIR = os.path.join(MMSD_DATASET_DIR, MMSD_VERSION)

# Images are extracted out of the parquet shards to here by build_mmsd_skeleton.py, as
# `<split>/<id>.jpg`. `img` values in the JSON are split-aware relative paths (e.g.
# "train/682716753374351360.jpg") so a single root resolves every split, matching the
# MAMI/Memotion convention.
MMSD_IMAGE_ROOT = os.path.join(MMSD_DATASET_DIR, "images")

# The dataset JSON is written by build_mmsd_skeleton.py and backfilled by mmsd_cap_gen.py.
# It defaults to a repo-local path (the MMSD/ dir) because the dataset directory belongs to
# the upstream release. Override with --data-path.
MMSD_DATA_PATH = os.path.join(REPO_ROOT, "MMSD", "mmsd_captions_complete.json")

# Split names as they appear in the parquet filenames, mapped to the split label stored in
# the JSON. MMSD ships all three, so (unlike Memotion) none has to be carved out of train.
PARQUET_SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

# Human-readable class names, for log headers and report tables.
POSITIVE_CLASS = "sarcastic"
NEGATIVE_CLASS = "not_sarcastic"


def split_mmsd_data(data):
    """Return (train_df, val_df, test_df) for the MMSD official train/val/test splits."""
    train_data = data[data["split"] == "train"]
    val_data = data[data["split"] == "val"]
    test_data = data[data["split"] == "test"]
    return train_data, val_data, test_data


def describe_task():
    """One-line description of the MMSD task for log headers."""
    return f"sarcasm (1 = {POSITIVE_CLASS}, 0 = {NEGATIVE_CLASS})"
