"""Shared Memotion configuration, task registry, and split helpers.

All three Memotion backbone scripts (`clip_xlm_roberta_memotion.py`,
`clip_vitL_14_memotion.py`, `siglip2_memotion.py`) are adaptations of the corresponding
FHM/MAMI scripts with only dataset-specific changes. This module centralizes the pieces
that differ so those copies stay in sync:

  * `REPO_ROOT` on sys.path so `from utils...` works when a script is launched as
    `python Memotion/clip_xlm_roberta_memotion.py` (whose sys.path[0] would otherwise be
    Memotion/).
  * the Memotion image roots and default dataset-JSON path.
  * the TASKS registry: Memotion 1.0 Task B is three independent binary problems
    (humour / sarcasm / offensive), and every script selects exactly one via `--task`.
  * `split_memotion_data()` which yields the train/val/test frames.

Memotion vs. MAMI: the one structural difference is that a Memotion row carries THREE
labels, not one. Rather than three separate dataset JSONs, the skeleton stores all of them
(`humour_label`, `sarcasm_label`, `offensive_label`) in a single JSON and the training
scripts project the chosen task onto the generic `label` column that the shared
`utils/` machinery expects. Captions are therefore generated once and reused by all three
tasks.
"""

import os
import sys

# Put the repo root on sys.path so the Memotion/ scripts can import the shared utils package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Memotion 1.0 dataset location. The dataset stays where it is; we only point at it.
MEMOTION_DATASET_DIR = "/home/vyshnav/MHA-MEME/dataset"

# `img` values in the JSON are split-aware relative paths like
# "train_meme_images/foo.jpg" / "test_meme_images/bar.png", so f'{IMAGE_ROOT}/{img}'
# resolves for every split (see Memotion/build_memotion_skeleton.py).
MEMOTION_IMAGE_ROOT = MEMOTION_DATASET_DIR

# Source CSVs (already binarized by the dataset's own eda_binarize.py).
MEMOTION_TRAIN_CSV = os.path.join(MEMOTION_DATASET_DIR, "train_binary.csv")
MEMOTION_TEST_CSV = os.path.join(MEMOTION_DATASET_DIR, "test_binary.csv")

# The dataset JSON is written by build_memotion_skeleton.py and backfilled by
# memotion_cap_gen.py. It defaults to a repo-local path (the Memotion/ dir) because the
# dataset directory belongs to another project. Override with --data-path.
MEMOTION_DATA_PATH = os.path.join(REPO_ROOT, "Memotion", "memotion_captions_complete.json")

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
# Memotion 1.0 Task B: three independent binary classifications over the same memes.
# Each entry maps the task name to:
#   csv_column   -- the binarized column in train_binary.csv / test_binary.csv
#   json_field   -- where the skeleton stores it (all three live in one JSON)
#   positive     -- human-readable name of the positive class (for logging/reports)
#   negative     -- human-readable name of the negative class
TASKS = {
    "humour": {
        "csv_column": "humour_bin",
        "json_field": "humour_label",
        "positive": "humorous",
        "negative": "not_funny",
    },
    "offensive": {
        "csv_column": "offensive_bin",
        "json_field": "offensive_label",
        "positive": "offensive",
        "negative": "not_offensive",
    },
    "sarcasm": {
        "csv_column": "sarcasm_bin",
        "json_field": "sarcasm_label",
        "positive": "sarcastic",
        "negative": "not_sarcastic",
    },
}

# Order the tasks are worked through (humour first, then offensive, then sarcasm).
TASK_ORDER = ["humour", "offensive", "sarcasm"]

# Fraction of train held out as a validation split, and the seed that fixes it.
# Memotion ships only train/test, so early stopping and caption selection need a val
# split carved out of train -- the test split is never used for model selection.
VAL_FRACTION = 0.1
VAL_SPLIT_SEED = 42


def task_label_field(task):
    """Return the JSON field holding the binary label for `task`."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose one of: {', '.join(TASK_ORDER)}")
    return TASKS[task]["json_field"]


def apply_task_labels(data, task):
    """Project the chosen task's label onto the generic `label` column.

    The shared TRACE machinery (`utils/loss_functions.py`, `utils/caption_selection.py`)
    and every dataset class read a single `label` field. Memotion rows carry three, so we
    copy the selected task's label into `label` and hand the frame downstream unchanged --
    which is what keeps the model/loss code identical to FHM/MAMI.
    """
    field = task_label_field(task)
    if field not in data.columns:
        raise SystemExit(
            f"Dataset JSON has no '{field}' column (task '{task}'). "
            f"Rebuild it with: python Memotion/build_memotion_skeleton.py"
        )
    data = data.copy()
    data["label"] = data[field].astype(int)
    return data


def split_memotion_data(data):
    """Return (train_df, val_df, test_df) for the Memotion train/val/test splits."""
    train_data = data[data["split"] == "train"]
    val_data = data[data["split"] == "val"]
    test_data = data[data["split"] == "test"]
    return train_data, val_data, test_data


def describe_task(task):
    """One-line description of a task for log headers."""
    cfg = TASKS[task]
    return f"{task} (1 = {cfg['positive']}, 0 = {cfg['negative']})"
