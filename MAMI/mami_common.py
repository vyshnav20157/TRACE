"""Shared MAMI configuration and split helpers for the MAMI training scripts.

All three MAMI backbone scripts (`clip_xlm_roberta_mami.py`, `clip_vitL_14_mami.py`,
`siglip2_mami.py`) are copies of the corresponding FHM scripts with only dataset-specific
changes. This module centralizes the pieces that differ from FHM so those copies stay in
sync:

  * `REPO_ROOT` on sys.path so `from utils...` works when a script is launched as
    `python MAMI/clip_xlm_roberta_mami.py` (whose sys.path[0] would otherwise be MAMI/).
  * the MAMI image root and default dataset JSON path.
  * `split_mami_data()` which replaces FHM's dev_seen/dev_unseen/test_seen/test_unseen
    scheme with MAMI's plain train/val/test.
"""

import os
import sys

# Put the repo root on sys.path so the MAMI/ scripts can import the shared utils package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# MAMI dataset location. `img` values in the JSON are split-aware relative paths like
# "training_images/8716.jpg" / "test_images/15236.jpg", so f'{MAMI_IMAGE_ROOT}/{img}'
# resolves for every split (see MAMI/build_mami_skeleton.py).
MAMI_IMAGE_ROOT = "/backup/girish_datasets/MAMI"

# The dataset JSON is written by build_mami_skeleton.py and backfilled by mami_cap_gen.py.
# It defaults to a repo-local path (the MAMI/ dir) because the dataset directory itself is
# typically read-only. Override with --data-path if you keep it elsewhere.
MAMI_DATA_PATH = os.path.join(REPO_ROOT, "MAMI", "mami_captions_complete.json")


def split_mami_data(data):
    """Return (train_df, val_df, test_df) for the MAMI train/val/test splits."""
    train_data = data[data["split"] == "train"]
    val_data = data[data["split"] == "val"]
    test_data = data[data["split"] == "test"]
    return train_data, val_data, test_data
