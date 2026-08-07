"""MAMI evaluation metrics: the published protocol plus the TRACE-comparable one.

MAMI (SemEval-2022 Task 5, Fersini et al.) ranks Sub-task A -- the binary "misogynous"
label this repo models -- on

    macro-averaged F1 at a fixed 0.5 decision threshold,

averaging the F1 of the misogynous and non-misogynous classes. Accuracy and per-class P/R
are reported alongside in the task overview. TRACE's own FHM/MultiOFF evaluation instead
reports macro P/R/F1 at the F1-*optimal* threshold (picked from the precision-recall curve)
plus AUROC, and the original `MAMI/clip_*_mami.py` scripts inherited that unchanged.

The two disagree, and both are wanted: the MAMI numbers are what make results comparable to
published SemEval-2022 Task 5 baselines, and the TRACE numbers are what make them comparable
to the FHM/MultiOFF results elsewhere in this repo. So `compute_metrics()` returns both,
with the official ones as the headline:

    macro_f1        -- OFFICIAL headline: macro-F1 @ 0.5. This is the number MAMI tables
                       rank Sub-task A on, and what training selects on.
    accuracy        -- OFFICIAL: accuracy @ 0.5.
    macro_precision / macro_recall
                    -- OFFICIAL secondary: macro-averaged P/R @ 0.5.
    binary_f1       -- OFFICIAL secondary: F1 of the misogynous (positive) class @ 0.5.
    binary_precision / binary_recall
                    -- positive-class P/R @ 0.5.
    neg_f1 / pos_f1 -- per-class F1 @ 0.5 (pos_f1 == binary_f1; kept to show which class is
                       failing, since macro-F1 alone hides it).
    tuned_*         -- TRACE-style, computed at the F1-optimal threshold, plus `auc`.

Naming follows `MMSD/mmsd_metrics.py`: the plain keys are the OFFICIAL @0.5 values and the
tuned TRACE values are prefixed `tuned_`. `SELECTION_METRIC` names the key selection uses,
so the training loops read identically across datasets.

MAMI Sub-task A is balanced (5,500/5,500), so unlike Memotion a degenerate single-class
predictor scores ~0.33 macro-F1 rather than something deceptively high -- but `neg_f1`/
`pos_f1` are still reported so a collapsed model is visible at a glance.

All values are returned as preformatted 4-dp strings, matching what the previous MAMI
`evaluate_model` implementations returned so the surrounding logging code is unchanged.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The MAMI published protocol scores at a fixed 0.5 threshold.
OFFICIAL_THRESHOLD = 0.5

# Key that training scripts select/early-stop on. Kept here so every backbone agrees.
# SemEval-2022 Task 5 Sub-task A ranks on macro-F1, so that is what model selection follows
# (replacing the AUROC-driven selection the scripts previously inherited from FHM).
SELECTION_METRIC = "macro_f1"


def optimal_threshold(labels, probs):
    """F1-optimal threshold from the PR curve (TRACE's existing behaviour)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return OFFICIAL_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best = int(np.argmax(f1_scores))
    # precision_recall_curve returns len(thresholds) == len(precision) - 1.
    if best >= len(thresholds):
        return OFFICIAL_THRESHOLD
    return float(thresholds[best])


def compute_metrics(labels, probs, avg_loss=None):
    """Return the MAMI-official and TRACE-style metrics for the misogyny task.

    Args:
        labels: iterable of 0/1 ground-truth labels.
        probs:  iterable of predicted positive-class probabilities.
        avg_loss: optional average loss to include in the dict.

    Returns:
        (metrics, labels, official_preds) where `metrics` is a dict of preformatted 4-dp
        strings. `macro_f1` is the official headline metric.

        Note the tuple order -- (metrics, labels, preds) -- matches what the MAMI
        `evaluate_model` functions already returned, so their callers are unchanged.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    # ---- OFFICIAL MAMI metrics: macro-F1 + accuracy at 0.5 ----
    official_preds = (probs >= OFFICIAL_THRESHOLD).astype(int)
    accuracy = accuracy_score(labels, official_preds)
    macro_precision = precision_score(labels, official_preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, official_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, official_preds, average="macro", zero_division=0)
    binary_precision = precision_score(labels, official_preds, pos_label=1, zero_division=0)
    binary_recall = recall_score(labels, official_preds, pos_label=1, zero_division=0)
    binary_f1 = f1_score(labels, official_preds, pos_label=1, zero_division=0)
    per_class_f1 = f1_score(labels, official_preds, average=None, zero_division=0, labels=[0, 1])

    # ---- TRACE-style metrics at the F1-optimal threshold (FHM/MultiOFF-comparable) ----
    threshold = optimal_threshold(labels, probs)
    tuned_preds = (probs >= threshold).astype(int)

    # AUROC is threshold-free but undefined when only one class is present.
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")

    metrics = {
        # Official (headline) -- MAMI protocol at a fixed 0.5 threshold.
        "macro_f1": f"{macro_f1:.4f}",
        "accuracy": f"{accuracy:.4f}",
        "macro_precision": f"{macro_precision:.4f}",
        "macro_recall": f"{macro_recall:.4f}",
        "binary_f1": f"{binary_f1:.4f}",
        "binary_precision": f"{binary_precision:.4f}",
        "binary_recall": f"{binary_recall:.4f}",
        "neg_f1": f"{per_class_f1[0]:.4f}",
        "pos_f1": f"{per_class_f1[1]:.4f}",
        # TRACE-style (secondary) -- tuned threshold, plus threshold-free AUROC.
        "auc": f"{auc:.4f}",
        "tuned_accuracy": f"{accuracy_score(labels, tuned_preds):.4f}",
        "tuned_precision": f"{precision_score(labels, tuned_preds, average='macro', zero_division=0):.4f}",
        "tuned_recall": f"{recall_score(labels, tuned_preds, average='macro', zero_division=0):.4f}",
        "tuned_f1": f"{f1_score(labels, tuned_preds, average='macro', zero_division=0):.4f}",
        "threshold": f"{threshold:.4f}",
    }
    if avg_loss is not None:
        metrics["loss"] = f"{avg_loss:.4f}"

    return metrics, labels, official_preds


def format_metrics(metrics, prefix=""):
    """Human-readable two-line summary: official metrics first, then TRACE-style."""
    head = (
        f"{prefix}[official @0.5] macro_f1={metrics['macro_f1']} "
        f"acc={metrics['accuracy']} P={metrics['macro_precision']} R={metrics['macro_recall']} "
        f"| pos_f1={metrics['pos_f1']} neg_f1={metrics['neg_f1']}"
    )
    tail = (
        f"{prefix}[trace @{metrics['threshold']}] f1={metrics['tuned_f1']} "
        f"acc={metrics['tuned_accuracy']} P={metrics['tuned_precision']} "
        f"R={metrics['tuned_recall']} auc={metrics['auc']}"
    )
    if "loss" in metrics:
        tail += f" loss={metrics['loss']}"
    return head + "\n" + tail


def wandb_metrics(metrics, stage):
    """Flatten a metrics dict into wandb-loggable floats for a given stage label."""
    out = {}
    for key, value in metrics.items():
        try:
            out[f"{stage} {key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out
