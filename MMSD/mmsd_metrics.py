"""MMSD2.0 evaluation metrics: the published protocol plus the TRACE-comparable one.

MMSD2.0 (Qin et al., Findings of ACL 2023) and the MMSD literature it builds on report, at
a fixed 0.5 decision threshold:

    Accuracy, and Precision / Recall / F1 for the POSITIVE (sarcastic) class,

with macro-averaged P/R/F1 usually given alongside. TRACE's own FHM/MultiOFF/MAMI
evaluation instead reports macro P/R/F1 at the F1-*optimal* threshold (picked from the
precision-recall curve) plus AUROC.

The two disagree, and both are wanted: the MMSD numbers are what make results comparable to
published MMSD2.0 baselines, and the TRACE numbers are what make them comparable to the
FHM/MultiOFF/MAMI/Memotion results elsewhere in this repo. So `compute_metrics()` returns
both, with the official ones as the headline:

    accuracy        -- OFFICIAL: accuracy @ 0.5. Model selection uses binary_f1 below.
    binary_f1       -- OFFICIAL headline: F1 of the sarcastic class @ 0.5. This is the
                       number MMSD2.0 tables rank on, and what training selects on.
    binary_precision / binary_recall
                    -- OFFICIAL: positive-class P/R @ 0.5.
    macro_f1, macro_precision, macro_recall
                    -- OFFICIAL secondary: macro-averaged @ 0.5.
    neg_f1 / pos_f1 -- per-class F1 @ 0.5 (pos_f1 == binary_f1; kept for symmetry with the
                       Memotion reports and to show which class is failing).
    tuned_*         -- TRACE-style, computed at the F1-optimal threshold, plus `auc`.

Note the naming difference from `Memotion/memotion_metrics.py`: there the headline was
macro-F1 and the plain keys (`accuracy`, `f1`, ...) were the tuned ones. Here the plain keys
are the OFFICIAL @0.5 values, because that is the MMSD convention, and the tuned TRACE
values are prefixed `tuned_`. `SELECTION_METRIC` names the key selection uses either way, so
the training loops read identically.

All values are returned as preformatted 4-dp strings, matching what the existing FHM/MAMI
`evaluate_model` implementations return so the surrounding logging code is unchanged.
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

# The MMSD published protocol scores at a fixed 0.5 threshold.
OFFICIAL_THRESHOLD = 0.5

# Key that training scripts select/early-stop on. Kept here so every backbone agrees.
# MMSD2.0 tables rank on the sarcastic-class F1, so that is what model selection follows.
SELECTION_METRIC = "binary_f1"


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
    """Return the MMSD-official and TRACE-style metrics for the sarcasm task.

    Args:
        labels: iterable of 0/1 ground-truth labels.
        probs:  iterable of predicted positive-class probabilities.
        avg_loss: optional average loss to include in the dict.

    Returns:
        (metrics, official_preds, labels) where `metrics` is a dict of preformatted 4-dp
        strings. `binary_f1` is the official headline metric.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    # ---- OFFICIAL MMSD metrics: accuracy + positive-class P/R/F1 at 0.5 ----
    official_preds = (probs >= OFFICIAL_THRESHOLD).astype(int)
    accuracy = accuracy_score(labels, official_preds)
    binary_precision = precision_score(labels, official_preds, pos_label=1, zero_division=0)
    binary_recall = recall_score(labels, official_preds, pos_label=1, zero_division=0)
    binary_f1 = f1_score(labels, official_preds, pos_label=1, zero_division=0)
    macro_precision = precision_score(labels, official_preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, official_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, official_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, official_preds, average=None, zero_division=0, labels=[0, 1])

    # ---- TRACE-style metrics at the F1-optimal threshold (FHM/MAMI-comparable) ----
    threshold = optimal_threshold(labels, probs)
    tuned_preds = (probs >= threshold).astype(int)

    # AUROC is threshold-free but undefined when only one class is present.
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")

    metrics = {
        # Official (headline) -- MMSD2.0 protocol at a fixed 0.5 threshold.
        "accuracy": f"{accuracy:.4f}",
        "binary_f1": f"{binary_f1:.4f}",
        "binary_precision": f"{binary_precision:.4f}",
        "binary_recall": f"{binary_recall:.4f}",
        "macro_f1": f"{macro_f1:.4f}",
        "macro_precision": f"{macro_precision:.4f}",
        "macro_recall": f"{macro_recall:.4f}",
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

    return metrics, official_preds, labels


def format_metrics(metrics, prefix=""):
    """Human-readable two-line summary: official metrics first, then TRACE-style."""
    head = (
        f"{prefix}[official @0.5] acc={metrics['accuracy']} "
        f"F1={metrics['binary_f1']} P={metrics['binary_precision']} R={metrics['binary_recall']} "
        f"| macro_f1={metrics['macro_f1']} "
        f"(neg_f1={metrics['neg_f1']} pos_f1={metrics['pos_f1']})"
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
