"""Memotion evaluation metrics: the official protocol plus the TRACE-comparable one.

Memotion 1.0 (SemEval-2020 Task 8) scores Task B by **macro-averaged F1 at a fixed 0.5
decision threshold** -- the metric the shared task leaderboard ranks on. TRACE's own
FHM/MultiOFF/MAMI evaluation instead reports macro P/R/F1 at the F1-*optimal* threshold
(picked from the precision-recall curve) alongside AUROC.

The two disagree, and both are wanted: the Memotion number is what makes results
comparable to published Memotion baselines, and the TRACE number is what makes them
comparable to the FHM/MultiOFF/MAMI results elsewhere in this repo. So
`compute_metrics()` returns both, with the official one as the headline:

    macro_f1        -- OFFICIAL Memotion metric: macro-F1 @ 0.5. Model selection uses this.
    accuracy, precision, recall, f1, auc
                    -- TRACE-style, computed at the F1-optimal threshold.
    threshold       -- the tuned threshold those TRACE-style numbers used.
    pos_f1 / neg_f1 -- per-class F1 @ 0.5, since the macro average hides which class a
                       model is failing on (Memotion's tasks are 61-78% positive, so a
                       degenerate all-positive predictor still scores well on accuracy).

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

# The Memotion official protocol scores at a fixed 0.5 threshold.
OFFICIAL_THRESHOLD = 0.5

# Key that training scripts select/early-stop on. Kept here so every backbone agrees.
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
    """Return the Memotion-official and TRACE-style metrics for one binary task.

    Args:
        labels: iterable of 0/1 ground-truth labels.
        probs:  iterable of predicted positive-class probabilities.
        avg_loss: optional average loss to include in the dict.

    Returns:
        dict of preformatted 4-dp strings. `macro_f1` is the official headline metric.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    # ---- OFFICIAL Memotion metric: macro-F1 at a fixed 0.5 threshold ----
    official_preds = (probs >= OFFICIAL_THRESHOLD).astype(int)
    macro_f1 = f1_score(labels, official_preds, average="macro", zero_division=0)
    official_acc = accuracy_score(labels, official_preds)
    official_precision = precision_score(labels, official_preds, average="macro", zero_division=0)
    official_recall = recall_score(labels, official_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, official_preds, average=None, zero_division=0, labels=[0, 1])

    # ---- TRACE-style metrics at the F1-optimal threshold (FHM/MAMI-comparable) ----
    threshold = optimal_threshold(labels, probs)
    tuned_preds = (probs >= threshold).astype(int)

    # AUROC is threshold-free but undefined when only one class is present.
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")

    metrics = {
        # Official (headline) -- macro-F1 @ 0.5.
        "macro_f1": f"{macro_f1:.4f}",
        "official_accuracy": f"{official_acc:.4f}",
        "official_precision": f"{official_precision:.4f}",
        "official_recall": f"{official_recall:.4f}",
        "neg_f1": f"{per_class_f1[0]:.4f}",
        "pos_f1": f"{per_class_f1[1]:.4f}",
        # TRACE-style (secondary) -- tuned threshold.
        "accuracy": f"{accuracy_score(labels, tuned_preds):.4f}",
        "precision": f"{precision_score(labels, tuned_preds, average='macro', zero_division=0):.4f}",
        "recall": f"{recall_score(labels, tuned_preds, average='macro', zero_division=0):.4f}",
        "f1": f"{f1_score(labels, tuned_preds, average='macro', zero_division=0):.4f}",
        "auc": f"{auc:.4f}",
        "threshold": f"{threshold:.4f}",
    }
    if avg_loss is not None:
        metrics["loss"] = f"{avg_loss:.4f}"

    return metrics, official_preds, labels


def format_metrics(metrics, prefix=""):
    """Human-readable two-line summary: official metrics first, then TRACE-style."""
    head = (
        f"{prefix}[official @0.5] macro_f1={metrics['macro_f1']} "
        f"acc={metrics['official_accuracy']} "
        f"P={metrics['official_precision']} R={metrics['official_recall']} "
        f"(neg_f1={metrics['neg_f1']} pos_f1={metrics['pos_f1']})"
    )
    tail = (
        f"{prefix}[trace @{metrics['threshold']}] f1={metrics['f1']} "
        f"acc={metrics['accuracy']} P={metrics['precision']} R={metrics['recall']} "
        f"auc={metrics['auc']}"
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
