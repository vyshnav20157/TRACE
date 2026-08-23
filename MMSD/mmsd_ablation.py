"""Run the MMSD2.0 modality ablation sweep and collect the results table.

This is the driver for the six-arm study defined in `mmsd_modality.py`: it trains each arm
in turn (or just a chosen subset), then gathers every arm's test metrics into one table so
the modality comparison can be read off directly instead of being reassembled by hand from
six scattered prediction files.

Each arm is launched as a SEPARATE subprocess rather than by importing the backbone and
looping in-process. That is deliberate: the backbone modules load a multi-GB vision-language
model at import time and hold global CUDA state, and training mutates module-level globals
(the `dataset` global in the roberta script, `calculate_loss_gs.call_count`, the RNG streams
seeded at import). Running arms in one process would leak all of that from one arm into the
next and quietly break the "everything held fixed except the modality" guarantee that makes
the ablation meaningful. A subprocess per arm gets each one a clean interpreter, identical
seeding, and a fresh GPU allocator.

Usage
-----
    # Train every arm sequentially on the default backbone, then print the table:
    python MMSD/mmsd_ablation.py --run

    # Only some arms (e.g. resuming a sweep that was interrupted):
    python MMSD/mmsd_ablation.py --run --arms image_only image_text

    # Just collect and print the table from finished runs (no training):
    python MMSD/mmsd_ablation.py

    # Markdown table, e.g. to paste into the thesis:
    python MMSD/mmsd_ablation.py --format markdown

    # Smoke-test the whole sweep end to end in a few minutes:
    python MMSD/mmsd_ablation.py --run --subset 48 --epochs 1

Arms are run one at a time because each needs the whole GPU. To parallelise across two GPUs,
launch two `train_mmsd.py --arm ...` commands yourself with different `CUDA_VISIBLE_DEVICES`
(see ToDo.md), then run this script with no `--run` to collect the table.
"""

import argparse
import json
import os
import subprocess
import sys

from mmsd_modality import ARMS, DEFAULT_ARM, get_arm, preds_name

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "MMSD", "train_mmsd.py")

# Columns pulled into the summary table, in order. These are MMSD's official headline
# metrics (accuracy and sarcastic-class F1 at 0.5) plus macro-F1 and AUROC; see
# mmsd_metrics.py for why both the official and TRACE-style numbers exist.
TABLE_COLUMNS = [
    ("accuracy", "Acc"),
    ("binary_f1", "F1(sarc)"),
    ("binary_precision", "P(sarc)"),
    ("binary_recall", "R(sarc)"),
    ("macro_f1", "Macro-F1"),
    ("auc", "AUROC"),
]

# Order arms appear in the table: ablations first, ascending in how much they are given,
# with full TRACE last as the reference the others are read against.
ARM_ORDER = ["image_only", "image_text", "image_taskcap", "image_genericcap",
             "image_unifiedcap", "trace"]


def train_arm(arm, backbone, extra_args):
    """Train a single arm in its own subprocess. Returns True on success."""
    cmd = [sys.executable, TRAIN_SCRIPT, "--backbone", backbone, "--arm", arm] + extra_args
    print(f"\n{'=' * 78}\n[ablation] training arm '{arm}' ({backbone})\n[ablation] $ {' '.join(cmd)}\n{'=' * 78}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[ablation] arm '{arm}' FAILED (exit {result.returncode})")
        return False
    return True


def load_results(backbone, arms):
    """Load each arm's test-prediction JSON, skipping arms that have not been run."""
    results = {}
    for arm in arms:
        path = os.path.join(REPO_ROOT, preds_name(backbone, arm))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            results[arm] = json.load(f)
    return results


def format_table(results, backbone, fmt="text"):
    """Render the collected per-arm metrics as an aligned text or markdown table."""
    arms = [a for a in ARM_ORDER if a in results]
    if not arms:
        return (
            "No results found. Train the arms first:\n"
            "    python MMSD/mmsd_ablation.py --run"
        )

    headers = ["Arm", "Text sources"] + [label for _, label in TABLE_COLUMNS]
    rows = []
    for arm in arms:
        record = results[arm]
        metrics = record.get("metrics", {})
        sources = record.get("text_sources") or get_arm(arm)["sources"]
        rendered = "+".join("<const>" if s == "null" else ("text" if s == "text" else s) for s in sources)
        row = [arm, rendered]
        for key, _ in TABLE_COLUMNS:
            value = metrics.get(key, "-")
            row.append(str(value))
        rows.append(row)

    if fmt == "markdown":
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(row) + " |" for row in rows]
        header = f"\nMMSD2.0 modality ablation -- backbone: {backbone}\n"
        return header + "\n".join(out)

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        f"\nMMSD2.0 modality ablation -- backbone: {backbone}",
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    lines += ["  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows]

    missing = [a for a in ARM_ORDER if a not in results]
    if missing:
        lines.append(f"\nNot yet run: {', '.join(missing)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", action="store_true",
                        help="Train the arms. Without this the script only collects and "
                             "prints the table from prediction files already on disk.")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=ARM_ORDER,
                        help="Which arms to run/collect (default: all six).")
    parser.add_argument("--backbone", default="roberta",
                        choices=["roberta", "vitl14", "siglip2"],
                        help="Backbone to run the sweep on (default: roberta, the primary).")
    parser.add_argument("--format", dest="fmt", default="text", choices=["text", "markdown"],
                        help="Table format (default: text).")
    parser.add_argument("--out", default=None,
                        help="Also write the table to this file.")
    # Everything else (--subset, --epochs, --no-resume, --wandb, ...) is forwarded verbatim
    # to each training run, so the sweep can be smoke-tested or configured as a whole.
    args, extra = parser.parse_known_args()

    if args.run:
        failed = []
        for arm in args.arms:
            if not train_arm(arm, args.backbone, extra):
                failed.append(arm)
        if failed:
            print(f"\n[ablation] arms that failed: {', '.join(failed)}")

    results = load_results(args.backbone, args.arms)
    table = format_table(results, args.backbone, args.fmt)
    print(table)

    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\nWrote table -> {args.out}")


if __name__ == "__main__":
    main()
