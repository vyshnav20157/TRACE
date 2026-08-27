"""Run the Memotion modality ablation sweep and collect the results table.

This is the driver for the six-arm study defined in `memotion_modality.py`: it trains each
arm in turn (or just a chosen subset), then gathers every arm's test metrics into one table so
the modality comparison can be read off directly instead of being reassembled by hand from
six scattered prediction files.

Memotion Task B is three independent binary problems over the same memes, so the sweep is
per-task: `--task humour` and `--task offensive` are separate six-arm studies with separate
checkpoints, prediction files, and tables. `--task all` runs the tasks in sequence and prints
one table per task.

Each arm is launched as a SEPARATE subprocess rather than by importing the backbone and
looping in-process. That is deliberate: the backbone module loads a multi-GB vision-language
model at import time and holds global CUDA state, and training mutates module-level globals
(the `dataset` global in the roberta script, `calculate_loss_gs.call_count`, the RNG streams
seeded at import). Running arms in one process would leak all of that from one arm into the
next and quietly break the "everything held fixed except the modality" guarantee that makes
the ablation meaningful. A subprocess per arm gets each one a clean interpreter, identical
seeding, and a fresh GPU allocator.

Usage
-----
    # Train every arm for humour sequentially, then print the table:
    python Memotion/memotion_ablation.py --run --task humour

    # Both thesis tasks, one after the other:
    python Memotion/memotion_ablation.py --run --task all

    # Only some arms (e.g. resuming a sweep that was interrupted):
    python Memotion/memotion_ablation.py --run --task humour --arms image_only image_text

    # Just collect and print the table from finished runs (no training):
    python Memotion/memotion_ablation.py --task humour

    # Markdown table, e.g. to paste into the thesis:
    python Memotion/memotion_ablation.py --task all --format markdown

    # Smoke-test the whole sweep end to end in a few minutes:
    python Memotion/memotion_ablation.py --run --task humour --subset 48 --epochs 1 --no-resume

Arms are run one at a time because each needs the whole GPU. To parallelise across two GPUs,
launch two `train_memotion.py --arm ...` commands yourself with different
`CUDA_VISIBLE_DEVICES` (see ToDo.md), then run this script with no `--run` to collect the
table.
"""

import argparse
import json
import os
import subprocess
import sys

from memotion_common import TASK_ORDER
from memotion_modality import ARMS, DEFAULT_CAPTIONER, FIELD_PREFIX, get_arm, preds_name

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "Memotion", "train_memotion.py")

# Columns pulled into the summary table, in order. Memotion's official headline metric is
# macro-F1 at a fixed 0.5 threshold (SemEval-2020 Task 8 Task B), reported first with its
# accuracy and per-class F1s; `f1`/`accuracy`/`auc` are TRACE's own tuned-threshold numbers,
# alongside for comparability with the FHM/MAMI/MMSD results. Keys must match the dict built
# by memotion_metrics.compute_metrics (values there are already 4-dp strings).
TABLE_COLUMNS = [
    ("macro_f1", "Macro-F1"),
    ("official_accuracy", "Acc@.5"),
    ("pos_f1", "F1(pos)"),
    ("neg_f1", "F1(neg)"),
    ("f1", "F1(tuned)"),
    ("accuracy", "Acc(tuned)"),
    ("auc", "AUROC"),
]

# Order arms appear in the table: ablations first, ascending in how much they are given,
# with full TRACE last as the reference the others are read against.
ARM_ORDER = ["image_only", "image_text", "image_taskcap", "image_genericcap",
             "image_unifiedcap", "trace"]

# Tasks the ablation study covers. Memotion has three Task B problems, but the thesis
# extension is scoped to humour and offensiveness; sarcasm is MMSD2.0's job.
ABLATION_TASKS = ["humour", "offensive"]


def train_arm(task, arm, backbone, captioner, extra_args):
    """Train a single (task, arm) in its own subprocess. Returns True on success."""
    cmd = [sys.executable, TRAIN_SCRIPT, "--task", task, "--backbone", backbone,
           "--arm", arm] + extra_args
    print(f"\n{'=' * 78}\n[ablation] training {task} / arm '{arm}' ({backbone})"
          f"\n[ablation] $ {' '.join(cmd)}\n{'=' * 78}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[ablation] {task} / arm '{arm}' FAILED (exit {result.returncode})")
        return False
    return True


def load_results(task, backbone, arms, captioner, smoke=False):
    """Load each arm's test-prediction JSON, skipping arms that have not been run."""
    results = {}
    for arm in arms:
        path = os.path.join(REPO_ROOT, preds_name(task, backbone, arm, captioner, smoke))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            results[arm] = json.load(f)
    return results


def format_table(results, task, backbone, captioner, fmt="text"):
    """Render the collected per-arm metrics as an aligned text or markdown table."""
    arms = [a for a in ARM_ORDER if a in results]
    if not arms:
        return (
            f"\nMemotion modality ablation -- task: {task}, backbone: {backbone}\n"
            f"No results found. Train the arms first:\n"
            f"    python Memotion/memotion_ablation.py --run --task {task}"
        )

    headers = ["Arm", "Text sources"] + [label for _, label in TABLE_COLUMNS]
    rows = []
    for arm in arms:
        record = results[arm]
        metrics = record.get("metrics", {})
        sources = record.get("text_sources") or get_arm(arm)["sources"]
        rendered = "+".join("<const>" if s == "null" else s for s in sources)
        row = [arm, rendered]
        for key, _ in TABLE_COLUMNS:
            row.append(str(metrics.get(key, "-")))
        rows.append(row)

    title = f"Memotion modality ablation -- task: {task}, backbone: {backbone} | captioner: {captioner}"

    if fmt == "markdown":
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(row) + " |" for row in rows]
        return f"\n### {title}\n\n" + "\n".join(out)

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        f"\n{title}",
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
    parser.add_argument("--task", default="humour", choices=TASK_ORDER + ["all"],
                        help="Which Memotion task to sweep, or 'all' for humour then "
                             "offensive (default: humour).")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=ARM_ORDER,
                        help="Which arms to run/collect (default: all six).")
    parser.add_argument("--backbone", default="roberta", choices=["roberta"],
                        help="Backbone to run the sweep on. Roberta only -- the arms are "
                             "implemented in clip_xlm_roberta_memotion.py.")
    parser.add_argument("--captioner", choices=list(FIELD_PREFIX), default=DEFAULT_CAPTIONER,
                        help="Which captioner's caption sets the sweep trains on and "
                             "collects (default: internvl). Run the sweep once per "
                             "captioner to build the captioner-ablation comparison; "
                             "results are filed separately so the two never collide.")
    parser.add_argument("--format", dest="fmt", default="text", choices=["text", "markdown"],
                        help="Table format (default: text).")
    parser.add_argument("--out", default=None,
                        help="Also write the table(s) to this file.")
    # Everything else (--subset, --epochs, --no-resume, --wandb, ...) is forwarded verbatim
    # to each training run, so the sweep can be smoke-tested or configured as a whole.
    args, extra = parser.parse_known_args()

    # A --subset sweep trains smoke runs, which write `_smoke` prediction files; collect
    # from those so the table reflects the run that just happened rather than silently
    # showing stale real results (or nothing at all).
    smoke = any(a == "--subset" or a.startswith("--subset=") for a in extra)

    tasks = ABLATION_TASKS if args.task == "all" else [args.task]

    if args.run:
        failed = []
        for task in tasks:
            for arm in args.arms:
                if not train_arm(task, arm, args.backbone, args.captioner, extra):
                    failed.append(f"{task}/{arm}")
        if failed:
            print(f"\n[ablation] runs that failed: {', '.join(failed)}")

    tables = []
    for task in tasks:
        results = load_results(task, args.backbone, args.arms, args.captioner, smoke)
        tables.append(format_table(results, task, args.backbone, args.captioner, args.fmt))
    output = "\n".join(tables)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print(f"\nWrote table -> {args.out}")


if __name__ == "__main__":
    main()
