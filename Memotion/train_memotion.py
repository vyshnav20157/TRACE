"""Unified entrypoint to train TRACE on Memotion with a selectable backbone and task.

Memotion 1.0 Task B is three independent binary problems over the same memes. You pick ONE
per run with `--task`; the three are never trained jointly, so each gets its own checkpoint,
predictions file, and metrics:

    python Memotion/train_memotion.py --task humour       # first
    python Memotion/train_memotion.py --task offensive    # second
    python Memotion/train_memotion.py --task sarcasm      # third

Backbone selection dispatches by lazily importing only the chosen module (each imports a
heavy vision-language model at load time, so we never import more than we run):

    python Memotion/train_memotion.py --backbone roberta   # primary/default (CLIP-XLM-RoBERTa)
    python Memotion/train_memotion.py --backbone vitl14    # CLIP-ViT-L/14
    python Memotion/train_memotion.py --backbone siglip2   # SigLIP2

All other flags are forwarded to the selected backbone's own argument parser, e.g.:

    python Memotion/train_memotion.py --task humour --subset 200 --epochs 1
    python Memotion/train_memotion.py --backbone vitl14 --task offensive --wandb

Run the individual scripts directly if you prefer; this is just a convenience wrapper.
"""

import memotion_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before the backbone module imports torch)

import argparse
import importlib
import sys

from memotion_common import TASK_ORDER, describe_task

BACKBONES = {
    "roberta": "clip_xlm_roberta_memotion",
    "vitl14": "clip_vitL_14_memotion",
    "siglip2": "siglip2_memotion",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TRACE on Memotion with a selectable backbone and task.",
        add_help=True,
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONES),
        default="roberta",
        help="Which vision-language backbone to fine-tune (default: roberta, the primary).",
    )
    parser.add_argument(
        "--task",
        choices=TASK_ORDER,
        default="humour",
        help="Which Memotion Task B binary problem to train (default: humour).",
    )
    # Parse only --backbone/--task here; everything else is forwarded to the backbone script.
    args, remaining = parser.parse_known_args()

    module_name = BACKBONES[args.backbone]
    print(f"[train_memotion] backbone={args.backbone} -> {module_name}")
    print(f"[train_memotion] task={describe_task(args.task)}")

    # Hand the remaining args (plus the task) to the backbone module's own parser.
    sys.argv = [module_name, "--task", args.task] + remaining
    backbone = importlib.import_module(module_name)
    backbone.main()


if __name__ == "__main__":
    main()
