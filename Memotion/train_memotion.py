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

Modality ablations
------------------
`--arm` selects which inputs the model sees, for the modality-importance study. All other
settings are held fixed, so differences between arms are attributable to the modality:

    python Memotion/train_memotion.py --task humour --arm image_only        # image alone
    python Memotion/train_memotion.py --task humour --arm image_text        # + meme OCR text
    python Memotion/train_memotion.py --task humour --arm image_taskcap     # + humour caption
    python Memotion/train_memotion.py --task humour --arm image_genericcap  # + generic caption
    python Memotion/train_memotion.py --task humour --arm image_unifiedcap  # + unified caption
    python Memotion/train_memotion.py --task humour --arm trace             # full TRACE (default)

The three caption arms (image_taskcap / image_genericcap / image_unifiedcap) differ only in
which prompt wrote the caption, so they isolate prompt specificity. Each is trained per task,
since Memotion trains one binary classifier per task.

The arms are ROBERTA-ONLY: `--arm` is implemented in `clip_xlm_roberta_memotion.py`, which is
the primary backbone for the ablation study. Passing a non-default `--arm` with
`--backbone vitl14/siglip2` is rejected rather than silently ignored.

`--arm` is forwarded to the backbone like every other flag; see
`Memotion/memotion_modality.py` for what each arm changes and `Memotion/memotion_ablation.py`
to run/collect the whole sweep.

Run the individual scripts directly if you prefer; this is just a convenience wrapper.
"""

import memotion_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before the backbone module imports torch)

import argparse
import importlib
import sys

from memotion_common import TASK_ORDER, describe_task
from memotion_modality import ARMS, DEFAULT_ARM

BACKBONES = {
    "roberta": "clip_xlm_roberta_memotion",
    "vitl14": "clip_vitL_14_memotion",
    "siglip2": "siglip2_memotion",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TRACE on Memotion with a selectable backbone and task.",
        epilog="Only --backbone/--task/--arm are consumed here; every other flag (--epochs, "
        "--subset, --caption-field, --no-resume, --wandb, ...) is forwarded to the chosen "
        "backbone's parser. Run e.g. `python Memotion/clip_xlm_roberta_memotion.py --help` to "
        "see them, or `python Memotion/memotion_ablation.py --help` for the modality sweep.",
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
    parser.add_argument(
        "--arm",
        choices=list(ARMS),
        default=DEFAULT_ARM,
        help="Modality ablation arm (default: trace, the unmodified architecture). Roberta only; "
        "see Memotion/memotion_modality.py.",
    )
    # Parse only --backbone/--task/--arm here; everything else is forwarded to the backbone.
    args, remaining = parser.parse_known_args()

    # Only the roberta backbone implements the arms. Failing here beats forwarding --arm to a
    # parser that has never heard of it (argparse would abort with a confusing error) or, worse,
    # silently training the full-TRACE arm and filing the result under an ablation name.
    if args.arm != DEFAULT_ARM and args.backbone != "roberta":
        raise SystemExit(
            f"--arm {args.arm} is implemented for --backbone roberta only "
            f"(got '{args.backbone}'). The modality ablation study runs on the primary backbone."
        )

    module_name = BACKBONES[args.backbone]
    print(f"[train_memotion] backbone={args.backbone} -> {module_name}")
    print(f"[train_memotion] task={describe_task(args.task)}")
    print(f"[train_memotion] arm={args.arm}")

    # Hand the remaining args (plus task and arm) to the backbone module's own parser. --arm is
    # re-added explicitly because this parser consumed it (it is declared here so it shows up
    # in --help); the roberta backbone has its own identical --arm option.
    sys.argv = [module_name, "--task", args.task] + remaining
    if args.backbone == "roberta":
        sys.argv += ["--arm", args.arm]
    backbone = importlib.import_module(module_name)
    backbone.main()


if __name__ == "__main__":
    main()
