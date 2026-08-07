"""Unified entrypoint to train TRACE on MMSD2.0 with a selectable backbone.

MMSD2.0 (Qin et al., Findings of ACL 2023) is a single binary task -- is this image+text
tweet pair sarcastic? -- so unlike Memotion there is no `--task` flag: one run trains the
one model.

Backbone selection dispatches by lazily importing only the chosen module (each imports a
heavy vision-language model at load time, so we never import more than we run):

    python MMSD/train_mmsd.py                       # primary/default: CLIP-XLM-RoBERTa
    python MMSD/train_mmsd.py --backbone roberta    # the same, named explicitly
    python MMSD/train_mmsd.py --backbone siglip2    # SigLIP2 (secondary)
    python MMSD/train_mmsd.py --backbone vitl14     # CLIP-ViT-L/14 (secondary)

All other flags are forwarded to the selected backbone's own argument parser, e.g.:

    python MMSD/train_mmsd.py --subset 200 --epochs 1
    python MMSD/train_mmsd.py --backbone siglip2 --wandb

Run the individual scripts directly if you prefer; this is just a convenience wrapper.
"""

import mmsd_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before the backbone module imports torch)

import argparse
import importlib
import sys

from mmsd_common import describe_task

BACKBONES = {
    "roberta": "clip_xlm_roberta_mmsd",
    "vitl14": "clip_vitL_14_mmsd",
    "siglip2": "siglip2_mmsd",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TRACE on MMSD2.0 with a selectable backbone.",
        add_help=True,
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONES),
        default="roberta",
        help="Which vision-language backbone to fine-tune (default: roberta, the primary; "
        "siglip2 and vitl14 are the secondary backbones).",
    )
    # Parse only --backbone here; everything else is forwarded to the backbone script.
    args, remaining = parser.parse_known_args()

    module_name = BACKBONES[args.backbone]
    print(f"[train_mmsd] backbone={args.backbone} -> {module_name}")
    print(f"[train_mmsd] task={describe_task()}")

    # Hand the remaining args to the backbone module's own parser.
    sys.argv = [module_name] + remaining
    backbone = importlib.import_module(module_name)
    backbone.main()


if __name__ == "__main__":
    main()
