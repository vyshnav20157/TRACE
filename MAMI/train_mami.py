"""Unified entrypoint to train TRACE on MAMI with a selectable backbone.

Dispatches to one of the three MAMI backbone scripts by lazily importing only the chosen
one (each imports a heavy vision-language model at module load, so we never import more
than we run):

    python MAMI/train_mami.py --backbone roberta   # primary (CLIP-XLM-RoBERTa)
    python MAMI/train_mami.py --backbone vitl14    # CLIP-ViT-L/14
    python MAMI/train_mami.py --backbone siglip2   # SigLIP2

All other flags are forwarded to the selected backbone's own argument parser, e.g.:

    python MAMI/train_mami.py --backbone roberta --subset 200 --epochs 1
    python MAMI/train_mami.py --backbone vitl14 --data-path /path/to/mami.json --wandb

Run the individual scripts directly if you prefer; this is just a convenience wrapper.
"""

import mami_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before the backbone module imports torch)

import argparse
import importlib
import sys

BACKBONES = {
    "roberta": "clip_xlm_roberta_mami",
    "vitl14": "clip_vitL_14_mami",
    "siglip2": "siglip2_mami",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TRACE on MAMI with a selectable backbone.",
        add_help=True,
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONES),
        default="roberta",
        help="Which vision-language backbone to fine-tune (default: roberta, the primary).",
    )
    # Parse only --backbone here; everything else is forwarded to the backbone script.
    args, remaining = parser.parse_known_args()

    module_name = BACKBONES[args.backbone]
    print(f"[train_mami] backbone={args.backbone} -> {module_name}")

    # Hand the remaining args to the backbone module's own parser via sys.argv.
    sys.argv = [module_name] + remaining
    backbone = importlib.import_module(module_name)
    backbone.main()


if __name__ == "__main__":
    main()
