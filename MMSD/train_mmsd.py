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

Modality ablations
------------------
`--arm` selects which inputs the model sees, for the modality-importance study. All other
settings are held fixed, so differences between arms are attributable to the modality:

    python MMSD/train_mmsd.py --arm image_only        # image alone
    python MMSD/train_mmsd.py --arm image_text        # image + the tweet's own text
    python MMSD/train_mmsd.py --arm image_taskcap     # image + sarcasm-prompt caption
    python MMSD/train_mmsd.py --arm image_genericcap  # image + generic-prompt caption
    python MMSD/train_mmsd.py --arm image_unifiedcap  # image + unified (all-task) caption
    python MMSD/train_mmsd.py --arm trace             # full TRACE (default)

The three caption arms (image_taskcap / image_genericcap / image_unifiedcap) differ only in
which prompt wrote the caption, so they isolate prompt specificity.

`--arm` is forwarded to the backbone like every other flag; see `MMSD/mmsd_modality.py` for
what each arm changes and `MMSD/mmsd_ablation.py` to run/collect the whole sweep.

Run the individual scripts directly if you prefer; this is just a convenience wrapper.
"""

import mmsd_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before the backbone module imports torch)

import argparse
import importlib
import sys

from mmsd_common import describe_task
from mmsd_modality import ARMS, DEFAULT_ARM

BACKBONES = {
    "roberta": "clip_xlm_roberta_mmsd",
    "vitl14": "clip_vitL_14_mmsd",
    "siglip2": "siglip2_mmsd",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TRACE on MMSD2.0 with a selectable backbone.",
        epilog="Only --backbone is consumed here; every other flag (--arm, --epochs, "
        "--subset, --caption-field, --no-resume, --wandb, ...) is forwarded to the chosen "
        "backbone's parser. Run e.g. `python MMSD/clip_xlm_roberta_mmsd.py --help` to see "
        "them, or `python MMSD/mmsd_ablation.py --help` for the modality-ablation sweep.",
        add_help=True,
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONES),
        default="roberta",
        help="Which vision-language backbone to fine-tune (default: roberta, the primary; "
        "siglip2 and vitl14 are the secondary backbones).",
    )
    parser.add_argument(
        "--arm",
        choices=list(ARMS),
        default=DEFAULT_ARM,
        help="Modality ablation arm (default: trace, the unmodified architecture). Forwarded "
        "to the backbone; see MMSD/mmsd_modality.py.",
    )
    # Parse only --backbone/--arm here; everything else is forwarded to the backbone script.
    args, remaining = parser.parse_known_args()

    module_name = BACKBONES[args.backbone]
    print(f"[train_mmsd] backbone={args.backbone} -> {module_name}")
    print(f"[train_mmsd] task={describe_task()}")
    print(f"[train_mmsd] arm={args.arm}")

    # Hand the remaining args to the backbone module's own parser. --arm is re-added
    # explicitly because this parser consumed it (it is declared here only so it shows up
    # in --help); the backbone has its own identical --arm option.
    sys.argv = [module_name, "--arm", args.arm] + remaining
    backbone = importlib.import_module(module_name)
    backbone.main()


if __name__ == "__main__":
    main()
