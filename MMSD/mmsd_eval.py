"""Standalone MMSD2.0 evaluator: load a trained checkpoint and report test-set metrics.

Reuses each backbone's own model/dataset/eval code from the MMSD training scripts (so the
architecture always matches the checkpoint), and evaluates on the official MMSD2.0 `test`
split. This mirrors the final test evaluation the training scripts already run, but lets you
re-evaluate a saved checkpoint without retraining.

Reported metrics follow `MMSD/mmsd_metrics.py`: the OFFICIAL MMSD2.0 headline numbers are
accuracy and the sarcastic-class Precision/Recall/F1 at a fixed 0.5 threshold (the protocol
used by Qin et al. and the MMSD literature), with macro-averaged values and TRACE's
tuned-threshold accuracy/precision/recall/F1/AUROC reported alongside for comparability with
the FHM/MultiOFF/MAMI/Memotion results elsewhere in this repo.

    python MMSD/mmsd_eval.py --backbone roberta \
        --checkpoint checkpoints/mmsd_roberta_best_model.pth
    python MMSD/mmsd_eval.py --backbone siglip2 \
        --checkpoint checkpoints/mmsd_siglip2_best_model.pth

Checkpoints are the ones written by the training scripts (a dict with 'model_state_dict').
"""

import mmsd_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before torch import)

import argparse
import importlib
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mmsd_common import MMSD_DATA_PATH, describe_task, split_mmsd_data
from mmsd_metrics import format_metrics
from mmsd_modality import (
    ARMS,
    DEFAULT_ARM,
    DEFAULT_CAPTIONER,
    FIELD_PREFIX,
    arm_sources,
    check_override,
    checkpoint_name,
    default_caption_field,
    describe_arm,
    loss_config_for,
)

# Imported from utils directly rather than through the backbone module: only the roberta
# scripts re-export these, and eval also dispatches to vitl14/siglip2.
from utils.caption_selection import (
    format_selection_distribution,
    select_best_captions,
    selection_distribution,
)

BACKBONES = {
    "roberta": "clip_xlm_roberta_mmsd",
    "vitl14": "clip_vitL_14_mmsd",
    "siglip2": "siglip2_mmsd",
}


def build_eval_dataset(mod, backbone, eval_data, caption_field, sources):
    """Instantiate the backbone's MemeDatasetJSON for the evaluation split.

    `sources` is the modality arm's text-source list and must match the one training used --
    evaluating an `image_only` checkpoint against the full TRACE caption list would feed the
    model inputs it never saw.
    """
    if backbone == "roberta":
        # roberta's dataset takes (df, preprocess_fn, tokenizer, caption_field, sources).
        return mod.MemeDatasetJSON(eval_data, mod.preprocess, mod.tokenizer, caption_field, sources)
    if backbone == "vitl14":
        return mod.MemeDatasetJSON(eval_data, mod.clip_processor, caption_field, sources)
    # siglip2
    return mod.MemeDatasetJSON(eval_data, mod.siglip_processor, caption_field, sources)


def build_model(mod, backbone):
    """Instantiate the backbone's classifier (matching the training script)."""
    if backbone == "roberta":
        return mod.CLIPClassifier(mod.model)
    if backbone == "vitl14":
        return mod.CLIPClassifier()
    return mod.SigLIP2Classifier()


def run_eval(mod, backbone, model, dataloader, device, loss_config):
    """Call the backbone's evaluate_model. All three now take the same signature."""
    return mod.evaluate_model(model, [dataloader], device, loss_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=list(BACKBONES), default="roberta")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a trained checkpoint (.pth). Defaults to the checkpoint "
                             "the given --backbone/--arm combination writes.")
    parser.add_argument("--data-path", dest="data_path", default=MMSD_DATA_PATH)
    parser.add_argument("--arm", choices=list(ARMS), default=DEFAULT_ARM,
                        help="Modality ablation arm the checkpoint was TRAINED with (default: "
                             "trace). Must match training, or the model is fed inputs it never saw.")
    parser.add_argument("--captioner", choices=list(FIELD_PREFIX), default=DEFAULT_CAPTIONER,
                        help="Which captioner's caption set to evaluate against "
                             "(default: internvl). Must match the captioner the "
                             "checkpoint was trained with -- it selects both the caption "
                             "columns read and the checkpoint/predictions filenames.")
    parser.add_argument("--caption-field", dest="caption_field_override", default=None,
                        help="Override the JSON caption field the arm reads. For the "
                             "task/generic/unified prompt comparison use the dedicated arms "
                             "(image_taskcap / image_genericcap / image_unifiedcap) instead. "
                             "Must match training.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-file", dest="log_file",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalresults.jsonl"),
                        help="Append metrics here as JSON lines (default: MMSD/evalresults.jsonl).")
    args = parser.parse_args()

    # Same guard as training: the checkpoint is chosen by arm name alone, so a mismatched
    # --caption-field would score one arm's model against another prompt's captions.
    check_override(args.arm, args.caption_field_override)

    module_name = BACKBONES[args.backbone]
    print(f"[mmsd_eval] backbone={args.backbone} -> {module_name}")
    print(f"[mmsd_eval] task={describe_task()}")
    print(f"[mmsd_eval] arm={describe_arm(args.arm, args.caption_field_override, args.captioner)}")
    mod = importlib.import_module(module_name)
    device = mod.device

    checkpoint_path = args.checkpoint or os.path.join(
        "checkpoints", checkpoint_name(args.backbone, args.arm, args.captioner)
    )
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    sources = arm_sources(args.arm, args.caption_field_override, args.captioner)
    # The secondary backbones read a single concrete field; it must follow the captioner
    # so an eval never scores a qwen checkpoint against internvl captions.
    caption_field = args.caption_field_override or next(
        (s for s in sources if s not in ("text", "null")),
        default_caption_field(args.captioner),
    )

    data = pd.read_json(args.data_path)
    splits = dict(zip(["train", "val", "test"], split_mmsd_data(data)))
    eval_df = splits[args.split]
    print(f"Evaluating on '{args.split}' split ({len(eval_df)} rows)")

    dataset = build_eval_dataset(mod, args.backbone, eval_df, caption_field, sources)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_fn)

    model = build_model(mod, args.backbone).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    # Training may have saved under DataParallel ('module.' prefix); strip it.
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading checkpoint.")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint.")
    model.eval()

    # Match the training scripts: select best captions on the eval set before scoring, and
    # only when the arm actually has multiple candidates to choose between.
    loss_config = loss_config_for(args.arm)
    caption_choices = None
    if len(sources) > 1:
        dataset.best_captions, caption_choices = select_best_captions(
            model, dataset, device, loss_config, return_choices=True
        )
        # Same discrimination check as training: report which source the scorer picked.
        print(format_selection_distribution(
            caption_choices, sources, title=f"{args.split.capitalize()} caption selection"
        ))
    else:
        print(f"Skipping caption selection (arm '{args.arm}' has a single text source).")

    result = run_eval(mod, args.backbone, model, dataloader, device, loss_config)
    metrics = result[0] if isinstance(result, tuple) else result
    print(f"\n{args.split.capitalize()} metrics (sarcasm):")
    print(format_metrics(metrics, prefix="  "))

    if args.log_file:
        record = {
            "backbone": args.backbone,
            "arm": args.arm,
            "text_sources": sources,
            "checkpoint": checkpoint_path,
            "split": args.split,
            "caption_selection": (
                {src: [n, frac] for src, (n, frac) in
                 selection_distribution(caption_choices, sources).items()}
                if caption_choices else None
            ),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        with open(args.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Logged {args.split} metrics to {args.log_file}")


if __name__ == "__main__":
    main()
