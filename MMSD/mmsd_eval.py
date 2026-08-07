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

BACKBONES = {
    "roberta": "clip_xlm_roberta_mmsd",
    "vitl14": "clip_vitL_14_mmsd",
    "siglip2": "siglip2_mmsd",
}


def build_eval_dataset(mod, backbone, eval_data, caption_field):
    """Instantiate the backbone's MemeDatasetJSON for the evaluation split."""
    if backbone == "roberta":
        # roberta's dataset takes (df, preprocess_fn, tokenizer, caption_field).
        return mod.MemeDatasetJSON(eval_data, mod.preprocess, mod.tokenizer, caption_field)
    if backbone == "vitl14":
        return mod.MemeDatasetJSON(eval_data, mod.clip_processor, caption_field)
    # siglip2
    return mod.MemeDatasetJSON(eval_data, mod.siglip_processor, caption_field)


def build_model(mod, backbone):
    """Instantiate the backbone's classifier (matching the training script)."""
    if backbone == "roberta":
        return mod.CLIPClassifier(mod.model)
    if backbone == "vitl14":
        return mod.CLIPClassifier()
    return mod.SigLIP2Classifier()


def run_eval(mod, backbone, model, dataloader, device, loss_config):
    """Call the backbone's evaluate_model with its own signature."""
    if backbone == "roberta":
        # roberta's evaluate_model takes no loss_config argument.
        return mod.evaluate_model(model, [dataloader], device)
    return mod.evaluate_model(model, [dataloader], device, loss_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=list(BACKBONES), default="roberta")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pth).")
    parser.add_argument("--data-path", dest="data_path", default=MMSD_DATA_PATH)
    parser.add_argument("--caption-field", dest="caption_field", default="ivl_8b_new_caption",
                        help="JSON field holding the generated caption. Must match training.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-file", dest="log_file",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalresults.jsonl"),
                        help="Append metrics here as JSON lines (default: MMSD/evalresults.jsonl).")
    args = parser.parse_args()

    module_name = BACKBONES[args.backbone]
    print(f"[mmsd_eval] backbone={args.backbone} -> {module_name}")
    print(f"[mmsd_eval] task={describe_task()}")
    mod = importlib.import_module(module_name)
    device = mod.device

    data = pd.read_json(args.data_path)
    splits = dict(zip(["train", "val", "test"], split_mmsd_data(data)))
    eval_df = splits[args.split]
    print(f"Evaluating on '{args.split}' split ({len(eval_df)} rows)")

    dataset = build_eval_dataset(mod, args.backbone, eval_df, args.caption_field)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_fn)

    model = build_model(mod, args.backbone).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    # Training may have saved under DataParallel ('module.' prefix); strip it.
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys when loading checkpoint.")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint.")
    model.eval()

    # Match the training scripts: select best captions on the eval set before scoring.
    loss_config = {"classification": True, "contrastive": False, "relevance": True}
    best_captions = mod.select_best_captions(model, dataset, device, loss_config)
    dataset.best_captions = best_captions

    result = run_eval(mod, args.backbone, model, dataloader, device, loss_config)
    metrics = result[0] if isinstance(result, tuple) else result
    print(f"\n{args.split.capitalize()} metrics (sarcasm):")
    print(format_metrics(metrics, prefix="  "))

    if args.log_file:
        record = {
            "backbone": args.backbone,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        with open(args.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Logged {args.split} metrics to {args.log_file}")


if __name__ == "__main__":
    main()
