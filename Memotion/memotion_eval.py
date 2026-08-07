"""Standalone Memotion evaluator: load a trained checkpoint and report test-set metrics.

Reuses each backbone's own model/dataset/eval code from the Memotion training scripts (so
the architecture always matches the checkpoint), and evaluates on the Memotion `test` split.
This mirrors the final test evaluation the training scripts already run, but lets you
re-evaluate a saved checkpoint without retraining.

Reported metrics follow `Memotion/memotion_metrics.py`: the OFFICIAL Memotion headline
number is macro-F1 at a fixed 0.5 threshold (SemEval-2020 Task 8 Task B), with TRACE's
tuned-threshold accuracy/precision/recall/F1/AUROC reported alongside for comparability
with the FHM/MultiOFF/MAMI results elsewhere in this repo.

    python Memotion/memotion_eval.py --task humour --backbone roberta \
        --checkpoint checkpoints/memotion_humour_roberta_best_model.pth
    python Memotion/memotion_eval.py --task offensive --backbone vitl14 \
        --checkpoint checkpoints/memotion_offensive_vitl14_best_model.pth

Checkpoints are the ones written by the training scripts (a dict with 'model_state_dict').
"""

import memotion_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before torch import)

import argparse
import importlib
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from memotion_common import (
    MEMOTION_DATA_PATH,
    TASK_ORDER,
    apply_task_labels,
    describe_task,
    split_memotion_data,
)
from memotion_metrics import format_metrics

BACKBONES = {
    "roberta": "clip_xlm_roberta_memotion",
    "vitl14": "clip_vitL_14_memotion",
    "siglip2": "siglip2_memotion",
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
    parser.add_argument("--task", choices=TASK_ORDER, default="humour",
                        help="Which Memotion Task B binary problem to evaluate (default: humour).")
    parser.add_argument("--backbone", choices=list(BACKBONES), default="roberta")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pth).")
    parser.add_argument("--data-path", dest="data_path", default=MEMOTION_DATA_PATH)
    parser.add_argument("--caption-field", dest="caption_field", default="ivl_8b_new_caption",
                        help="JSON field holding the generated caption. Must match training.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-file", dest="log_file",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalresults.jsonl"),
                        help="Append metrics here as JSON lines (default: Memotion/evalresults.jsonl).")
    args = parser.parse_args()

    module_name = BACKBONES[args.backbone]
    print(f"[memotion_eval] backbone={args.backbone} -> {module_name}")
    print(f"[memotion_eval] task={describe_task(args.task)}")
    mod = importlib.import_module(module_name)
    device = mod.device

    data = pd.read_json(args.data_path)
    data = apply_task_labels(data, args.task)
    splits = dict(zip(["train", "val", "test"], split_memotion_data(data)))
    eval_df = splits[args.split]
    print(f"Evaluating on '{args.split}' split ({len(eval_df)} rows)")

    dataset = build_eval_dataset(mod, args.backbone, eval_df, args.caption_field)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_fn)

    model = build_model(mod, args.backbone).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("task") not in (None, args.task):
        print(
            f"WARNING: checkpoint was trained on task '{checkpoint['task']}' but you asked "
            f"for '{args.task}'. Metrics below will be meaningless."
        )
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
    print(f"\n{args.split.capitalize()} metrics ({args.task}):")
    print(format_metrics(metrics, prefix="  "))

    if args.log_file:
        record = {
            "task": args.task,
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
