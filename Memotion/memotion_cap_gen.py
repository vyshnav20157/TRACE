"""
Memotion caption generation: RAM++ tags -> GroundingDINO boxes -> InternVL caption.

This mirrors `MAMI/mami_cap_gen.py` but is driven by the Memotion skeleton JSON produced by
`Memotion/build_memotion_skeleton.py`. Gemini has been dropped from this flow: only
InternVL runs, on the local GPU, backfilling `ivl_caption_unified`. No API key is required.

Prompts
-------
Memotion Task B is three tasks over the SAME memes, and captioning 8,397 images is by far
the most expensive step in the pipeline -- so by default a single task-neutral prompt
(`--prompt all`) produces one caption set that covers all three dimensions, and humour,
offensive, and sarcasm runs all train on it. That keeps captioning to one pass and keeps
the three tasks comparable (identical inputs, only the label changes).

Per-task prompts are also available (`--prompt humour|offensive|sarcasm`) for a
task-specialized caption ablation; each writes to its OWN field
(`ivl_caption_humour`, ...) so it never clobbers the shared caption. These are what the
modality ablation's `image_taskcap` arm reads (see `Memotion/memotion_modality.py`).

`--prompt generic` is the fifth variant: a plain descriptive prompt with NO task cues at all,
written to `ivl_caption_generic`. It is the `image_genericcap` arm of the modality ablation,
and is deliberately NOT the same as `--prompt all` -- 'all' is the union of every task's cues,
which makes it more task-loaded than any single task prompt, not less.

Prompt wording is verbatim from `prompts.md` at the repo root, which is the single source
of truth -- keep the two in sync when either changes. The 40-word cap in each prompt is
sized to the tightest text encoder in the pipeline (SigLIP2, 64 tokens); see
CAPTION_TOKEN_BUDGET below, which enforces the same budget at generation time.

Every prompt asks for descriptive language and forbids verdict labels ("this is funny",
"this is offensive") -- the classifier must infer the label, so a caption that states it
would leak the target.

Output / concurrency
--------------------
A run writes ONLY its own per-task sidecar (`memotion_captions_complete.<task>.json`, a
flat {img: caption} map) and never the dataset JSON. That is what makes it safe to caption
several tasks at once on different GPUs: previously every run loaded the whole dataset JSON
and rewrote it after each image, so two concurrent runs would each serialize a stale
snapshot and the slower writer would erase the other's captions.

Merge the finished sidecars into the dataset JSON in a single pass afterwards with
`--merge` (run it only when no caption job is active).

Resume behaviour: a run skips images already present in its sidecar or its `.txt` tracker,
so an interrupted run picks up where it left off. The sidecar is written atomically
(temp file + rename, one `.bak` kept), so a crash cannot corrupt it.

Requires: `ram_plus_swin_large_14m.pth` and
`openimages_rare_200_llm_tag_descriptions.json` (repo root), plus RAM/GroundingDINO deps.

Usage:
    python Memotion/memotion_cap_gen.py                      # unified caption, all tasks
    python Memotion/memotion_cap_gen.py --prompt humour      # humour-specialized ablation
    python Memotion/memotion_cap_gen.py --prompt generic     # generic caption (modality arm 4)
    python Memotion/memotion_cap_gen.py --limit 5            # smoke test

    # Two tasks at once, one per GPU (run in separate terminals):
    python Memotion/memotion_cap_gen.py --prompt humour    --gpu 0
    python Memotion/memotion_cap_gen.py --prompt offensive --gpu 1
    # ...then, once both have finished:
    python Memotion/memotion_cap_gen.py --merge
"""

import os
import sys

# --gpu must take effect BEFORE memotion_gpu (and hence torch) is imported, so it is parsed
# straight off sys.argv here rather than in main(). Once CUDA is initialized it is too late
# to change which devices are visible. argparse still declares --gpu so it shows up in
# --help and is accepted normally.
for _i, _arg in enumerate(sys.argv[1:]):
    _gpu = None
    if _arg == "--gpu" and _i + 2 < len(sys.argv):
        _gpu = sys.argv[_i + 2]
    elif _arg.startswith("--gpu="):
        _gpu = _arg.split("=", 1)[1]
    if _gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
        break

import memotion_gpu  # noqa: E402,F401  (pins CUDA_VISIBLE_DEVICES before torch import)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile

# One Memotion train image (got_GOT-Meme-9.png) is missing its trailing PNG chunk;
# the pixel data decodes fine apart from the last few rows, so tolerate it rather
# than dropping the sample.
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    AutoTokenizer,
)

from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram_plus

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
from memotion_common import MEMOTION_IMAGE_ROOT as MEMOTION_ROOT, MEMOTION_DATA_PATH as DEFAULT_JSON

# Grounding + captioning model assets (repo-root relative, as in vg_caption_gen.py).
RAM_PRETRAINED_PATH = "ram_plus_swin_large_14m.pth"
RAM_IMAGE_SIZE = 384
INTERNVL_MODEL = "OpenGVLab/InternVL2_5-8B"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

# Which JSON field each prompt variant writes, and its resume tracker. The shared 'all'
# prompt writes the canonical field the training scripts read by default.
#
# `shard` is the per-task sidecar this run writes to (see `shard_path`). Runs NEVER write
# the main dataset JSON directly -- that is what makes concurrent runs safe; use
# `--merge` to fold finished shards back into the main JSON.
PROMPT_CONFIG = {
    "all": {
        "field": "ivl_caption_unified",
        "tracker": "processed_memotion_internvl_images.txt",
    },
    "humour": {
        "field": "ivl_caption_humour",
        "tracker": "processed_memotion_humour_images.txt",
    },
    "offensive": {
        "field": "ivl_caption_offensive",
        "tracker": "processed_memotion_offensive_images.txt",
    },
    "sarcasm": {
        "field": "ivl_caption_sarcasm",
        "tracker": "processed_memotion_sarcasm_images.txt",
    },
    # The GENERIC prompt: plain description with no task cues at all. This is the
    # `image_genericcap` arm of the modality ablation (Memotion/memotion_modality.py) and is
    # NOT the same thing as 'all' -- 'all' is the union of every task's cues, which makes it
    # more task-loaded than any single task prompt, not less.
    "generic": {
        "field": "ivl_caption_generic",
        "tracker": "processed_memotion_generic_images.txt",
    },
}


def shard_path(json_path: str, task: str) -> str:
    """Path of the per-task caption sidecar for `task`.

    Concurrency: two cap-gen runs on two GPUs would otherwise both load the whole dataset
    JSON and rewrite it after every image, so each would serialize a snapshot taken before
    the other's captions existed and the slower writer would silently erase the faster
    one's work. Giving every task its own file removes the shared mutable state entirely --
    no locking, and a crashed run can never corrupt another task's captions.

    The sidecar is a flat {img: caption} map, written next to the dataset JSON as
    e.g. `memotion_captions_complete.humour.json`.
    """
    base, ext = os.path.splitext(json_path)
    return f"{base}.{task}{ext}"


def load_shard(path: str) -> Dict[str, str]:
    """Load a per-task caption sidecar, tolerating a missing or truncated file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        # A crash mid-write can leave a partial file; the .bak is the last good flush.
        backup = path + ".bak"
        if os.path.exists(backup):
            print(f"WARNING: {path} is unreadable; recovering from {backup}")
            with open(backup, "r") as f:
                return json.load(f)
        print(f"WARNING: {path} is unreadable and has no backup; starting empty")
        return {}


def save_shard(path: str, captions: Dict[str, str]) -> None:
    """Atomically write the sidecar: temp file + rename, keeping one backup.

    os.replace is atomic on POSIX, so an interrupted run leaves either the old file or the
    new one -- never a half-written JSON.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(captions, f, indent=2)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Verbatim from `prompts.md` at the repo root, which is the single source of truth for
# prompt wording across datasets. `all` uses the UNIFIED prompt (one caption set covering
# all three dimensions); the per-task variants use the task-specific prompts. Every prompt
# shares the same hard constraints: one caption, 35-50 words, describe the mechanism rather
# than emitting the verdict the classifier is meant to predict.
#
# Each entry is the body that follows the grounding block -- `build_prompt` prepends the
# `<image>` placeholder and the grounding info.

_TASK_PROMPT = {
    "all": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If a person or group (by gender, race, religion, nationality, appearance, or other identity) is explicitly targeted or described, identify the target and describe any observable insults, comparisons, generalizations, sexualization, or stereotypical statements expressed by the text or imagery
- If the text and image create observable contrast, exaggeration, reversal, incongruity, rhetorical questioning, or wordplay, describe that relationship
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'offensive', 'sarcastic', 'misogynistic', 'funny', 'hateful') - describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",

    "humour": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If the content creates an expectation and then subverts, exaggerates, or twists it, describe the setup and the turn; if the content is presented straight with no comedic turn, state that
- If a comedic device is present (exaggeration, absurdity, wordplay), name it and describe how it operates here; if no such device is present, do not invent one
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'funny', 'not funny') - describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",

    "offensive": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If the meme directs its text or imagery at a person or group (individual, profession, nationality, appearance, belief), name the target; if no target is identifiable, state that
- If the text or imagery contains insults, profanity, derogatory comparisons, threats, wishes of harm, or negative generalizations toward the identified target, describe those elements and how the image and text reinforce each other, noting whether profanity is used as general emphasis or directed at the target and whether the phrasing is aggressive or neutral; if the tone is benign or self-directed, state that
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'offensive', 'harmless') - describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",

    "sarcasm": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If the text and image create a contrast, contradiction, reversal, exaggeration, or other observable incongruity between what the text claims and what the image shows, describe it explicitly
- Describe linguistic cues such as hyperbole, rhetorical questions, exaggerated praise, or obvious understatement when they are explicitly present in the text
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'sarcastic', 'ironic') - describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",

    "generic": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels - describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",
}


def build_prompt(grounding_prompt: str, task: str) -> str:
    """Build a Memotion captioning prompt for the given task variant.

    Prepends the RAM++/GroundingDINO grounding block to the task's prompt from
    `prompts.md`, so the model describes what was actually detected.
    """
    # The <image> placeholder is required by InternVL's chat template.
    return f"<image>\n{grounding_prompt}\n\n{_TASK_PROMPT[task]}\n"


# ---------------------------------------------------------------------------
# Grounding: RAM++ tags -> GroundingDINO boxes
# ---------------------------------------------------------------------------
def load_ram_model(pretrained_path, image_size):
    model = ram_plus(pretrained=pretrained_path, image_size=image_size, vit="swin_l")
    model.eval()
    return model.to(device)


def recognize_tags(image_path: str, model, transform) -> str:
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        tags = inference(image, model)
    return tags[0]


def extract_grounding_info(image_path: str, tags: str, processor, model) -> Dict[str, List[Tuple]]:
    image = Image.open(image_path).convert("RGB")
    text = tags.lower().replace(" | ", ". ") + "." if tags else "no tags."

    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]

    grounding_info: Dict[str, List[Tuple]] = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i.item(), 2) for i in box]
        score = round(score.item(), 2)
        grounding_info.setdefault(label, []).append(tuple(box + [score]))
    return grounding_info


def format_grounding_prompt(grounding_info: Dict[str, List[Tuple]]) -> str:
    grounding_prompt = "Key elements in the image:"
    if grounding_info:
        for obj, boxes in grounding_info.items():
            confidence = max(box[4] for box in boxes)
            if confidence > 0.7:
                certainty = "high certainty"
            elif confidence > 0.5:
                certainty = "moderate certainty"
            else:
                certainty = "low certainty"
            grounding_prompt += f" {obj} (detected with {certainty}, count: {len(boxes)}),"
        grounding_prompt = grounding_prompt.rstrip(",") + "."
    else:
        grounding_prompt += " No specific object information available."
    return grounding_prompt


# ---------------------------------------------------------------------------
# InternVL captioner
# ---------------------------------------------------------------------------
def load_internvl():
    model = (
        AutoModel.from_pretrained(
            INTERNVL_MODEL,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_image_internvl(image_path, input_size=448):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Caption length budget
# ---------------------------------------------------------------------------
# The three Memotion backbones cap text at different lengths:
#   CLIP ViT-L/14            77 tokens  (clip_vitL_14_memotion.py)
#   OpenCLIP XLM-R ViT-H/14  77 tokens  (positional embedding table; a hard limit)
#   SigLIP2                  64 tokens  (siglip2_memotion.py)
# Anything longer is silently truncated at train time, and because these prompts put the
# discriminative part (target, comedic turn, whether the image reinforces or contradicts
# the text) at the END, truncation preferentially destroys the signal and keeps generic
# scene description. So captions are budgeted to the tightest limit -- SigLIP2's 64 -- and
# all three backbones then see identical, complete captions.
CAPTION_TOKEN_BUDGET = 64
_LENGTH_TOKENIZER = "google/siglip2-base-patch16-224"

# ~80 new tokens leaves headroom for the "Caption:" preamble while still stopping the
# model from rambling into a 120-word paragraph.
CAPTION_MAX_NEW_TOKENS = 80

_length_tokenizer = None


def get_length_tokenizer():
    """Lazily load the tokenizer used to measure captions against the budget."""
    global _length_tokenizer
    if _length_tokenizer is None:
        _length_tokenizer = AutoTokenizer.from_pretrained(_LENGTH_TOKENIZER)
    return _length_tokenizer


def fit_to_budget(caption: str, budget: int = CAPTION_TOKEN_BUDGET) -> str:
    """Trim `caption` to `budget` tokens at a sentence boundary.

    The backbones would otherwise each hack off a different arbitrary tail mid-word. Here
    the caption is cut back to the last COMPLETE sentence that fits, so what gets stored is
    always a well-formed string. Falls back to a word-boundary cut when even the first
    sentence is over budget.
    """
    tok = get_length_tokenizer()
    if len(tok(caption)["input_ids"]) <= budget:
        return caption

    # Prefer dropping whole trailing sentences.
    parts = re.split(r"(?<=[.!?])\s+", caption.strip())
    kept = ""
    for part in parts:
        candidate = (kept + " " + part).strip()
        if len(tok(candidate)["input_ids"]) > budget:
            break
        kept = candidate
    if kept:
        return kept

    # First sentence alone is too long: cut on a word boundary instead.
    words = caption.split()
    while words and len(tok(" ".join(words))["input_ids"]) > budget:
        words.pop()
    return " ".join(words).rstrip(",;:")


def generate_caption_internvl(image_path: str, grounding_info, model, tokenizer, task: str) -> str:
    pixel_values = load_image_internvl(image_path).to(torch.bfloat16).to(device)
    prompt = build_prompt(format_grounding_prompt(grounding_info), task)
    generation_config = dict(
        max_new_tokens=CAPTION_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    caption = response.split("Caption:")[-1].strip() if "Caption:" in response else response.strip()
    return fit_to_budget(caption)


# ---------------------------------------------------------------------------
# Main JSON-driven loop
# ---------------------------------------------------------------------------
def is_empty(val) -> bool:
    return pd.isna(val) or str(val).strip() in ("", "None", "nan")


def process(json_path: str, task: str, limit: int = None, save_every: int = 20):
    field = PROMPT_CONFIG[task]["field"]
    tracker_file = PROMPT_CONFIG[task]["tracker"]
    out_path = shard_path(json_path, task)

    # The dataset JSON is read-only here: it supplies the image list, nothing more.
    with open(json_path, "r") as f:
        data = json.load(f)
    images = [row["img"] for row in data]

    # Resume support: the sidecar itself records what is done. The legacy per-task tracker
    # is still honoured on read so an older interrupted run resumes correctly.
    captions = load_shard(out_path)
    processed = set(captions)
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            processed |= {line.strip() for line in f if line.strip()}

    todo = [img for img in images if img not in processed]
    if limit is not None:
        todo = todo[:limit]
    print(f"[{task}] {len(todo)} images to caption -> {out_path} (of {len(images)} total)")
    print(f"[{task}] GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if not todo:
        return

    # Load grounding models.
    ram_model = load_ram_model(RAM_PRETRAINED_PATH, RAM_IMAGE_SIZE)
    transform = get_transform(image_size=RAM_IMAGE_SIZE)
    gd_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(device)

    internvl_model, internvl_tokenizer = load_internvl()

    since_save = 0
    for img_rel in tqdm(todo, desc=f"Captioning ({task})", unit="image"):
        image_path = os.path.join(MEMOTION_ROOT, img_rel)
        try:
            tags = recognize_tags(image_path, ram_model, transform)
            grounding_info = extract_grounding_info(image_path, tags, gd_processor, gd_model)
            caption = generate_caption_internvl(
                image_path, grounding_info, internvl_model, internvl_tokenizer, task
            )

            if caption is None or not str(caption).strip():
                print(f"Empty caption for {img_rel}; skipping.")
                continue

            captions[img_rel] = str(caption).strip()
            with open(tracker_file, "a") as f:
                f.write(img_rel + "\n")

            # Flush periodically rather than after every image: the sidecar is small, but
            # rewriting it 8,397 times is pure overhead. The tracker above is appended
            # immediately, and re-captioning a few images after a crash is cheap.
            since_save += 1
            if since_save >= save_every:
                save_shard(out_path, captions)
                since_save = 0
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {img_rel}: {e}")
            continue

    save_shard(out_path, captions)
    print(f"[{task}] done. {len(captions)} captions in {out_path}")
    print(f"[{task}] merge into the dataset JSON with: "
          f"python Memotion/memotion_cap_gen.py --merge --prompt {task}")


def merge(json_path: str, tasks: List[str]) -> None:
    """Fold per-task caption sidecars back into the main dataset JSON.

    Run this once the concurrent caption runs have finished. It is the only step that
    writes the dataset JSON, so it must not overlap with a running cap-gen job -- by then
    nothing else is writing, and merging all tasks in one pass keeps every field.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    for task in tasks:
        field = PROMPT_CONFIG[task]["field"]
        path = shard_path(json_path, task)
        captions = load_shard(path)
        if not captions:
            print(f"[merge] {task}: no captions at {path}, skipping")
            continue
        if field not in df.columns:
            df[field] = ""
        filled = df["img"].map(captions)
        df[field] = filled.where(filled.notna(), df[field]).fillna("")
        print(f"[merge] {task}: {int(filled.notna().sum())} captions -> '{field}'")

    df.to_json(json_path, orient="records", indent=2)
    print(f"[merge] wrote {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prompt",
        choices=list(PROMPT_CONFIG),
        default="all",
        help="Which prompt variant to run. 'all' (default) writes the shared "
        "ivl_caption_unified used by every task; the per-task variants write their own "
        "field for a task-specialized caption ablation; 'generic' writes ivl_caption_generic "
        "(plain description, no task cues) for the modality ablation's image_genericcap arm.",
    )
    parser.add_argument("--json", default=DEFAULT_JSON, help="Path to the Memotion skeleton JSON.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only caption the first N pending rows (for smoke tests).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Which GPU to run on (see also CUDA_VISIBLE_DEVICES). Handled before torch is "
        "imported; run one task per GPU to caption several tasks concurrently.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge finished per-task caption sidecars into the dataset JSON and exit. "
        "With --prompt, merges only that task; otherwise merges all of them. Run this "
        "after the caption jobs finish, never while one is running.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Flush the caption sidecar to disk every N images (default: 20).",
    )
    args = parser.parse_args()

    if args.merge:
        # --prompt has a default, so only treat it as a filter if given explicitly.
        explicit = any(a.startswith("--prompt") for a in sys.argv[1:])
        merge(args.json, [args.prompt] if explicit else list(PROMPT_CONFIG))
        return

    process(args.json, args.prompt, args.limit, args.save_every)


if __name__ == "__main__":
    main()
