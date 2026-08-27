"""MAMI caption generation: RAM++ tags -> GroundingDINO boxes -> VLM caption.

This mirrors `MMSD/mmsd_cap_gen.py` but is driven by the MAMI skeleton JSON produced by
`MAMI/build_mami_skeleton.py`. As with Memotion and MMSD, Gemini is not part of this flow:
only InternVL runs, on the local GPU. No API key is required.

Prompts
-------
MAMI is a single task -- misogyny identification -- so the default `--prompt misogyny` uses
the MISOGYNY prompt from `prompts.md`, which foregrounds exactly what this dataset turns on:
the portrayed role of any depicted women, and any gender-related comparison, insult,
stereotype, or objectification the text and image jointly express.

`--prompt all` (the UNIFIED prompt) and `--prompt generic` are also available, so the
caption-specialization ablation run for Memotion/MMSD can be repeated here: each variant
writes to its OWN field (`ivl_caption_unified`, `ivl_caption_generic`) so it never clobbers
the primary caption, and training picks it up via `--caption-field`.

Prompt wording is verbatim from `prompts.md` at the repo root, which is the single source of
truth -- keep the two in sync when either changes. The 40-word cap in each prompt is sized
to the tightest text encoder in the pipeline (SigLIP2, 64 tokens); see CAPTION_TOKEN_BUDGET
below, which enforces the same budget at generation time.

Every prompt asks for descriptive language and forbids verdict labels ("this is
misogynistic", "this is sexist") -- the classifier must infer the label, so a caption that
states it would leak the target.

Captioner ablation (--captioner)
--------------------------------
`--captioner internvl` (default) is the primary captioner, InternVL2_5-8B, and writes the
`ivl_caption_*` fields every published run uses. `--captioner qwen` runs Qwen2.5-VL-7B-Instruct
instead and writes `qwen_caption_*`, so the two caption sets coexist in this dataset JSON and
can be trained and compared as a captioner ablation.

Only the captioner changes. The RAM++ tag pass, the GroundingDINO boxes and their thresholds,
the grounding block, the prompt text, greedy decoding, the 80-token generation cap, the
SigLIP2 64-token trim, and the image list and its order are all shared code and run
identically for both -- see `utils/captioner_backends.py`, which also pins Qwen to InternVL's
448x448 visual budget (Qwen's processor is dynamic-resolution by default, which would
otherwise confound "different captioner" with "different amount of visual detail").

Each captioner keeps its own sidecars, trackers, caption fields, checkpoints, and predictions
files, so a Qwen run can never overwrite an InternVL result.

Qwen2.5-VL needs transformers >= 4.49, while the training env is pinned at 4.46.3 for
InternVL/RAM/open_clip/LAVIS. Run the Qwen captioning pass in the separate `trace-qwen` env
(see ToDo.md); merging and training happen back in the `trace` env as usual.

Output / concurrency
--------------------
A run writes ONLY its own per-variant sidecar (`mami_captions_complete.<variant>.json`, a
flat {img: caption} map) and never the dataset JSON. That is what makes it safe to caption
several variants at once in two terminals: the previous design loaded the whole dataset JSON
and rewrote it after every image, so two concurrent runs would each serialize a snapshot
taken before the other's captions existed and the slower writer would silently erase the
other's work.

Merge the finished sidecars into the dataset JSON in a single pass afterwards with
`--merge` (run it only when no caption job is active).

Resume behaviour: a run skips images already present in its sidecar or its `.txt` tracker,
so an interrupted run picks up where it left off. The sidecar is written atomically (temp
file + rename, one `.bak` kept), so a crash cannot corrupt it.

Requires: `ram_plus_swin_large_14m.pth` and
`openimages_rare_200_llm_tag_descriptions.json` (repo root), plus RAM/GroundingDINO deps.

Usage:
    python MAMI/mami_cap_gen.py                        # misogyny prompt (primary)
    python MAMI/mami_cap_gen.py --prompt all           # unified-prompt ablation
    python MAMI/mami_cap_gen.py --captioner qwen     # captioner ablation (trace-qwen env)
    python MAMI/mami_cap_gen.py --limit 5              # smoke test

    # Two variants at once, one per GPU (run in separate terminals):
    python MAMI/mami_cap_gen.py --prompt all     --gpu 0
    python MAMI/mami_cap_gen.py --prompt generic --gpu 1
    # ...then, once both have finished:
    python MAMI/mami_cap_gen.py --merge
"""

import os
import sys

# --gpu must take effect BEFORE mami_gpu (and hence torch) is imported, so it is parsed
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

import mami_gpu  # noqa: E402,F401  (pins CUDA_VISIBLE_DEVICES before torch import)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
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
from mami_common import MAMI_IMAGE_ROOT as MAMI_ROOT, MAMI_DATA_PATH as DEFAULT_JSON

# Captioner backends for the captioner ablation. The dataset script owns the whole
# pipeline (grounding, prompt, budget, resume, merge); the backend supplies ONLY the
# image+prompt -> string step, which is what keeps the ablation single-variable.
from utils.captioner_backends import caption_field, get_backend

# Grounding + captioning model assets (repo-root relative, as in vg_caption_gen.py).
RAM_PRETRAINED_PATH = "ram_plus_swin_large_14m.pth"
RAM_IMAGE_SIZE = 384
# Captioner model ids live in utils/captioner_backends.py, one per backend, so the
# ablation's two models are declared side by side with the controls that equalize them.
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

# Which JSON field each prompt variant writes, and its resume tracker. The primary
# 'misogyny' prompt writes the canonical field the training scripts read by default.
#
# `shard_path` gives the per-variant sidecar this run writes to. Runs NEVER write the main
# dataset JSON directly -- that is what makes concurrent runs safe; use `--merge` to fold
# finished sidecars back into the main JSON.
PROMPT_CONFIG = {
    "misogyny": {
        "suffix": "task",
        "legacy_tracker": "processed_mami_misogyny_images.txt",
    },
    "all": {
        "suffix": "unified",
        "legacy_tracker": "processed_mami_unified_images.txt",
    },
    "generic": {
        "suffix": "generic",
        "legacy_tracker": "processed_mami_generic_images.txt",
    },
}


def shard_path(json_path: str, variant: str, backend: str = "internvl") -> str:
    """Path of the per-(variant, backend) caption sidecar.

    Concurrency: two cap-gen runs in two terminals would otherwise both load the whole
    dataset JSON and rewrite it after every image, so each would serialize a snapshot taken
    before the other's captions existed and the slower writer would silently erase the
    faster one's work. Giving every variant its own file removes the shared mutable state
    entirely -- no locking, and a crashed run can never corrupt another variant's captions.

    The sidecar is a flat {img: caption} map, written next to the dataset JSON as
    e.g. `mami_captions_complete.<variant>.json`.

    Captioner ablation: the `qwen` backend appends its name
    (`mami_captions_complete.<variant>.qwen.json`) so its caption sets sit alongside
    InternVL's instead of colliding with them. InternVL keeps the un-suffixed historical
    names -- every sidecar, `.bak`, and tracker already on disk stays exactly where the
    existing runs left it, so switching to this version re-captions nothing.
    """
    base, ext = os.path.splitext(json_path)
    suffix = "" if backend == "internvl" else f".{backend}"
    return f"{base}.{variant}{suffix}{ext}"


def tracker_path(variant: str, backend: str = "internvl") -> str:
    """Resume-tracker path for a (variant, backend) pair.

    InternVL returns the historical per-variant tracker filename recorded in PROMPT_CONFIG,
    so an interrupted InternVL run still resumes off the file it has been appending to.
    Other backends get their own namespaced tracker, so a Qwen run neither reads InternVL's
    completed-image list (which would make it skip every image and produce nothing) nor
    appends to it (which would make a later InternVL run skip images it never captioned).
    """
    if backend == "internvl":
        return PROMPT_CONFIG[variant]["legacy_tracker"]
    return f"processed_mami_{variant}_{backend}_images.txt"


def load_shard(path: str) -> Dict[str, str]:
    """Load a per-variant caption sidecar, tolerating a missing or truncated file."""
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
# prompt wording across datasets. 'misogyny' is the MISOGYNY prompt (the primary for this
# dataset), 'all' is the UNIFIED prompt, and 'generic' is the GENERIC prompt -- the latter
# two exist for the caption-specialization ablation. Every prompt shares the same hard
# constraints: one caption, no more than 40 words, describe the mechanism rather than
# emitting the verdict the classifier is meant to predict.
#
# Each entry is the body that follows the grounding block -- `build_prompt` prepends the
# `<image>` placeholder and the grounding info.

_VARIANT_PROMPT = {
    "misogyny": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If women are depicted or referenced, describe their portrayed role, actions, relationships to other subjects, and any explicit comparisons or stereotypical descriptions expressed by the text or imagery
- If the image and text together convey a gender-related comparison, insult, stereotype, or objectification, describe how the text and image combine to express it using concrete observations; if not, do not invent one
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'misogynistic', 'sexist') — describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",

    "all": """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If a person or group (by gender, race, religion, nationality, appearance, or other identity) is explicitly targeted or described, identify the target and describe any observable insults, comparisons, generalizations, sexualization, or stereotypical statements expressed by the text or imagery
- If the text and image create observable contrast, exaggeration, reversal, incongruity, rhetorical questioning, or wordplay, describe that relationship
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'offensive', 'sarcastic', 'misogynistic', 'funny', 'hateful') — describe the content and mechanism, not a verdict

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
- Avoid judgmental labels — describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]""",
}


def build_prompt(grounding_prompt: str, variant: str) -> str:
    """Build a MAMI captioning prompt for the given prompt variant.

    Prepends the RAM++/GroundingDINO grounding block to the variant's prompt from
    `prompts.md`, so the model describes what was actually detected.
    """
    # The <image> placeholder is required by InternVL's chat template.
    return f"<image>\n{grounding_prompt}\n\n{_VARIANT_PROMPT[variant]}\n"


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
# Caption length budget
# ---------------------------------------------------------------------------
# The three MAMI backbones cap text at different lengths:
#   CLIP ViT-L/14            77 tokens  (clip_vitL_14_mami.py)
#   OpenCLIP XLM-R ViT-H/14  77 tokens  (positional embedding table; a hard limit)
#   SigLIP2                  64 tokens  (siglip2_mami.py)
# Anything longer is silently truncated at train time, and because these prompts put the
# discriminative part (the portrayed role of women, and how text and image combine) at the
# END, truncation preferentially destroys the signal and keeps generic scene description.
# So captions are budgeted to the tightest limit -- SigLIP2's 64 -- and all three backbones
# then see identical, complete captions.
CAPTION_TOKEN_BUDGET = 64
_LENGTH_TOKENIZER = "google/siglip2-base-patch16-224"

# The generation cap (CAPTION_MAX_NEW_TOKENS) now lives in utils/captioner_backends.py so
# both captioners provably share one value; see that module.

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


# ---------------------------------------------------------------------------
# Captioner (backend-dispatched)
# ---------------------------------------------------------------------------
# The model that turns (image, prompt) into a caption is the ONE thing the captioner
# ablation varies. Everything above -- RAM++ tags, GroundingDINO boxes and thresholds,
# the grounding block, the prompt text -- and everything below -- the token budget, the
# image list and its order, resume, merge -- is shared by both backends, so a difference
# between the two caption sets is attributable to the captioner alone.
# See utils/captioner_backends.py for the controls each backend pins (notably the 448x448
# visual budget and greedy decoding, held identical across models).
def load_captioner(backend: str):
    """Instantiate the captioner backend (`internvl` or `qwen`) on this run's device."""
    return get_backend(backend)(device)


def generate_caption(image_path: str, grounding_info, captioner, variant: str) -> str:
    """Caption one image: build the shared prompt, run the backend, trim to budget.

    The prompt is built here, not in the backend, so both backends receive byte-identical
    instruction text. `fit_to_budget` is likewise applied here, so both caption sets are
    trimmed to the same SigLIP2 64-token budget at the same sentence boundaries.
    """
    prompt = build_prompt(format_grounding_prompt(grounding_info), variant)
    response = captioner.caption(image_path, prompt)
    caption = response.split("Caption:")[-1].strip() if "Caption:" in response else response.strip()
    return fit_to_budget(caption)


# ---------------------------------------------------------------------------
# Main JSON-driven loop
# ---------------------------------------------------------------------------
def is_empty(val) -> bool:
    return pd.isna(val) or str(val).strip() in ("", "None", "nan")


def process(json_path: str, variant: str, backend: str, limit: int = None, save_every: int = 20):
    tracker_file = tracker_path(variant, backend)
    out_path = shard_path(json_path, variant, backend)

    # The dataset JSON is read-only here: it supplies the image list, nothing more.
    with open(json_path, "r") as f:
        data = json.load(f)
    images = [row["img"] for row in data]

    # Resume support: the sidecar itself records what is done. The tracker file is still
    # honoured on read so an older interrupted run resumes correctly.
    captions = load_shard(out_path)
    processed = set(captions)
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            processed |= {line.strip() for line in f if line.strip()}

    todo = [img for img in images if img not in processed]
    if limit is not None:
        todo = todo[:limit]
    print(f"[{backend}/{variant}] {len(todo)} images to caption -> {out_path} (of {len(images)} total)")
    print(f"[{backend}/{variant}] GPU: CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if not todo:
        return

    # Load grounding models.
    ram_model = load_ram_model(RAM_PRETRAINED_PATH, RAM_IMAGE_SIZE)
    transform = get_transform(image_size=RAM_IMAGE_SIZE)
    gd_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(device)

    captioner = load_captioner(backend)

    since_save = 0
    for img_rel in tqdm(todo, desc=f"Captioning ({backend}/{variant})", unit="image"):
        image_path = os.path.join(MAMI_ROOT, img_rel)
        try:
            tags = recognize_tags(image_path, ram_model, transform)
            grounding_info = extract_grounding_info(image_path, tags, gd_processor, gd_model)
            caption = generate_caption(image_path, grounding_info, captioner, variant)

            if caption is None or not str(caption).strip():
                print(f"Empty caption for {img_rel}; skipping.")
                continue

            captions[img_rel] = str(caption).strip()
            with open(tracker_file, "a") as f:
                f.write(img_rel + "\n")

            # Flush periodically rather than after every image: the sidecar is small, but
            # rewriting it 11,000 times is pure overhead. The tracker above is appended
            # immediately, and re-captioning a few images after a crash is cheap.
            since_save += 1
            if since_save >= save_every:
                save_shard(out_path, captions)
                since_save = 0
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {img_rel}: {e}")
            continue

    save_shard(out_path, captions)
    print(f"[{backend}/{variant}] done. {len(captions)} captions in {out_path}")
    # The hint carries --captioner too: merging without it would fold this run's sidecar
    # into the OTHER backend's caption field, silently mixing the two arms of the ablation.
    print(f"[{backend}/{variant}] merge into the dataset JSON with: "
          f"python MAMI/mami_cap_gen.py --merge --prompt {variant} --captioner {backend}")


def merge(json_path: str, variants: List[str], backend: str) -> None:
    """Fold per-variant caption sidecars back into the main dataset JSON.

    Run this once the concurrent caption runs have finished. It is the only step that
    writes the dataset JSON, so it must not overlap with a running cap-gen job -- by then
    nothing else is writing, and merging all variants in one pass keeps every field.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    for variant in variants:
        field = caption_field(backend, PROMPT_CONFIG[variant]["suffix"])
        path = shard_path(json_path, variant, backend)
        captions = load_shard(path)
        if not captions:
            print(f"[merge] {backend}/{variant}: no captions at {path}, skipping")
            continue
        if field not in df.columns:
            df[field] = ""
        filled = df["img"].map(captions)
        df[field] = filled.where(filled.notna(), df[field]).fillna("")
        print(f"[merge] {backend}/{variant}: {int(filled.notna().sum())} captions -> '{field}'")

    df.to_json(json_path, orient="records", indent=2)
    print(f"[merge] wrote {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prompt",
        choices=list(PROMPT_CONFIG),
        default="misogyny",
        help="Which prompt variant to run. 'misogyny' (default) writes the canonical "
        "ivl_caption_task the training scripts read; 'all' (unified) and 'generic' write "
        "their own field for a caption-specialization ablation.",
    )
    parser.add_argument("--json", default=DEFAULT_JSON, help="Path to the MAMI skeleton JSON.")
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
        "imported; run one variant per GPU to caption several variants concurrently.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge finished per-variant caption sidecars into the dataset JSON and exit. "
        "With --prompt, merges only that variant; otherwise merges all of them. Run this "
        "after the caption jobs finish, never while one is running.",
    )
    parser.add_argument(
        "--captioner",
        choices=["internvl", "qwen"],
        default="internvl",
        help="Which VLM writes the captions. 'internvl' (default) is the primary "
        "captioner and writes the ivl_caption_* fields the published runs use; 'qwen' "
        "(Qwen2.5-VL-7B-Instruct) writes qwen_caption_* for the captioner ablation. "
        "Everything else about the pipeline is identical between the two, so the caption "
        "sets differ by captioner alone. Qwen needs transformers>=4.49 -- see ToDo.md.",
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
        merge(args.json, [args.prompt] if explicit else list(PROMPT_CONFIG), args.captioner)
        return

    process(args.json, args.prompt, args.captioner, args.limit, args.save_every)


if __name__ == "__main__":
    main()
