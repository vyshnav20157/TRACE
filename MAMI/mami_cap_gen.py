"""
MAMI caption generation: RAM++ tags -> GroundingDINO boxes -> (InternVL | Gemini) caption.

This mirrors the FHM pipeline in `vg_caption_gen.py` but is driven by the MAMI skeleton
JSON produced by `MAMI/build_mami_skeleton.py`, and it splits the two captioners across
two machines via `--generator`:

  * `--generator internvl`  (PRIMARY, this GPU server): runs InternVL locally on the GPU
    and backfills the `ivl_8b_new_caption` field. No API key required. This does the bulk
    of the captioning because Gemini's free tier is too slow for the full 11k images.

  * `--generator gemini`    (SECONDARY, user's local machine): pure Gemini API calls,
    backfills the `gemini_caption` field. Requires GOOGLE_API_KEY.

Both generators read and write the SAME JSON, keyed by `img`, and each only ever touches
its own caption field -- so an InternVL run on the server and a Gemini run on a laptop can
be produced independently and merged simply by copying the JSON between machines. Each run
skips rows whose target field is already populated (resumable), and also records processed
images in a per-generator `.txt` tracker.

The prompts are tuned for MISOGYNY (the MAMI task): they ask the model to surface gender,
role, and the four MAMI dimensions (body-shaming, gender stereotyping, objectification,
violence/threats toward women) *descriptively*, without emitting judgmental labels.

Requires (both generators, for grounding): `ram_plus_swin_large_14m.pth` and
`openimages_rare_200_llm_tag_descriptions.json` (repo root), plus RAM/GroundingDINO deps.

Usage (server, primary):
    python MAMI/mami_cap_gen.py --generator internvl \
        --json /backup/girish_datasets/MAMI/mami_captions_complete.json

Usage (local machine, secondary):
    GOOGLE_API_KEY=... python MAMI/mami_cap_gen.py --generator gemini \
        --json /path/to/mami_captions_complete.json
"""

import mami_gpu  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before torch import)

import argparse
import io
import json
import os
import random
import time
from typing import Dict, List, Tuple

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

# Grounding + captioning model assets (repo-root relative, as in vg_caption_gen.py).
RAM_PRETRAINED_PATH = "ram_plus_swin_large_14m.pth"
RAM_IMAGE_SIZE = 384
INTERNVL_MODEL = "OpenGVLab/InternVL2_5-8B"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

# Per-generator target field + progress tracker.
GENERATOR_CONFIG = {
    "internvl": {
        "field": "ivl_8b_new_caption",
        "tracker": "processed_mami_internvl_images.txt",
    },
    "gemini": {
        "field": "gemini_caption",
        "tracker": "processed_mami_gemini_images.txt",
    },
}

# ---------------------------------------------------------------------------
# Misogyny-tuned prompt (shared wording for both captioners)
# ---------------------------------------------------------------------------
def build_prompt(grounding_prompt: str, for_internvl: bool) -> str:
    """Build a misogyny-aware captioning prompt.

    Foregrounds gender/role and the four MAMI dimensions descriptively, without emitting
    judgmental labels, and keeps captions CLIP-suitable (single caption, < 77 tokens).
    """
    # The <image> placeholder is required by InternVL's chat template; harmless for Gemini.
    return f"""<image>
    {grounding_prompt}
    Task: Analyze this meme image using the above grounding information and generate a **single caption** (under 77 tokens) suitable for CLIP fine-tuning.

    The caption should:
    - Describe the main visual elements (people, objects, setting), noting the gender and role of any depicted person.
    - Summarize the text overlay (if short) or explain its meaning concisely.
    - If the meme concerns women or girls, describe *descriptively* (never with labels like 'misogynous'/'offensive') any of the following when present:
        * body-shaming or mockery of a woman's appearance or body,
        * gender stereotyping (assumptions about women's roles, behavior, or worth),
        * objectification (reducing a woman to a sexual object or body parts),
        * violence, threats, or hostility directed at women.
    - Explain how the text and image interact to target, demean, mock, or stereotype women, if they do.

    Avoid subjective verdicts or judgmental labels. Focus on factual, descriptive language.

    Format the response as:
    Caption: [Generated caption here]
    """


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
    image = Image.open(image_path)
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
# InternVL captioner (primary, GPU)
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


def generate_caption_internvl(image_path: str, grounding_info, model, tokenizer) -> str:
    pixel_values = load_image_internvl(image_path).to(torch.bfloat16).to(device)
    prompt = build_prompt(format_grounding_prompt(grounding_info), for_internvl=True)
    generation_config = dict(max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    caption = response.split("Caption:")[-1].strip() if "Caption:" in response else response.strip()
    return caption


# ---------------------------------------------------------------------------
# Gemini captioner (secondary, API)
# ---------------------------------------------------------------------------
def load_image_for_gemini(image_path):
    image = Image.open(image_path).convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=95)
    img_byte_arr.seek(0)
    return Image.open(img_byte_arr)


def predict_with_retry(model, prompt, image, safety_settings, max_retries=5, initial_delay=5):
    delay = initial_delay
    last_error = None
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0, 1))
            return model.generate_content([prompt, image], safety_settings=safety_settings)
        except Exception as e:  # noqa: BLE001
            last_error = e
            error_str = str(e).lower()
            if any(k in error_str for k in ("429", "resource exhausted", "quota")):
                wait = delay + random.uniform(0, delay * 0.1)
                print(f"Rate limit on attempt {attempt+1}/{max_retries}; waiting {wait:.1f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 60)
            elif any(k in error_str for k in ("500", "503", "backend error")):
                wait = delay * 1.5 + random.uniform(0, delay * 0.2)
                print(f"Server error on attempt {attempt+1}/{max_retries}; waiting {wait:.1f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 45)
            elif attempt < max_retries - 1:
                wait = delay * 0.5 + random.uniform(0, delay * 0.1)
                print(f"Error on attempt {attempt+1}/{max_retries}: {e}; waiting {wait:.1f}s...")
                time.sleep(wait)
                delay = min(delay * 1.5, 30)
            else:
                raise
    print(f"All {max_retries} retries failed. Last error: {last_error}")
    return None


def load_gemini():
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set; required for --generator gemini.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    return model, safety_settings


def generate_caption_gemini(image_path: str, grounding_info, model, safety_settings) -> str:
    time.sleep(3)  # gentle pacing between API calls
    image = load_image_for_gemini(image_path)
    prompt = build_prompt(format_grounding_prompt(grounding_info), for_internvl=False)
    response = predict_with_retry(model, prompt, image, safety_settings)
    if response is None:
        return None
    text = response.text.strip()
    return text.split("Caption:", 1)[1].strip() if "Caption:" in text else text


# ---------------------------------------------------------------------------
# Main JSON-driven loop
# ---------------------------------------------------------------------------
def is_empty(val) -> bool:
    return pd.isna(val) or str(val).strip() in ("", "None", "nan")


def process(json_path: str, generator: str, limit: int = None):
    field = GENERATOR_CONFIG[generator]["field"]
    tracker_file = GENERATOR_CONFIG[generator]["tracker"]

    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if field not in df.columns:
        df[field] = ""

    # Resume support: skip already-populated rows and previously processed images.
    processed = set()
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            processed = {line.strip() for line in f}

    todo = [i for i in range(len(df)) if is_empty(df.at[i, field]) and df.at[i, "img"] not in processed]
    if limit is not None:
        todo = todo[:limit]
    print(f"[{generator}] {len(todo)} rows to caption -> field '{field}' (of {len(df)} total)")
    if not todo:
        return

    # Load grounding models (both generators need these).
    ram_model = load_ram_model(RAM_PRETRAINED_PATH, RAM_IMAGE_SIZE)
    transform = get_transform(image_size=RAM_IMAGE_SIZE)
    gd_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(device)

    # Load the chosen captioner.
    if generator == "internvl":
        internvl_model, internvl_tokenizer = load_internvl()
    else:
        gemini_model, gemini_safety = load_gemini()

    for i in tqdm(todo, desc=f"Captioning ({generator})", unit="image"):
        img_rel = df.at[i, "img"]
        image_path = os.path.join(MAMI_ROOT, img_rel)
        try:
            tags = recognize_tags(image_path, ram_model, transform)
            grounding_info = extract_grounding_info(image_path, tags, gd_processor, gd_model)

            if generator == "internvl":
                caption = generate_caption_internvl(image_path, grounding_info, internvl_model, internvl_tokenizer)
            else:
                caption = generate_caption_gemini(image_path, grounding_info, gemini_model, gemini_safety)

            if caption is None or not str(caption).strip():
                print(f"Empty caption for {img_rel}; skipping.")
                continue

            df.at[i, field] = str(caption).strip()

            # Save after each success (only this generator's field is ever written).
            df.to_json(json_path, orient="records", indent=2)
            processed.add(img_rel)
            with open(tracker_file, "a") as f:
                f.write(img_rel + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {img_rel}: {e}")
            continue

    print(f"[{generator}] done. Wrote captions to {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator",
        choices=["internvl", "gemini"],
        default="internvl",
        help="internvl (primary, local GPU) writes ivl_8b_new_caption; "
        "gemini (secondary, API) writes gemini_caption.",
    )
    parser.add_argument("--json", default=DEFAULT_JSON, help="Path to the MAMI skeleton JSON.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only caption the first N pending rows (for smoke tests).",
    )
    args = parser.parse_args()

    process(args.json, args.generator, args.limit)


if __name__ == "__main__":
    main()
