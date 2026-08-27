"""Pluggable captioner backends for the captioner ablation (InternVL vs. Qwen2.5-VL).

The captioner ablation asks one question: **how much of TRACE's performance depends on
which VLM wrote the captions?** That question is only answerable if the captioner is the
ONLY thing that differs between the two caption sets. Everything upstream and downstream of
the captioner -- the RAM++ tag pass, the GroundingDINO boxes and their thresholds, the
grounding block prepended to the prompt, the prompt body itself, the decoding strategy, the
token budget the caption is trimmed to, the image list, and the order it is walked in --
must be bit-identical across backends. Any of those drifting turns a "captioner ablation"
into an uncontrolled two-variable comparison.

This module is how that control is enforced *structurally* rather than by discipline. The
per-dataset `*_cap_gen.py` scripts own the pipeline (grounding, budget, resume, merge) and
call in here only for the final image+prompt -> string step. A backend therefore has no way
to influence anything except the caption text, because nothing else is routed through it.

What is held identical
----------------------
    stage                       who owns it              varies by backend?
    --------------------------  -----------------------  ------------------
    RAM++ tags                  <ds>_cap_gen.py          no
    GroundingDINO boxes         <ds>_cap_gen.py          no (same thresholds)
    grounding block text        <ds>_cap_gen.py          no (same formatter)
    prompt body                 <ds>_cap_gen.py          no (verbatim prompts.md)
    image resolution            this module              448x448 both (see below)
    decoding                    this module              no -- greedy, 80 new tokens
    caption trim                <ds>_cap_gen.py          no (SigLIP2 64-token budget)
    image list + order          <ds>_cap_gen.py          no

The two knobs that cannot be made literally identical (each model brings its own weights and
its own chat template) are exactly the intended independent variable.

On image resolution
-------------------
InternVL here is fed a single 448x448 tile (`load_image_internvl` uses a plain Resize, not
InternVL's dynamic multi-tile preprocessing). Qwen2.5-VL's processor would otherwise pick its
own resolution per image from the native aspect ratio, which would confound "different
captioner" with "different amount of visual detail". So the Qwen backend pins
min_pixels = max_pixels = 448*448, giving both models the same visual budget. That is the
single most important control in this file -- see QwenCaptioner.

On the prompt
-------------
Both backends receive the SAME prompt string, built by the dataset script. InternVL's chat
template wants a literal `<image>` placeholder at the top; Qwen's wants the image as a
structured content part. So each backend adapts the *envelope* while the prompt text passes
through byte-for-byte: the InternVL backend keeps the `<image>\n` prefix it is given, and the
Qwen backend strips exactly that prefix and supplies the image through its own message
schema. Neither adds, removes, or reorders a single word of the instruction text.

Environments
------------
Qwen2.5-VL needs transformers >= 4.49; the training env is pinned at 4.46.3 (InternVL, RAM,
open_clip, LAVIS all sit on that pin). The two backends are therefore expected to run in two
different conda envs, which is why this module imports each backend's heavy dependencies
lazily, inside the class, instead of at module import. Importing this file in the training
env with only InternVL installed must not -- and does not -- fail.
"""

import re

# Both backends decode greedily with the same budget. Greedy (do_sample=False) matters more
# than usual here: sampling would add run-to-run variance on top of the captioner difference,
# so a caption-set delta could no longer be attributed to the captioner alone. Held in this
# module rather than per backend so the two cannot drift apart.
CAPTION_MAX_NEW_TOKENS = 80

# The single visual budget both captioners get; see the module docstring. 448 is InternVL's
# native tile size and the resolution the existing InternVL captions were generated at, so
# pinning Qwen to it keeps the already-generated InternVL caption sets valid as the control
# arm -- they do not need regenerating.
IMAGE_SIDE = 448


class InternVLCaptioner:
    """The existing captioner: OpenGVLab/InternVL2_5-8B, greedy, one 448x448 tile.

    This is a faithful extraction of what `<ds>_cap_gen.py` did inline before the ablation
    existed -- same model id, dtype, transform, and generation config -- so the InternVL arm
    of the ablation is the captions already in the dataset JSONs, not a re-run that might
    differ.
    """

    name = "internvl"
    model_id = "OpenGVLab/InternVL2_5-8B"

    def __init__(self, device, model_id=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.torch = torch
        if model_id:
            self.model_id = model_id

        self.model = (
            AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )

    def _load_image(self, image_path):
        import torchvision.transforms as T
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        transform = T.Compose(
            [
                T.Resize((IMAGE_SIDE, IMAGE_SIDE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(image).unsqueeze(0).to(self.device)

    def caption(self, image_path: str, prompt: str) -> str:
        """Caption one image. `prompt` arrives with InternVL's `<image>\\n` prefix intact."""
        pixel_values = self._load_image(image_path).to(self.torch.bfloat16).to(self.device)
        generation_config = dict(
            max_new_tokens=CAPTION_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with self.torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        return response


class QwenCaptioner:
    """The ablation captioner: Qwen/Qwen2.5-VL-7B-Instruct, greedy, pinned to 448x448.

    Controls that matter, and why each is here rather than left at its default:

    * **Resolution.** Qwen2.5-VL's processor is natively dynamic-resolution: left alone it
      picks a token count per image from the aspect ratio, so a tall meme and a wide one get
      different visual budgets and neither matches InternVL's single 448x448 tile. Setting
      `min_pixels == max_pixels == 448*448` forces every image to the same budget InternVL
      gets. Without this the ablation compares "InternVL at 448" against "Qwen at whatever
      it felt like", and a win for either model could be a resolution artifact.

    * **Greedy decoding, same token budget.** `do_sample=False` and the shared
      CAPTION_MAX_NEW_TOKENS, so neither sampling noise nor a longer leash distinguishes the
      arms. Qwen's generation_config ships with sampling params (temperature/top_p/top_k)
      that transformers warns about when do_sample=False; they are cleared explicitly rather
      than merely overridden, so the log stays clean and nothing sampling-related survives.

    * **Prompt text.** The dataset script hands over the InternVL-shaped prompt; the
      `<image>\\n` prefix is stripped (Qwen carries the image as a content part instead) and
      the remaining instruction text is passed through unmodified. Stripping is anchored to
      the start of the string so a `<image>` occurring anywhere else -- it does not, but the
      anchor makes that guarantee explicit -- is left alone.

    * **bfloat16**, matching InternVL's dtype, so the arms differ in weights and not in
      numeric precision. Attention implementation is left at the transformers default
      (SDPA); flash-attention is not installed in this project's envs and enabling it on one
      arm only would be another uncontrolled difference.
    """

    name = "qwen"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Pinned visual budget: exactly one 448x448 tile's worth of pixels, matching InternVL.
    MIN_PIXELS = IMAGE_SIDE * IMAGE_SIDE
    MAX_PIXELS = IMAGE_SIDE * IMAGE_SIDE

    def __init__(self, device, model_id=None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.torch = torch
        if model_id:
            self.model_id = model_id

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(device)
        )
        # min_pixels/max_pixels are processor-level, so every image goes through the same
        # resize regardless of its native size -- this is the resolution control.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )

        # Qwen ships sampling defaults in generation_config; with do_sample=False they are
        # unused but transformers logs a warning for each. Clearing them keeps the run log
        # readable and makes it unambiguous that decoding is greedy.
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, attr):
                setattr(self.model.generation_config, attr, None)
        self.model.generation_config.do_sample = False

    @staticmethod
    def _strip_image_token(prompt: str) -> str:
        """Remove the leading InternVL `<image>` placeholder; leave the instruction intact.

        The dataset scripts build one prompt string for both backends. InternVL's template
        requires the placeholder; Qwen's message schema carries the image itself, and leaving
        a literal `<image>` in the user turn would put a stray token in front of the
        instruction that InternVL's arm does not have.
        """
        return re.sub(r"^<image>\s*", "", prompt)

    def caption(self, image_path: str, prompt: str) -> str:
        """Caption one image, mirroring InternVL's greedy 80-token generation."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._strip_image_token(prompt)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=CAPTION_MAX_NEW_TOKENS,
                do_sample=False,
            )
        # generate() returns prompt + completion; keep only what the model added.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


BACKENDS = {
    "internvl": InternVLCaptioner,
    "qwen": QwenCaptioner,
}

# Caption JSON fields and sidecar/tracker names are namespaced by backend so the two caption
# sets coexist in one dataset JSON and one directory. InternVL keeps its historical `ivl_`
# prefix and un-suffixed sidecar names -- renaming them would invalidate every existing
# caption file, checkpoint, and predictions file in the repo.
FIELD_PREFIX = {
    "internvl": "ivl_caption",
    "qwen": "qwen_caption",
}


def caption_field(backend: str, variant_suffix: str) -> str:
    """JSON field for a (backend, variant) pair, e.g. ('qwen','task') -> 'qwen_caption_task'."""
    return f"{FIELD_PREFIX[backend]}_{variant_suffix}"


def get_backend(name: str):
    """Return the captioner class for `name`, with a helpful error on a typo."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown captioner '{name}'. Choose one of: {', '.join(BACKENDS)}")
    return BACKENDS[name]
