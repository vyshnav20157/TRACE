**comment out CLIP from requirements.txt**
**comment out flash-attention and (if required, apex) from requirements.txt**

## How to pip install dependencies

```bash
conda activate trace

python3 -m pip install -U pip wheel
python3 -m pip install "setuptools<81"
python3 -c "import pkg_resources; print('pkg_resources OK')"

python3 -m pip install torch==2.3.1 torchvision==0.18.1
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

python3 -m pip install --no-build-isolation \
  "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

#python -m pip install flash-attn==2.6.3 --no-build-isolation

python3 -m pip install -r requirements.txt --no-build-isolation

python3 -m pip install "open-clip-torch<=2.26.0" google-generativeai statsmodels

python3 -m pip install "timm==0.4.12"
```
## Captioner-ablation env (`trace-qwen`)

Qwen2.5-VL-7B needs `transformers>=4.49`, but the `trace` env is pinned at 4.46.3 because
InternVL, RAM, open_clip and LAVIS all sit on that pin. So Qwen caption GENERATION gets its
own env; merging, training and eval stay in `trace`.

```bash
conda create -y -n trace-qwen python=3.10
conda activate trace-qwen

python3 -m pip install -U pip wheel "setuptools<81"
python3 -m pip install torch==2.3.1 torchvision==0.18.1

# 4.49.0, NOT 4.51: 4.51's timm_wrapper requires a timm newer than any that still exposes
# the legacy timm.models.registry / timm.models.hub APIs that RAM imports.
python3 -m pip install "transformers==4.49.0" "accelerate>=0.26" \
    pandas tqdm pillow sentencepiece protobuf ftfy

# 0.9.16, NOT RAM's pinned 0.4.12: transformers needs timm.data.ImageNetInfo, which 0.4.12
# does not have, and GroundingDINO's AutoModel lookup enumerates every model config so the
# timm wrapper is imported even though no timm model is used here. 0.9.16 has both that
# symbol and every legacy API RAM needs. The `ram 0.0.1 requires timm==0.4.12` pip warning
# is a stale pin, not a real conflict.
python3 -m pip install "timm==0.9.16"

python3 -m pip install --no-build-isolation \
  "ram @ git+https://github.com/xinyu1205/recognize-anything.git@88c2b0ca13e38cca6655f83ad0185271167dbcbf"

# verify
python3 -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForZeroShotObjectDetection
from ram.models import ram_plus
print('trace-qwen OK')
"
```

torch is deliberately the SAME version as in `trace` (2.3.1+cu121), so the shared RAM++ /
GroundingDINO grounding stage behaves identically in both envs -- which is what lets the
captioner ablation attribute its result to the captioner alone.
