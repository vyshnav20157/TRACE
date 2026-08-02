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