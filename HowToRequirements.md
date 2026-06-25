## comment out CLIP from requirements.txt
## comment out flash-attention and (if required, apex) from requirements.txt

```bash
conda activate trace

python -m pip install -U pip wheel
python -m pip install "setuptools<81"
python -c "import pkg_resources; print('pkg_resources OK')"

python -m pip install torch==2.3.1 torchvision==0.18.1
python -c "import torch; print(torch.__version__, torch.version.cuda)"

python -m pip install --no-build-isolation \
  "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

#python -m pip install flash-attn==2.6.3 --no-build-isolation

python -m pip install -r requirements.txt --no-build-isolation

python -m pip install open-clip-torch google-generativeai openai statsmodels

python -m pip install "timm==0.4.12"
```