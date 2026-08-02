# Thesis Contributions

This file documents **my extensions to the TRACE codebase** (original work by Girish A.
Koushik et al., see [`README.md`](README.md)) as part of my thesis. It is maintained
separately from the upstream `README.md` and is updated as my work progresses.

**Author:** Vyshnav
**Base work:** TRACE — *Textual Relevance Augmentation and Contextual Encoding for
Multimodal Hate Detection* (Koushik, Treharne, Joshi, Kanojia; AAAI 2026 AISI).
**My Work:**  

---

## Thesis goal

Evaluate whether the TRACE architecture — visual grounding + relevance-aware caption
augmentation + PEFT fine-tuning of CLIP-family models — generalizes beyond hate-speech
detection to other tasks like **misogyny detection**, benchmarking it against the current SOTA on the
**MAMI (Multimedia Automatic Misogyny Identification)** dataset.

---

## Contribution log

### 1. Environment reproducibility (first commit)

Getting TRACE's dependency stack to install cleanly was non-trivial (CLIP, flash-attention,
`timm`/`setuptools` version pins, build isolation issues). To make the environment
reproducible I added:

- **`HowToRequirements.md`** — a step-by-step install recipe for the `trace` conda
  environment: the exact `torch`/`torchvision` versions, which lines to comment out of `requirements.txt` (CLIP, flash-attention/apex), and the follow-up installs (`open-clip-torch<=0.26.0`,`google-generativeai`, `openai`, `statsmodels`, `timm==0.4.12`).
- **`requirements.txt`** — reconciled to the versions that actually resolve in the `trace`
  environment.

### 2. Extending TRACE to the MAMI dataset (current work)

The core thesis contribution: a full MAMI workflow that reuses TRACE's shared machinery
(caption selection, Gumbel-Softmax caption weighting, the classification/relevance/
contrastive loss stack) **without modifying it**, so results are directly comparable to the
original TRACE method. All new code lives in **`MAMI/`**.

MAMI's primary task ("misogynous" 0/1) is a balanced binary problem — structurally
identical to FHM/MultiOFF — so it maps onto TRACE's binary head, sigmoid + Focal loss, and
AUROC-driven training with no changes to `utils/`. Only **Task A** (the binary label) is
modeled; the four sub-category labels (shaming, stereotype, objectification, violence) are
preserved in the dataset JSON but not trained on.

#### New files

| File | Purpose |
|------|---------|
| `MAMI/build_mami_skeleton.py` | Converts MAMI's three TSVs (`train`/`validation`/`test`) into the FHM-style dataset JSON. Normalizes the `mifile_name` header typo in `test.tsv`; stores split-aware `img` paths (`training_images/…`, `test_images/…`) so a single image root resolves every split; carries the four sub-category columns through unused. Modeled on `utils/build_fhm_skeleton.py`. |
| `MAMI/mami_cap_gen.py` | Caption generation (RAM++ tags → GroundingDINO grounding → captioner) with a `--generator {internvl,gemini}` flag and **misogyny-tuned prompts**. See the two-machine design below. |
| `MAMI/mami_common.py` | Shared MAMI config (image root, dataset-JSON path, `train/val/test` split helper) and a `sys.path` shim so the `MAMI/` scripts can import the repo's `utils/` package when launched as `python MAMI/…`. |
| `MAMI/train_mami.py` | Unified training entrypoint with `--backbone {roberta,vitl14,siglip2}`; lazily imports only the chosen backbone (each loads a heavy VLM at import). |
| `MAMI/clip_xlm_roberta_mami.py` | CLIP-XLM-RoBERTa backbone (**primary**), adapted from `clip_xlm_roberta_ft.py`. |
| `MAMI/clip_vitL_14_mami.py` | CLIP-ViT-L/14 backbone, adapted from `clip_vitL_14_ft.py`. |
| `MAMI/siglip2_mami.py` | SigLIP2 backbone, adapted from `siglip2_ft.py`. |
| `MAMI/mami_eval.py` | Standalone evaluator: loads a trained checkpoint into the matching backbone class and reports test-set metrics. |

#### Design decisions

- **Reuse over rewrite.** The MAMI backbone scripts are minimal adaptations of the FHM
  scripts — the only changes are: dataset path, image root, `train/val/test` split logic
  (replacing FHM's `dev_seen`/`dev_unseen`/`test_seen`/`test_unseen` scheme), checkpoint and
  predictions filenames, the wandb dataset tag, and added CLI flags (`--data-path`,
  `--epochs`, `--subset`, `--wandb`). The model architecture, loss functions, caption
  selection, and metrics are untouched, keeping the comparison to original TRACE clean.

- **Two-machine captioning split.** Caption generation is split across machines to work
  around Gemini's slow free tier:
  - **InternVL (primary)** runs on the GPU server, writes `ivl_8b_new_caption`. No API key.
  - **Gemini (secondary)** runs on a local machine (because pure API calls), writes `gemini_caption`.

  Both generators read/write the **same JSON keyed by `img`**, each touches only its own
  field, and each skips already-populated rows — so the two runs are independent, resumable,
  and merge simply by copying the JSON between machines. Training adapts to whatever captions
  are present: `[text, ivl, gemini]`, `[text, ivl]`, or `[text]` alone before captioning.

- **Misogyny-tuned prompts.** The captioning prompts (shared by both generators) were
  rewritten from the generic meme/hate wording to foreground gender and the four MAMI
  dimensions (body-shaming, gender stereotyping, objectification, violence/threats toward
  women) *descriptively*, while keeping the "no judgmental labels" and "CLIP-suitable,
  < 77 tokens" constraints.

---

## Verification status

- **Skeleton builder** — verified end-to-end: produces 11,000 records (balanced
  5,500/5,500), splits 9,000/1,000/1,000, and all image paths resolve.
- **Training pipeline** — verified end-to-end on the `vitl14` backbone (`--subset 40
  --epochs 1`): data + images load, captions correctly fall back to `[text]` pre-captioning,
  loss decreases, all six metrics (loss/accuracy/precision/recall/F1/AUROC) print, and the
  predictions JSON is written.
- **Caption generation** — non-model logic (path/field/tracker mapping, empty-row
  filtering, misogyny-prompt rendering) verified. A live run additionally requires the
  RAM++ weights (`ram_plus_swin_large_14m.pth`) at the repo root and a GPU-loaded InternVL.

### Environment notes

- A live `mami_cap_gen.py` run requires `ram_plus_swin_large_14m.pth` at the repo root.

---

## Quickstart for MAMI

```bash
# 1. Build the MAMI dataset JSON from the raw TSVs
python MAMI/build_mami_skeleton.py

# 2. Generate captions
python MAMI/mami_cap_gen.py --generator internvl              # primary, GPU server

# 3. Train (roberta is primary; vitl14 / siglip2 also available)
python MAMI/train_mami.py --backbone roberta

# 4. Evaluate a saved checkpoint
python MAMI/mami_eval.py --backbone roberta --checkpoint checkpoints/mami_roberta_best_model.pth
```
