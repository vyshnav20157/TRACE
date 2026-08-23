# Thesis Contributions

This file documents **my extensions to the TRACE codebase** (original work by Girish A.
Koushik et al., see [`README.md`](README.md)) as part of my thesis. It is maintained
separately from the upstream `README.md` and is updated as my work progresses.

**Author:** Vyshnav \
**Base work:** TRACE — *Textual Relevance Augmentation and Contextual Encoding for
Multimodal Hate Detection* (Koushik, Treharne, Joshi, Kanojia; AAAI 2026 AISI).
 

---

## Thesis goal

Evaluate whether the TRACE architecture - visual grounding + relevance-aware caption
augmentation + PEFT fine-tuning of CLIP-family models - generalizes beyond hate-speech
detection to other multimodal meme-understanding tasks, benchmarking it against the current
SOTA on:

- **MAMI (Multimedia Automatic Misogyny Identification)** — misogyny detection.
- **Memotion 1.0 (SemEval-2020 Task 8)** — humour, sarcasm, and offensiveness detection.
- **MMSD2.0 (Findings of ACL 2023)** — multimodal sarcasm detection on Twitter image+text
  pairs.

Memotion extends the question beyond *harm* detection: humour and sarcasm are figurative,
non-harmful properties, so they test whether TRACE's relevance-aware captioning helps with
image-text incongruity in general, or only with the harmful-content signal it was designed
for.

MMSD2.0 sharpens that same question on the cleanest available testbed. Sarcasm here is
*defined* by incongruity between what the text says and what the image shows — which is
precisely the relationship TRACE's captioning prompts are written to describe — and unlike
Memotion it is a large (24,635-pair), de-biased benchmark with an established leaderboard.
It also moves the domain off memes entirely: these are Twitter photos with the text supplied
separately, not image macros with overlaid text, so it tests whether the approach survives
without the OCR-style text-in-image signal that FHM, MAMI, and Memotion all provide.

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
| `MAMI/mami_metrics.py` | Dual-metric evaluator: MAMI-official macro-F1 @ 0.5 **and** TRACE's tuned-threshold metrics + AUROC, from one pass. |
| `MAMI/mami_cap_gen.py` | Caption generation (RAM++ tags → GroundingDINO grounding → InternVL) with the **misogyny prompt** as primary, `--prompt misogyny\|all\|generic` for the caption-specialization ablation, and per-variant sidecars for concurrent runs. Gemini removed. |
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

- **Official-metric reporting, MAMI's convention.** MAMI Sub-task A (SemEval-2022 Task 5)
  ranks on **macro-F1 at a fixed 0.5 threshold**. `mami_metrics.py` reports macro-F1 @ 0.5 as the headline
  (alongside accuracy, per-class F1, and positive-class P/R) with TRACE's tuned-threshold
  metrics + AUROC retained under `tuned_*` for comparability with the FHM/MultiOFF numbers
  in this repo. **Selection and early stopping follow `macro_f1`.** Naming and the
  `SELECTION_METRIC` indirection match `MMSD/mmsd_metrics.py`, so the three training loops
  read identically. 

- **Unified and generic prompts for the ablation.** `--prompt all` (UNIFIED) and
  `--prompt generic` (GENERIC) are wired up alongside the primary, each writing its **own**
  field (`ivl_caption_unified` / `ivl_caption_generic`) so a variant run never clobbers the
  primary captions. Training and eval select a caption set with `--caption-field`, so the
  caption-specialization ablation run for Memotion/MMSD can be repeated here. 

- **Caption length budgeted to the tightest encoder.** Captions are now capped at 64 tokens
  (`CAPTION_TOKEN_BUDGET`), SigLIP2's limit and the smallest of the three backbones (the
  other two allow 77). These prompts put the discriminative content *last*, so silent
  truncation at train time would preferentially destroy the signal and keep generic scene
  description. Overruns are trimmed back to the last complete sentence, so what is stored is
  always well-formed. The previous prompt asked for "< 77 tokens" in prose and enforced
  nothing, generating captions up to `max_new_tokens=1024`.

- **Gemini removed.** Captioning is InternVL-only, matching Memotion and MMSD. The dataset classes now read `[text, ivl_caption]` (or `[text]` alone before
  captioning).

### 3. Extending TRACE to Memotion 1.0 (current work)

A full Memotion workflow that, like the MAMI one, reuses TRACE's shared machinery
(caption selection, Gumbel-Softmax caption weighting, the classification/relevance/
contrastive loss stack) **without modifying it**. All new code lives in **`Memotion/`**;
the dataset stays in place at `/home/vyshnav/MHA-MEME/dataset` and is only pointed at.

Memotion differs from FHM/MAMI in three ways that drove the design:

1. **Three tasks, not one.** Memotion Task B labels every meme for humour, sarcasm, *and*
   offensiveness. These are modeled as three independent binary problems selected with
   `--task`, never trained jointly — so each has its own checkpoint, predictions file, and
   metrics, and they can be run one at a time (humour → offensive → sarcasm).
2. **A different official metric.** Memotion ranks on **macro-F1 at a fixed 0.5
   threshold**, whereas TRACE reports macro P/R/F1 at the F1-*optimal* threshold plus
   AUROC. Both are now reported (see below).
3. **No validation split.** Memotion ships only train/test, so one had to be carved out.

#### New files

| File | Purpose |
|------|---------|
| `Memotion/build_memotion_skeleton.py` | Converts `train_binary.csv`/`test_binary.csv` into the FHM-style dataset JSON. Stores all three task labels side by side, split-aware `img` paths (`train_meme_images/…`, `test_meme_images/…`), and carves the stratified val split. |
| `Memotion/memotion_common.py` | Shared config: dataset/image paths, the **`TASKS` registry**, `apply_task_labels()` (projects the chosen task onto the generic `label` column), the split helper, and the `sys.path` shim for importing the repo's `utils/`. |
| `Memotion/memotion_metrics.py` | The dual-metric evaluator: Memotion-official macro-F1 @ 0.5 **and** TRACE's tuned-threshold metrics, from one pass over the predictions. |
| `Memotion/memotion_cap_gen.py` | Caption generation (RAM++ tags → GroundingDINO grounding → InternVL). Gemini removed; **task-tuned prompts** via `--prompt`. |
| `Memotion/train_memotion.py` | Unified entrypoint with `--backbone {roberta,vitl14,siglip2}` and `--task {humour,offensive,sarcasm}`; lazily imports only the chosen backbone. |
| `Memotion/clip_xlm_roberta_memotion.py` | CLIP-XLM-RoBERTa backbone (**primary/default**). |
| `Memotion/clip_vitL_14_memotion.py` | CLIP-ViT-L/14 backbone. |
| `Memotion/siglip2_memotion.py` | SigLIP2 backbone. |
| `Memotion/memotion_eval.py` | Standalone evaluator: loads a checkpoint into the matching backbone and reports test metrics for a given task. |
| `Memotion/memotion_gpu.py` | GPU pinning, imported first by every entrypoint (mirrors `MAMI/mami_gpu.py`). |

#### Design decisions

- **One label column, three tasks.** Rather than three dataset JSONs, the skeleton stores
  `humour_label`, `sarcasm_label`, and `offensive_label` together, and
  `apply_task_labels()` copies the selected one onto the generic `label` field that the
  shared `utils/` machinery expects. This means **captions are generated once and reused by
  all three tasks**, and the three runs differ *only* in the label — making cross-task
  comparison clean, and cutting the most expensive pipeline stage (captioning 8,397 images)
  from three passes to one.

- **Dual metric reporting.** `memotion_metrics.py` reports the **official** macro-F1 @ 0.5
  as the headline (comparable to published Memotion results) alongside TRACE's
  tuned-threshold accuracy/P/R/F1/AUROC (comparable to the FHM/MultiOFF/MAMI numbers in
  this repo). Model selection and early stopping follow the **official** metric, replacing
  MAMI's AUROC-driven selection. Per-class F1 (`neg_f1`/`pos_f1`) is also reported, because
  these tasks are 61–78% positive and macro-F1 alone hides which class is failing — on the
  test priors an all-positive predictor scores 0.77 accuracy but only 0.44 macro-F1.

- **Stratified validation split.** Memotion has no val split, and using test for early
  stopping would bias the headline numbers. The builder carves a deterministic 10% of train
  (seed 42) stratified on the **joint** (humour, sarcasm, offensive) label triple, so a
  single shared split preserves every task's positive rate — verified to within 0.001 of
  the train marginals — and all three tasks train on identical rows.

- **Task-tuned prompts.** Prompts were rewritten per task to foreground the cues each
  depends on: comedic mechanism and setup/twist for humour; literal-vs-intended meaning and
  irony markers for sarcasm; targets, slurs, and stereotyping for offensiveness. All of
  them forbid stating the verdict ("funny", "sarcastic", "offensive") — a caption asserting
  the label would leak the target into the model's input. The default `--prompt all`
  produces one shared caption set covering all three dimensions; the per-task variants
  write to their own fields for a caption-specialization ablation.

### 4. Extending TRACE to MMSD2.0 (current work)

A full MMSD2.0 workflow that, like the MAMI and Memotion ones, reuses TRACE's shared
machinery (caption selection, Gumbel-Softmax caption weighting, the classification/
relevance/contrastive loss stack) **without modifying it**. All new code lives in
**`MMSD/`**; the dataset stays in place at `/home/vyshnav/MMSD2.0` and is only read from.

MMSD2.0 is the *easiest* structural fit of the three extensions — one balanced binary task
with official splits — but the *hardest* plumbing fit, because of how it is distributed.
Three differences drove the design:

1. **Images are not files.** The HuggingFace release ships six parquet shards with the
   images stored as bytes inside the table (`image: struct<bytes, path>`), not as a
   directory of JPEGs. Every other dataset in this repo — and the entire captioning and
   training pipeline — is built around opening an image by path.
2. **One task, official splits.** Unlike Memotion there is no task registry and no `--task`
   flag, and unlike Memotion nothing has to be carved out of train: MMSD ships
   train/validation/test (19,816 / 2,410 / 2,409). The skeleton writes the generic `label`
   column directly, so no `apply_task_labels`-style projection is needed.
3. **A different official metric again.** MMSD2.0 ranks on accuracy and the **sarcastic-class
   (binary) F1 at a fixed 0.5 threshold**, not Memotion's macro-F1 and not TRACE's
   tuned-threshold AUROC.

#### New files

| File | Purpose |
|------|---------|
| `MMSD/build_mmsd_skeleton.py` | Reads the parquet shards, **extracts all 24,635 images** to `<root>/<split>/<id>.jpg`, and writes the FHM-style dataset JSON. Resumable and idempotent (`--overwrite-images` to force, `--skip-images` for JSON-only). |
| `MMSD/mmsd_common.py` | Shared config: parquet/image/dataset-JSON paths, the `MMSD_VERSION` selector, the split helper, and the `sys.path` shim for importing the repo's `utils/`. |
| `MMSD/mmsd_metrics.py` | Dual-metric evaluator: MMSD-official accuracy + sarcastic-class P/R/F1 @ 0.5 **and** TRACE's tuned-threshold metrics + AUROC, from one pass. |
| `MMSD/mmsd_cap_gen.py` | Caption generation (RAM++ tags → GroundingDINO grounding → InternVL) with the **sarcasm prompt** as primary and `--prompt all\|generic` for the caption-specialization ablation. |
| `MMSD/train_mmsd.py` | Unified entrypoint with `--backbone {roberta,vitl14,siglip2}`; lazily imports only the chosen backbone. No `--task` flag — MMSD is one task. |
| `MMSD/clip_xlm_roberta_mmsd.py` | CLIP-XLM-RoBERTa backbone (**primary/default**). |
| `MMSD/siglip2_mmsd.py` | SigLIP2 backbone (secondary). |
| `MMSD/clip_vitL_14_mmsd.py` | CLIP-ViT-L/14 backbone (secondary). |
| `MMSD/mmsd_eval.py` | Standalone evaluator: loads a checkpoint into the matching backbone and reports test metrics. |
| `MMSD/mmsd_gpu.py` | GPU pinning, imported first by every entrypoint (mirrors `MAMI/mami_gpu.py`). |

#### Design decisions

- **Extract images once, then look like every other dataset.** Rather than teaching the
  captioning script and all three backbones to read parquet, the builder decodes the
  embedded bytes to JPEG files a single time and stores a split-aware relative `img` path.
  After that step MMSD is indistinguishable from MAMI/Memotion downstream, so the backbone
  scripts needed **no parquet code path at all** — which is what kept them near-mechanical
  adaptations of the Memotion ones. Images are re-encoded through PIL rather than dumped
  raw: this normalizes mode and guarantees each file actually decodes, so a corrupt row
  fails during the 10-minute build instead of hours into a GPU captioning run.

- **Official-metric reporting, MMSD's convention this time.** `mmsd_metrics.py` reports
  accuracy and **sarcastic-class P/R/F1 @ 0.5** as the headline (comparable to published
  MMSD2.0 results), with macro-averaged values and TRACE's tuned-threshold metrics + AUROC
  alongside (comparable to the FHM/MultiOFF/MAMI/Memotion numbers here). Model selection and
  early stopping follow `binary_f1`. Note the **naming differs from Memotion on purpose**:
  there the plain keys (`accuracy`, `f1`) were the tuned values and macro-F1 was the
  headline; here the plain keys are the official @0.5 values and the tuned ones are prefixed
  `tuned_`, because that is each dataset's own convention. `SELECTION_METRIC` names the
  selection key in both, so the training loops read identically.

- **Sarcasm prompt as primary.** MMSD is a single task, so there is no prompt matrix: the
  default uses the SARCASM prompt from `prompts.md` verbatim, which foregrounds exactly the
  signal this dataset turns on (observable contrast/reversal between text and image, plus
  hyperbole, rhetorical questions, and exaggerated praise). The UNIFIED and GENERIC prompts
  are wired up as `--prompt all|generic`, each writing its **own** field, so the
  caption-specialization ablation run for Memotion can be repeated here.

- **Grounding carries more weight here.** These are Twitter photos, not image macros: there
  is usually no text overlay in the image at all, and the tweet text arrives separately in
  the `text` field. The RAM++/GroundingDINO stage is therefore doing more of the work than
  on Memotion/MAMI, and the prompts' "if a listed signal is absent, do not invent it" rule
  is what keeps the model from hallucinating an overlay that isn't there.

- **Concurrency-safe captioning, carried over.** The per-variant sidecar + atomic-write +
  resume-tracker design from Memotion is reused unchanged. It matters more here: at 24,635
  images this is the longest captioning run in the thesis, so resuming an interrupted job
  correctly is essential rather than convenient.

- **Reuse over rewrite, again.** The three backbone scripts are mechanical adaptations of
  the Memotion ones. Diffed against their sources, the only changes are: the GPU-pin and
  config imports, the image root, the removal of the task-selection/`apply_task_labels`
  block, the wandb tags, checkpoint/predictions filenames, and comment wording. **The model
  architecture, loss functions, caption selection, and metrics plumbing are untouched**,
  keeping the comparison to original TRACE clean.

### 5. Modality-importance ablation across all three datasets (current work)

A six-arm ablation, run identically on MAMI, Memotion, and MMSD2.0, that answers *where
TRACE's performance actually comes from*: the image, the meme's own text, the generated
caption, or the caption-scoring architecture itself. Every arm holds backbone, splits,
losses, schedule, and seed fixed and changes only the text stream, so the deltas between
arms are attributable to the modality.

| Arm | Text stream fed to the model | Caption scoring |
|-----|------------------------------|-----------------|
| `image_only` | a fixed constant string | off |
| `image_text` | the meme's own OCR / tweet text | off |
| `image_taskcap` | task-specific caption | off |
| `image_genericcap` | generic-prompt caption | off |
| `image_unifiedcap` | unified (all-task) caption | off |
| `trace` | text + **all three** generated captions | **on** |

#### New files

| File | Purpose |
|------|---------|
| `MAMI/mami_modality.py`, `Memotion/memotion_modality.py`, `MMSD/mmsd_modality.py` | Arm definitions and everything derived from them: the ordered text-source list per arm, the per-arm loss config, per-arm checkpoint/predictions filenames, and `resolve_arm()`, which validates an arm's caption fields against the dataset JSON before training starts. |
| `MAMI/mami_ablation.py`, `Memotion/memotion_ablation.py`, `MMSD/mmsd_ablation.py` | Sweep drivers: train every arm (or a chosen subset), then collect all arms' test metrics into one text or markdown table. |

#### Design decisions

- **One code path, one command per arm.** `--arm` is a flag on the existing training
  entrypoints rather than a separate script, and `trace` is the default — so the reference
  arm *is* the unmodified pipeline, and every number in the ablation table comes out of the
  same code. The arms are roberta-only (the primary backbone); passing a non-default `--arm`
  with `vitl14`/`siglip2` is rejected rather than silently ignored and filed under an
  ablation name.

- **Single-caption arms disable the relevance loss.** With one candidate the Gumbel-Softmax
  selection degenerates correctly on its own (softmax over a length-1 vector is exactly 1.0),
  so no code in `utils/` changes. The relevance loss does *not* degenerate harmlessly: it
  would train the caption scorer to predict the label from one fixed text, adding a second
  text-only classifier alongside the real one. Leaving it on would make the single-caption
  arms "TRACE *plus* an extra head" rather than "TRACE *minus* caption selection", so every
  arm except `trace` runs with `relevance: False`.

- **`image_only` uses a constant string, not the empty string.** CLIP-family backbones here
  have no image-only forward path — the classifier head consumes a fused image+text vector,
  so some text must be supplied. The empty string is not neutral: it still tokenizes to a
  BOS/EOS pair the model can key on. Every sample instead gets the *same* fixed placeholder,
  so the text branch contributes one constant vector across the whole dataset and any
  accuracy above chance is attributable to the image branch.

- **Every arm files its own results.** Checkpoint and predictions filenames are keyed on the
  arm (and, for Memotion, the task), so arms never overwrite one another and can train
  concurrently on separate GPUs. `--caption-field` overrides are rejected when they would
  file one caption set's results under another arm's name — that would silently overwrite a
  real arm's numbers with a different prompt's — and are rejected on `trace` outright, since
  it has multiple caption slots and no single one to override.

- **Caption fields are validated before training.** `resolve_arm()` fails fast if an arm's
  caption field is missing from the dataset JSON or present but empty. Training on a column
  of empty strings would otherwise silently degrade e.g. `image_genericcap` into an
  image-only run and quietly invalidate the whole table.

### 6. Caption-selection instrumentation (current work)

`trace` now ranks four highly-correlated candidates (the same image described under three
different prompts), which raises a question the previous two-candidate setup did not: is the
scorer genuinely discriminating between them, or has it collapsed onto one source, making
the selection machinery an expensive no-op?

`utils/caption_selection.py` already computed the winning index and then discarded it. It
now optionally returns it (`return_choices=True`), and two new helpers —
`selection_distribution()` and `format_selection_distribution()` — tally which source won
across the split and render it for logs, with an explicit warning when ≥95% of picks land on
a single source. This is wired into all three training scripts and all three evaluators:
printed after the selection pass, saved into the predictions/log JSON under
`caption_selection`, and logged to wandb as `CaptionSelection/<source>` when `--wandb` is on.

```
Test set caption selection (n=24):
  text                      13   54.2%  ######################
  ivl_caption_task           5   20.8%  ########
  ivl_caption_generic        3   12.5%  #####
  ivl_caption_unified        3   12.5%  #####
```

The changes to `utils/` are strictly additive — `return_choices` defaults to `False` and the
existing return signature is unchanged — so this is diagnostic instrumentation rather than a
modification of TRACE's method, and the "reuse over rewrite" guarantee above still holds.

---

## Quickstart for MAMI

```bash
# 1. Build the MAMI dataset JSON from the raw TSVs
python MAMI/build_mami_skeleton.py

# 2. Generate captions (InternVL; the misogyny prompt is the default/primary)
python MAMI/mami_cap_gen.py
python MAMI/mami_cap_gen.py --merge          # fold the sidecar into the dataset JSON

#    ...or run two prompt variants at once, one per GPU, in two terminals:
python MAMI/mami_cap_gen.py --prompt all     --gpu 0    # terminal 1: unified caption
python MAMI/mami_cap_gen.py --prompt generic --gpu 1    # terminal 2: generic caption
python MAMI/mami_cap_gen.py --merge                     # ...then merge, once both finish

# 3. Train (roberta is primary; vitl14 / siglip2 also available)
python MAMI/train_mami.py --backbone roberta

#    ...train on an ablation caption set
python MAMI/train_mami.py --backbone roberta --caption-field ivl_caption_unified

# 4. Evaluate a saved checkpoint
python MAMI/mami_eval.py --backbone roberta --checkpoint checkpoints/mami_roberta_trace_best_model.pth

# 5. Modality ablation: one arm, or the whole sweep + results table
python MAMI/train_mami.py --arm image_text
python MAMI/mami_ablation.py --run                  # train every arm, then print the table
python MAMI/mami_ablation.py --format markdown      # collect only, as markdown
```

Each prompt variant writes its **own** sidecar and its own caption field, so two terminals
never contend for the same file — only `--merge` writes the dataset JSON, and it is run once
both jobs have finished. Give each terminal its own device with `--gpu`. The same flags apply as for Memotion
and MMSD (`--subset N --epochs 1`, `--no-resume`, `--wandb`, `--caption-field`).

## Quickstart for Memotion

```bash
# 1. Build the Memotion dataset JSON from the binarized CSVs (carves the val split)
python Memotion/build_memotion_skeleton.py --verify-images

# 2. Generate captions ONCE -- shared by all three tasks
python Memotion/memotion_cap_gen.py                    # InternVL, --prompt all

# 3. Train one task at a time (roberta is the default backbone)
python Memotion/train_memotion.py --task humour
python Memotion/train_memotion.py --task offensive
python Memotion/train_memotion.py --task sarcasm

#    ...or pick another backbone
python Memotion/train_memotion.py --task humour --backbone vitl14
python Memotion/train_memotion.py --task humour --backbone siglip2

# 4. Evaluate a saved checkpoint
python Memotion/memotion_eval.py --task humour --backbone roberta \
    --checkpoint checkpoints/memotion_humour_roberta_trace_best_model.pth

# 5. Modality ablation (per task)
python Memotion/train_memotion.py --task humour --arm image_text
python Memotion/memotion_ablation.py --run --task humour
```

Useful flags: `--subset N --epochs 1` for a smoke test, `--no-resume` to ignore an existing
checkpoint, `--wandb` for logging, and `--caption-field` to train on a task-specialized
caption set.

## Quickstart for MMSD2.0

```bash
# 1. Build the dataset JSON AND extract the 24,635 images out of the parquet shards.
#    Resumable and idempotent -- rerunning skips images already on disk.
python MMSD/build_mmsd_skeleton.py --verify-images

# 2. Generate captions (InternVL; sarcasm prompt is the default/primary)
python MMSD/mmsd_cap_gen.py
python MMSD/mmsd_cap_gen.py --merge          # fold the sidecar into the dataset JSON

# 3. Train (roberta is the default/primary; siglip2 and vitl14 are the secondaries)
python MMSD/train_mmsd.py
python MMSD/train_mmsd.py --backbone siglip2
python MMSD/train_mmsd.py --backbone vitl14

# 4. Evaluate a saved checkpoint
python MMSD/mmsd_eval.py --backbone roberta \
    --checkpoint checkpoints/mmsd_roberta_trace_best_model.pth

# 5. Modality ablation
python MMSD/train_mmsd.py --arm image_text
python MMSD/mmsd_ablation.py --run
```

The same flags apply as for Memotion (`--subset N --epochs 1`, `--no-resume`, `--wandb`,
`--caption-field`). There is no `--task` flag: MMSD2.0 is a single binary task.
