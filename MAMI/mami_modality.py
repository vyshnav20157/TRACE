"""Modality / caption-source ablations for MAMI (SemEval-2022 Task 5A, misogyny).

The question this answers: how much of TRACE's MAMI performance comes from the image, how
much from the meme's own OCR text, how much from a *task-specific* generated caption, and how
much from the caption-scoring architecture itself? Each arm below removes or replaces one of
those inputs and leaves everything else -- backbone, splits, losses, schedule, seed --
identical, so the deltas between arms are attributable to the modality change alone.

    ARM                 TEXT STREAM FED TO THE MODEL              CAPTION SCORING
    ------------------  ----------------------------------------  ---------------
    image_only          (none -- a fixed neutral prompt)           off
    image_text          meme OCR `text`                            off
    image_taskcap       `ivl_caption_task` (misogyny prompt)       off
    image_genericcap    `ivl_caption_generic` (generic prompt)     off
    image_unifiedcap    `ivl_caption_unified` (all-task prompt)    off
    trace               [meme text, task, generic, unified]        ON (full TRACE)

`trace` is the full architecture: every text source available for a meme -- its OCR text plus
all three generated captions -- is offered as a candidate, and the caption scorer picks the
best one via Gumbel-Softmax, trained by the relevance loss. It is the reference arm, so every
number in the ablation table comes out of one code path and one command.

The single-caption arms above each hand the model ONE fixed text source; `trace` hands it all
four and lets the scorer choose per meme. So the trace-vs-arm deltas measure the value of
having a choice at all, and the deltas among the caption arms measure prompt specificity with
the choice held out.

MAMI vs. Memotion: one task, so one task caption
------------------------------------------------
Memotion Task B is three binary problems over the same memes, so its task caption is resolved
per-task. MAMI Task A is a single binary problem (misogynous / not), so the task-specific
caption lives in one field, `ivl_caption_task`, written by `mami_cap_gen.py` under its
default (`--prompt misogyny`) setting -- the same arrangement as MMSD2.0.

The three caption arms: task vs. generic vs. unified
---------------------------------------------------
`ivl_caption_unified` -- the `--prompt all` UNIFIED caption -- is deliberately NOT the generic
arm. Its prompt carries the task cues of every dataset in the study, which makes it *more*
task-loaded than the misogyny prompt, not less. The generic arm needs the GENERIC prompt from
prompts.md (plain description, no task cues), which `mami_cap_gen.py --prompt generic` writes
to `ivl_caption_generic`.

So the unified set gets its OWN arm, `image_unifiedcap`, rather than being folded into either
of the others. The three caption arms are identical in every respect except which prompt wrote
the caption the model reads, which is what makes the prompt-specificity comparison a clean
single-variable result:

    image_taskcap     misogyny prompt  -- cues for THIS task only
    image_genericcap  generic prompt   -- no task cues at all
    image_unifiedcap  all-task prompt  -- cues for every task in the study

Each writes its own checkpoint and predictions file, so all three can be trained and compared
without touching one another. Prefer these arms over `--arm image_taskcap --caption-field
ivl_caption_unified`: the override changes what the model reads but NOT where the run is
filed, so it would overwrite the real `image_taskcap` results.

All three caption sets are ALSO candidates in the `trace` arm, which is what makes the
comparison complete: each prompt is measured alone in its own arm, and then all of them
together under the scorer.

Why the single-caption arms disable relevance loss
--------------------------------------------------
With one candidate caption the selection machinery degenerates *correctly* on its own --
`gumbel_softmax` and `softmax` over a length-1 score vector both return exactly 1.0, so the
weighted "selection" is a pass-through of the only caption, and no code in
`utils/loss_functions.py` needs to change. The relevance loss, however, does NOT degenerate
harmlessly: it trains `caption_scorer` to predict the misogyny label from that single fixed
text, which is a second classifier riding alongside the real one rather than the
caption-ranking signal it is meant to be. Leaving it on would mean the single-caption arms
are not "TRACE minus caption selection" but "TRACE plus an extra text-only head", and the
comparison against the `trace` arm would no longer isolate caption scoring. So every arm
except `trace` runs `relevance: False`.

The image_only arm
------------------
CLIP-family backbones have no image-only forward path here: the classifier head consumes a
fused image+text vector, so some text must be supplied. Feeding the empty string is not
neutral -- the tokenizer maps "" to its own BOS/EOS pair, which is still a learnable embedding
the model can key on. Instead every sample gets the SAME fixed placeholder (`NULL_TEXT`), so
the text stream carries exactly zero per-sample information: the text branch contributes one
constant vector for the whole dataset, and any accuracy above chance is attributable to the
image branch. This is the standard way to ablate a modality out of a fusion architecture
without changing the architecture.

Usage
-----
    python MAMI/train_mami.py --arm image_only
    python MAMI/train_mami.py --arm image_text
    python MAMI/train_mami.py --arm image_taskcap
    python MAMI/train_mami.py --arm image_genericcap
    python MAMI/train_mami.py --arm image_unifiedcap
    python MAMI/train_mami.py --arm trace            # default: full TRACE, all captions scored

Each arm writes its own checkpoint and predictions file (see `checkpoint_name` / `preds_name`)
so arms never overwrite each other and can run concurrently on separate GPUs.
"""

# Caption fields are namespaced by captioner (utils/captioner_backends.py), so the
# InternVL and Qwen2.5-VL caption sets coexist in one dataset JSON and one results
# directory. See `resolve_captioner` below for how a run picks its set.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.captioner_backends import FIELD_PREFIX, caption_field

DEFAULT_CAPTIONER = "internvl"

# The constant string used as the text stream in the image_only arm. It is deliberately
# contentful-but-uninformative rather than empty: an empty string still tokenizes to a
# backbone-specific BOS/EOS pair, whereas this is the same for every sample, so the text
# branch is a constant and cannot carry per-sample signal.
NULL_TEXT = "no text available"

# Caption JSON fields, matching the writers in `mami_cap_gen.py` PROMPT_CONFIG.
TASK_CAPTION_FIELD = "ivl_caption_task"      # --prompt misogyny (task-specific, the default)
GENERIC_CAPTION_FIELD = "ivl_caption_generic"  # --prompt generic (plain description)
UNIFIED_CAPTION_FIELD = "ivl_caption_unified"  # --prompt all (union of task cues)

# Caption field -> its role suffix. This is what lets an arm written in InternVL's namespace
# be re-pointed at another captioner's caption set (`captioner_sources`) without duplicating
# every arm per captioner: the role (task / generic / unified) is the stable part, the
# captioner prefix is the variable one.
CAPTION_SUFFIX_BY_FIELD = {
    TASK_CAPTION_FIELD: "task",
    GENERIC_CAPTION_FIELD: "generic",
    UNIFIED_CAPTION_FIELD: "unified",
}

# Arm definitions.
#
#   sources   -- ordered list of per-sample text sources building the caption list.
#                "text" is the meme's OCR text; "null" is the fixed NULL_TEXT constant;
#                anything else is a JSON caption field name.
#   relevance -- whether the relevance loss (and hence caption-scorer training) is on.
#                Only the full-TRACE arm sets this; see the module docstring.
#   label     -- short human-readable name for logs and the results table.
ARMS = {
    "image_only": {
        "sources": ["null"],
        "relevance": False,
        "label": "image only (text stream held constant)",
    },
    "image_text": {
        "sources": ["text"],
        "relevance": False,
        "label": "image + meme OCR text",
    },
    "image_taskcap": {
        "sources": [TASK_CAPTION_FIELD],
        "relevance": False,
        "label": "image + task-specific caption (misogyny prompt)",
    },
    "image_genericcap": {
        "sources": [GENERIC_CAPTION_FIELD],
        "relevance": False,
        "label": "image + generic caption",
    },
    "image_unifiedcap": {
        "sources": [UNIFIED_CAPTION_FIELD],
        "relevance": False,
        "label": "image + unified caption (all-task prompt)",
    },
    "trace": {
        "sources": ["text", TASK_CAPTION_FIELD, GENERIC_CAPTION_FIELD, UNIFIED_CAPTION_FIELD],
        "relevance": True,
        "label": "full TRACE (meme text + all generated captions, caption scoring)",
    },
}

DEFAULT_ARM = "trace"


def default_caption_field(captioner=DEFAULT_CAPTIONER):
    """The dataset's default caption column for `captioner`.

    This is the column a backbone falls back to when no --caption-field is given. It must
    follow the captioner: otherwise a `--captioner qwen` run with no explicit field would
    read InternVL's captions while filing its results under the qwen name, which is the exact
    mislabelling the captioner namespacing exists to prevent.
    """
    return caption_field(captioner, "task")


def get_arm(name):
    """Return the arm spec for `name`, with a helpful error listing the valid arms."""
    if name not in ARMS:
        raise ValueError(f"Unknown arm '{name}'. Choose one of: {', '.join(ARMS)}")
    return ARMS[name]


def owning_arm(caption_field_name):
    """Return the arm that owns `caption_field_name`, for ANY captioner, or None.

    Ownership is a property of the caption's ROLE (task / generic / unified), not of which
    VLM wrote it: `qwen_caption_unified` belongs to `image_unifiedcap` exactly as
    `ivl_caption_unified` does. Resolving it by suffix rather than by a flat table of
    InternVL names means the override guard keeps working the moment a second captioner
    exists -- otherwise `--arm image_taskcap --caption-field qwen_caption_unified` would slip
    past the check and file unified-caption numbers under the task-caption arm.
    """
    for prefix in FIELD_PREFIX.values():
        if not caption_field_name.startswith(prefix + "_"):
            continue
        suffix = caption_field_name[len(prefix) + 1:]
        return {
            "task": "image_taskcap",
            "generic": "image_genericcap",
            "unified": "image_unifiedcap",
        }.get(suffix)
    return None


def check_override(name, caption_field):
    """Reject a --caption-field that would file one arm's results under another arm's name.

    `checkpoint_name` / `preds_name` key on the ARM ALONE, so
    `--arm image_taskcap --caption-field ivl_caption_unified` reads unified captions but
    writes `mami_roberta_image_taskcap_*` -- silently overwriting the real task-caption run
    with numbers from a different prompt. Every caption set that has its own arm must be
    reached through that arm, so the results land in their own files.

    Overrides at a field with no arm of its own (a new or experimental caption set) still work;
    they are the case the override exists for.

    `trace` is exempt: it reads every caption field already, so an override there does not
    relabel one prompt's results as another's -- it narrows the candidate set, which is a
    different experiment and is rejected in `arm_sources` instead.
    """
    if caption_field is None or name == DEFAULT_ARM:
        return
    owner = owning_arm(caption_field)
    if owner is not None and owner != name:
        raise SystemExit(
            f"--caption-field {caption_field} belongs to --arm {owner}, but --arm {name} was "
            f"given. Results are filed by arm name alone, so this would overwrite {name}'s "
            f"checkpoint and predictions with {caption_field} numbers.\n"
            f"    Use: --arm {owner}"
        )


def captioner_sources(sources, captioner):
    """Re-point an arm's caption fields at `captioner`'s caption set.

    An arm names its caption fields in InternVL's namespace (`ivl_caption_*`), because that
    is the primary captioner. The captioner ablation reruns the SAME arms against a second
    caption set written by a different VLM, so the fields are rewritten here rather than
    duplicating every arm per captioner: the arm definitions stay the single source of truth
    for *which role* each caption plays (task / generic / unified), and the captioner decides
    only *whose* captions fill that role.

    "text" and "null" slots are captioner-independent (the meme's own OCR text, and the fixed
    constant) and pass through untouched -- which is exactly what makes the image_text and
    image_only arms shared baselines rather than something each captioner needs its own copy of.
    """
    if captioner == DEFAULT_CAPTIONER:
        return list(sources)
    remapped = []
    for src in sources:
        if src in ("text", "null"):
            remapped.append(src)
            continue
        suffix = CAPTION_SUFFIX_BY_FIELD.get(src)
        if suffix is None:
            # A --caption-field override naming an explicit column: the user asked for that
            # exact column, so honour it verbatim instead of guessing a captioner namespace.
            remapped.append(src)
        else:
            remapped.append(caption_field(captioner, suffix))
    return remapped


def arm_sources(name, caption_field_override=None, captioner=DEFAULT_CAPTIONER):
    """Resolve an arm's text sources, letting --caption-field override the caption slot.

    The single-caption arms name a specific JSON field, but a user may want to point one of
    them at a caption set that has no arm of its own. `caption_field_override` substitutes for
    the caption source while leaving "text"/"null" slots alone, so the override composes with
    every arm instead of being mutually exclusive with it.

    The three caption sets that DO have their own arms (task / generic / unified) must be
    reached through those arms -- see `check_override`, which callers run first.

    `trace` carries MULTIPLE caption slots, so substituting into each of them would build a
    candidate list holding the same caption three times -- the scorer would be ranking
    duplicates, and the run would silently not be TRACE. There is no single slot to override,
    so the override is rejected rather than guessed at.

    `captioner` selects WHOSE captions fill the arm's caption slots; it is applied last so an
    explicit --caption-field still wins over the captioner namespace.
    """
    sources = list(get_arm(name)["sources"])
    if caption_field_override is None:
        return captioner_sources(sources, captioner)

    caption_slots = [src for src in sources if src not in ("text", "null")]
    if len(caption_slots) > 1:
        raise SystemExit(
            f"--caption-field is not supported for --arm {name}: it reads "
            f"{len(caption_slots)} caption fields ({', '.join(caption_slots)}) and there is no "
            f"single slot to override. Use a single-caption arm to test one field."
        )
    return [caption_field_override if src not in ("text", "null") else src for src in sources]


def build_captions(row, sources, fallback="No caption"):
    """Build the ordered caption list for one dataset row under the given text sources.

    Mirrors the list-building the backbone dataset already did inline, but driven by the arm's
    `sources` rather than hardcoded to [text, ivl caption]. Empty / "nan" / "none" entries are
    dropped, exactly as before, and a row left with nothing falls back to a placeholder --
    dropping the row instead would silently alter MAMI's official test split.
    """
    captions = []
    for src in sources:
        if src == "null":
            captions.append(NULL_TEXT)
        else:
            captions.append(str(row.get(src, fallback)))

    captions = [cap for cap in captions
                if cap.strip() and cap.strip().lower() not in ("nan", "none")]
    if not captions:
        captions = [fallback]
    return captions


def loss_config_for(name):
    """Loss configuration for an arm.

    Classification is always on; contrastive stays off across the board (matching the MAMI
    training scripts' existing setting, so the arms differ from the published TRACE runs in
    exactly one respect); relevance follows the arm.
    """
    return {
        "classification": True,
        "contrastive": False,
        "relevance": get_arm(name)["relevance"],
    }


def _smoke_tag(smoke):
    """Filename fragment marking a throwaway smoke run (`--subset`).

    A --subset run trains on a handful of rows and evaluates on a handful more, so its
    metrics are meaningless -- but without this tag it would write to the SAME checkpoint and
    predictions filenames as the real run of that arm and destroy hours of training. Smoke
    runs therefore get their own `_smoke` files, which are safe to delete and impossible to
    confuse with a real result.
    """
    return "_smoke" if smoke else ""


def _captioner_tag(captioner):
    """Filename fragment identifying the captioner, empty for the primary one.

    InternVL runs keep their historical un-tagged filenames, so every checkpoint and
    predictions file already on disk stays valid and the published numbers are not orphaned.
    A second captioner gets its name in the filename instead, which is what stops a Qwen run
    of an arm from overwriting the InternVL run of that same arm -- the two are different
    experiments and must be readable side by side.
    """
    return "" if captioner == DEFAULT_CAPTIONER else f"{captioner}_"


def checkpoint_name(backbone, arm, captioner=DEFAULT_CAPTIONER, smoke=False):
    """Per-(arm, captioner) checkpoint filename, so runs never overwrite one another.

    Every arm, `trace` included, is suffixed with its own name. `trace` used to keep the
    un-suffixed `mami_<backbone>_best_model.pth` for back-compat with the pre-ablation
    scripts, but it now scores all four text sources rather than two, so a run under the old
    name would no longer mean what the old checkpoints meant. Suffixing it makes the change
    visible in the filename instead of silently redefining an existing one.

    The captioner tag is prepended for any non-primary captioner; see `_captioner_tag`.
    """
    return f"mami_{_captioner_tag(captioner)}{backbone}_{arm}{_smoke_tag(smoke)}_best_model.pth"


def preds_name(backbone, arm, captioner=DEFAULT_CAPTIONER, smoke=False):
    """Per-(arm, captioner) test-predictions filename (same rule as `checkpoint_name`)."""
    return f"mami_{_captioner_tag(captioner)}{backbone}_{arm}{_smoke_tag(smoke)}_preds.json"


def resolve_arm(args, data, data_path):
    """Print the arm header, validate its caption fields against `data`, return its sources.

    Raises SystemExit on a caption field that was never generated or merged -- training on a
    column of empty strings would otherwise silently degrade e.g. `image_genericcap` into an
    image-only run and quietly invalidate the ablation table -- and on a --caption-field that
    would overwrite another arm's results (see `check_override`).
    """
    check_override(args.arm, args.caption_field_override)
    # `captioner` may be absent on an older caller; default to the primary one so nothing
    # that predates the captioner ablation changes behaviour.
    captioner = getattr(args, "captioner", DEFAULT_CAPTIONER)
    print(f"=== arm: {describe_arm(args.arm, args.caption_field_override, captioner)} ===")
    sources = arm_sources(args.arm, args.caption_field_override, captioner)

    for src in sources:
        if src in ("text", "null"):
            continue
        if src not in data.columns:
            raise SystemExit(
                f"Caption field '{src}' is not in {data_path}. Generate it first, e.g.\n"
                f"    python MAMI/mami_cap_gen.py --prompt generic --captioner {captioner}\n"
                f"    python MAMI/mami_cap_gen.py --merge --captioner {captioner}"
            )
        col = data[src].astype(str).str.strip()
        filled = int((~col.isin(["", "nan", "None"])).sum())
        print(f"  caption field '{src}': {filled}/{len(data)} rows populated")
        if filled == 0:
            raise SystemExit(
                f"Caption field '{src}' exists but is empty in {data_path}. "
                f"Run the matching mami_cap_gen.py prompt with --captioner {captioner}, "
                f"then --merge --captioner {captioner}."
            )

    return sources


def describe_arm(name, caption_field=None, captioner=DEFAULT_CAPTIONER):
    """One-line description of an arm for log headers.

    The captioner is named explicitly: two runs of the same arm can now differ ONLY by which
    VLM wrote the captions, so a header without it would make the two indistinguishable in a
    log.
    """
    spec = get_arm(name)
    sources = arm_sources(name, caption_field, captioner)
    rendered = ", ".join("<constant>" if s == "null" else s for s in sources)
    scoring = "on" if spec["relevance"] else "off"
    return (f"{name} [captioner: {captioner}] -- {spec['label']} | "
            f"text sources: [{rendered}] | caption scoring: {scoring}")
