"""Modality / caption-source ablations for Memotion 1.0 Task B.

The question this answers: how much of TRACE's Memotion performance comes from the image, how
much from the meme's own OCR text, how much from a *task-specific* generated caption, and how
much from the caption-scoring architecture itself? Each arm below removes or replaces one of
those inputs and leaves everything else -- backbone, splits, losses, schedule, seed --
identical, so the deltas between arms are attributable to the modality change alone.

    ARM                 TEXT STREAM FED TO THE MODEL              CAPTION SCORING
    ------------------  ----------------------------------------  ---------------
    image_only          (none -- a fixed neutral prompt)           off
    image_text          meme OCR `text`                            off
    image_taskcap       `ivl_caption_<task>` (task prompt)         off
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
the choice held out. Note the task caption stays per-task inside `trace` too: under
`--task humour` the candidates are [text, ivl_caption_humour, generic, unified], so the arm
still means "the caption written for the task being classified, plus the task-agnostic ones".

Memotion vs. MMSD: the task caption is PER-TASK
-----------------------------------------------
MMSD2.0 is one task, so its task-specific caption lives in one field. Memotion Task B is
three independent binary problems over the SAME memes, and `memotion_cap_gen.py` writes a
separate caption set per task (`ivl_caption_humour`, `ivl_caption_offensive`,
`ivl_caption_sarcasm`). So every arm here is resolved *against a task*: running
`--task humour --arm image_taskcap` reads `ivl_caption_humour`, and the same arm under
`--task offensive` reads `ivl_caption_offensive`. That is what makes the arm mean the same
thing ("the caption written for the task being classified") across all three tasks.

The three caption arms: task vs. generic vs. unified
---------------------------------------------------
`ivl_caption_unified` -- the `--prompt all` UNIFIED caption -- is deliberately NOT the generic
arm. Its prompt is the union of all three tasks' cues (humour + offence + sarcasm), which
makes it *more* task-loaded than any single task prompt, not less. The generic arm needs the
GENERIC prompt from prompts.md (plain description, no task cues), which
`memotion_cap_gen.py --prompt generic` writes to `ivl_caption_generic`.

So the unified set gets its OWN arm, `image_unifiedcap`, rather than being folded into either
of the others. The three caption arms are identical in every respect except which prompt wrote
the caption the model reads, which is what makes the prompt-specificity comparison a clean
single-variable result:

    image_taskcap     task prompt      -- cues for THIS task only
    image_genericcap  generic prompt   -- no task cues at all
    image_unifiedcap  all-task prompt  -- cues for every task at once

Note what `image_unifiedcap` is and is not, given Memotion trains one binary classifier per
task. The unified CAPTION is the same text for every task -- one column, written once -- but
the MODEL is still per-task, so the arm is trained separately under `--task humour` and
`--task offensive` and files its results per (task, arm) like every other arm. The comparison
it supports is per-task: for humour, does a caption carrying all three tasks' cues beat one
written for humour alone (`image_taskcap`) or one with no cues at all (`image_genericcap`)?

Prefer these arms over `--arm image_taskcap --caption-field ivl_caption_unified`: the override
changes what the model reads but NOT where the run is filed, so it would overwrite the real
`image_taskcap` results for that task.

Why the single-caption arms disable relevance loss
--------------------------------------------------
With one candidate caption the selection machinery degenerates *correctly* on its own --
`gumbel_softmax` and `softmax` over a length-1 score vector both return exactly 1.0, so the
weighted "selection" is a pass-through of the only caption, and no code in
`utils/loss_functions.py` needs to change. The relevance loss, however, does NOT degenerate
harmlessly: it trains `caption_scorer` to predict the task label from that single fixed
text, which is a second classifier riding alongside the real one rather than the
caption-ranking signal it is meant to be. Leaving it on would mean the single-caption arms
are not "TRACE minus caption selection" but "TRACE plus an extra text-only head", and the
comparison against the `trace` arm would no longer isolate caption scoring. So every arm
except `trace` runs `relevance: False`.

The image_only arm
------------------
CLIP-family backbones have no image-only forward path here: the classifier head consumes a
fused image+text vector, so some text must be supplied. Feeding the empty string is not
neutral -- the tokenizer maps "" to its own BOS/EOS pair, which is still a learnable
embedding the model can key on. Instead every sample gets the SAME fixed placeholder
(`NULL_TEXT`), so the text stream carries exactly zero per-sample information: the text
branch contributes one constant vector for the whole dataset, and any accuracy above chance
is attributable to the image branch. This is the standard way to ablate a modality out of a
fusion architecture without changing the architecture.

Usage
-----
    python Memotion/train_memotion.py --task humour --arm image_only
    python Memotion/train_memotion.py --task humour --arm image_text
    python Memotion/train_memotion.py --task humour --arm image_taskcap
    python Memotion/train_memotion.py --task humour --arm image_genericcap
    python Memotion/train_memotion.py --task humour --arm image_unifiedcap
    python Memotion/train_memotion.py --task humour --arm trace      # full TRACE, all captions

The caption arms are trained once per task, e.g. the unified arm for both single-task models:

    python Memotion/train_memotion.py --task humour    --arm image_unifiedcap
    python Memotion/train_memotion.py --task offensive --arm image_unifiedcap

Each (task, arm) pair writes its own checkpoint and predictions file (see `checkpoint_name` /
`preds_name`) so runs never overwrite each other and can run concurrently on separate GPUs.
"""

from memotion_common import TASKS, task_label_field  # noqa: F401  (re-export for callers)

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

# Caption JSON fields, matching the writers in `memotion_cap_gen.py` PROMPT_CONFIG.
UNIFIED_CAPTION_FIELD = "ivl_caption_unified"    # --prompt all (union of all task cues)
GENERIC_CAPTION_FIELD = "ivl_caption_generic"   # --prompt generic (plain description)

# Per-task caption fields (--prompt humour|offensive|sarcasm). Kept as an explicit map rather
# than an f-string so a typo'd task fails loudly here instead of resolving to a column that
# silently does not exist.
TASK_CAPTION_FIELDS = {
    "humour": "ivl_caption_humour",
    "offensive": "ivl_caption_offensive",
    "sarcasm": "ivl_caption_sarcasm",
}

# Caption field -> its role suffix, so an arm written in InternVL's namespace can be
# re-pointed at another captioner's caption set (`caption_field(captioner, suffix)`).
# The per-task caption fields need no entry: their suffix IS the task name, which is how
# TASKCAP resolves above.
CAPTION_SUFFIX_BY_FIELD = {
    UNIFIED_CAPTION_FIELD: "unified",
    GENERIC_CAPTION_FIELD: "generic",
}

# Sentinel used inside `ARMS` for "the task-specific caption field". It is resolved against
# the run's --task by `arm_sources`, because unlike MMSD there is no single task caption.
TASKCAP = "<taskcap>"

# Arm definitions.
#
#   sources   -- ordered list of per-sample text sources building the caption list.
#                "text" is the meme's OCR text; "null" is the fixed NULL_TEXT constant;
#                TASKCAP resolves to the current task's caption field; anything else is a
#                literal JSON caption field name.
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
        "sources": [TASKCAP],
        "relevance": False,
        "label": "image + task-specific caption",
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
        "sources": ["text", TASKCAP, GENERIC_CAPTION_FIELD, UNIFIED_CAPTION_FIELD],
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
    return caption_field(captioner, "unified")


def get_arm(name):
    """Return the arm spec for `name`, with a helpful error listing the valid arms."""
    if name not in ARMS:
        raise ValueError(f"Unknown arm '{name}'. Choose one of: {', '.join(ARMS)}")
    return ARMS[name]


def task_caption_field(task, captioner=DEFAULT_CAPTIONER):
    """Return the caption field written by `<ds>_cap_gen.py --prompt <task> --captioner <c>`.

    The task name doubles as the field's role suffix, so the captioner namespace composes
    directly: ('humour', 'qwen') -> 'qwen_caption_humour'. The membership check is what keeps
    a typo'd task loud -- without it an unknown task would quietly resolve to a column that
    does not exist and train as if the caption were empty.
    """
    if task not in TASK_CAPTION_FIELDS:
        raise ValueError(
            f"No task caption field for '{task}'. Known: {', '.join(TASK_CAPTION_FIELDS)}"
        )
    if captioner == DEFAULT_CAPTIONER:
        return TASK_CAPTION_FIELDS[task]
    return caption_field(captioner, task)


def owning_arm(caption_field, task):
    """Return the arm that owns `caption_field` for `task`, or None if no arm owns it.

    An arm "owns" a caption set when its checkpoint and predictions files are understood to
    hold that set's results. `image_taskcap` owns whichever per-task field the CURRENT task
    maps to -- under `--task humour` that is `ivl_caption_humour` -- which is why ownership is
    resolved against the task rather than from a flat table.

    Ownership is a property of the caption's ROLE, not of which VLM wrote it, so a
    `qwen_caption_*` field is owned by the same arm as its `ivl_caption_*` counterpart.
    Resolving by suffix keeps the override guard working across captioners: otherwise
    `--arm image_taskcap --caption-field qwen_caption_unified` would slip past the check and
    file unified-caption numbers under the task-caption arm.
    """
    for prefix in FIELD_PREFIX.values():
        if not caption_field.startswith(prefix + "_"):
            continue
        suffix = caption_field[len(prefix) + 1:]
        if suffix == "generic":
            return "image_genericcap"
        if suffix == "unified":
            return "image_unifiedcap"
        if suffix == task:
            return "image_taskcap"
        return None
    return None


def check_override(name, task, caption_field):
    """Reject a --caption-field that would file one arm's results under another arm's name.

    `checkpoint_name` / `preds_name` key on (task, arm) ALONE, so
    `--task humour --arm image_taskcap --caption-field ivl_caption_unified` reads unified
    captions but writes `memotion_humour_roberta_image_taskcap_*` -- silently overwriting the
    real task-caption run with numbers from a different prompt. Every caption set that has its
    own arm must be reached through that arm, so results land in their own files.

    Two overrides stay legal, because neither can overwrite another arm's results:
      * a field with no arm of its own (a new or experimental caption set);
      * ANOTHER task's caption field, e.g. scoring the humour model against
        `ivl_caption_offensive`, which is a cross-task probe rather than a relabelled arm.

    `trace` is exempt: it reads every caption field already, so an override there does not
    relabel one prompt's results as another's -- it narrows the candidate set, which is a
    different experiment and is rejected in `arm_sources` instead.
    """
    if caption_field is None or name == DEFAULT_ARM:
        return
    owner = owning_arm(caption_field, task)
    if owner is not None and owner != name:
        raise SystemExit(
            f"--caption-field {caption_field} belongs to --arm {owner} (task '{task}'), but "
            f"--arm {name} was given. Results are filed by (task, arm) alone, so this would "
            f"overwrite {name}'s checkpoint and predictions with {caption_field} numbers.\n"
            f"    Use: --arm {owner}"
        )


def arm_sources(name, task, caption_field_override=None, captioner=DEFAULT_CAPTIONER):
    """Resolve an arm's text sources for `task`, honouring a --caption-field override.

    The TASKCAP sentinel becomes the task's own caption field, so `image_taskcap` and `trace`
    mean "the caption written for the task being classified" without the caller having to
    thread field names around.

    `caption_field_override` substitutes for the caption slot while leaving "text"/"null"
    alone, so an arm can be pointed at a caption set that has no arm of its own, and the
    override composes with every arm instead of being mutually exclusive with it.

    `captioner` selects WHOSE captions fill the arm's caption slots -- the captioner ablation
    reruns the same arms against a second VLM's caption set. It is applied when the slot is
    resolved, so the arm definitions stay the single source of truth for which ROLE each
    caption plays (task / generic / unified) and the captioner decides only whose captions
    fill that role. An explicit --caption-field still wins over the captioner namespace, and
    "text"/"null" slots are captioner-independent by construction.

    The three caption sets that DO have their own arms (task / generic / unified) must be
    reached through those arms -- see `check_override`, which callers run first.

    `trace` carries MULTIPLE caption slots, so substituting into each of them would build a
    candidate list holding the same caption three times -- the scorer would be ranking
    duplicates, and the run would silently not be TRACE. There is no single slot to override,
    so the override is rejected rather than guessed at.
    """
    spec_sources = get_arm(name)["sources"]
    if caption_field_override is not None:
        caption_slots = [src for src in spec_sources if src not in ("text", "null")]
        if len(caption_slots) > 1:
            rendered = ", ".join(
                task_caption_field(task) if s == TASKCAP else s for s in caption_slots
            )
            raise SystemExit(
                f"--caption-field is not supported for --arm {name}: it reads "
                f"{len(caption_slots)} caption fields ({rendered}) and there is no single slot "
                f"to override. Use a single-caption arm to test one field."
            )

    resolved = []
    for src in spec_sources:
        if src in ("text", "null"):
            resolved.append(src)
        elif caption_field_override is not None:
            resolved.append(caption_field_override)
        elif src == TASKCAP:
            resolved.append(task_caption_field(task, captioner))
        else:
            resolved.append(caption_field(captioner, CAPTION_SUFFIX_BY_FIELD[src]))
    return resolved


def build_captions(row, sources, fallback="No caption"):
    """Build the ordered caption list for one dataset row under the given text sources.

    Mirrors the list-building the backbone dataset already did inline, but driven by the arm's
    `sources` rather than hardcoded to [text, ivl caption]. Empty / "nan" entries are dropped,
    exactly as before, and a row left with nothing falls back to a placeholder -- dropping the
    row instead would silently alter the official Memotion test split (35 memes carry no
    usable OCR text).
    """
    captions = []
    for src in sources:
        if src == "null":
            captions.append(NULL_TEXT)
        else:
            captions.append(str(row.get(src, fallback)))

    captions = [cap for cap in captions if cap.strip() and cap.strip().lower() != "nan"]
    if not captions:
        captions = [fallback]
    return captions


def loss_config_for(name):
    """Loss configuration for an arm.

    Classification is always on; contrastive stays off across the board (matching the Memotion
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


def checkpoint_name(task, backbone, arm, captioner=DEFAULT_CAPTIONER, smoke=False):
    """Per-(task, arm, captioner) checkpoint filename, so runs never overwrite one another.

    Every arm, `trace` included, is suffixed with its own name. `trace` used to keep the
    un-suffixed `memotion_<task>_<backbone>_best_model.pth` for back-compat with the
    pre-ablation scripts, but it now scores all four text sources rather than two, so a run
    under the old name would no longer mean what the old checkpoints meant. Suffixing it makes
    the change visible in the filename instead of silently redefining an existing one.

    The captioner tag is prepended for any non-primary captioner; see `_captioner_tag`.
    """
    return f"memotion_{task}_{_captioner_tag(captioner)}{backbone}_{arm}{_smoke_tag(smoke)}_best_model.pth"


def preds_name(task, backbone, arm, captioner=DEFAULT_CAPTIONER, smoke=False):
    """Per-(task, arm, captioner) test-predictions filename (same rule as `checkpoint_name`)."""
    return f"memotion_{task}_{_captioner_tag(captioner)}{backbone}_{arm}{_smoke_tag(smoke)}_preds.json"


def resolve_arm(args, data, data_path):
    """Print the arm header, validate its caption fields against `data`, return its sources.

    Raises SystemExit on a caption field that was never generated or merged -- training on a
    column of empty strings would otherwise silently degrade e.g. `image_genericcap` into an
    image-only run and quietly invalidate the ablation table. Also rejects a --caption-field
    that would overwrite another arm's results (see `check_override`).
    """
    check_override(args.arm, args.task, args.caption_field_override)
    # `captioner` may be absent on an older caller; default to the primary one so nothing
    # that predates the captioner ablation changes behaviour.
    captioner = getattr(args, "captioner", DEFAULT_CAPTIONER)
    print(f"=== arm: {describe_arm(args.arm, args.task, args.caption_field_override, captioner)} ===")
    sources = arm_sources(args.arm, args.task, args.caption_field_override, captioner)

    for src in sources:
        if src in ("text", "null"):
            continue
        if src not in data.columns:
            raise SystemExit(
                f"Caption field '{src}' is not in {data_path}. Generate it first, e.g.\n"
                f"    python Memotion/memotion_cap_gen.py --prompt generic --captioner {captioner}\n"
                f"    python Memotion/memotion_cap_gen.py --merge --captioner {captioner}"
            )
        col = data[src].astype(str).str.strip()
        filled = int((~col.isin(["", "nan", "None"])).sum())
        print(f"  caption field '{src}': {filled}/{len(data)} rows populated")
        if filled == 0:
            raise SystemExit(
                f"Caption field '{src}' exists but is empty in {data_path}. "
                f"Run the matching memotion_cap_gen.py prompt with --captioner {captioner}, "
                f"then --merge --captioner {captioner}."
            )

    return sources


def describe_arm(name, task, caption_field=None, captioner=DEFAULT_CAPTIONER):
    """One-line description of an arm for log headers.

    The captioner is named explicitly: two runs of the same arm can now differ ONLY by which
    VLM wrote the captions, so a header without it would make the two indistinguishable in a
    log.
    """
    spec = get_arm(name)
    sources = arm_sources(name, task, caption_field, captioner)
    rendered = ", ".join("<constant>" if s == "null" else s for s in sources)
    scoring = "on" if spec["relevance"] else "off"
    return (f"{name} [captioner: {captioner}] -- {spec['label']} | "
            f"text sources: [{rendered}] | caption scoring: {scoring}")
