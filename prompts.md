> **Caption length -** Every prompt below caps captions at 40 words. The backbones truncate
> text at different limits — CLIP ViT-L/14 and OpenCLIP XLM-R ViT-H/14 at 77 tokens (a hard
> limit set by the positional embedding table), SigLIP2 at 64 — and these prompts put the
> discriminative content (target, comedic turn, whether the image reinforces or contradicts
> the text) at the *end*, so overrunning truncates exactly the signal the classifier needs.
> Captions measure ~1.22 tokens/word, so 40 words ≈ 49 tokens, fitting the tightest limit
> with headroom.

# Generic Descriptive Prompt 
```
GENERIC_PROMPT = """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels — describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]"""
```

# Task Specific Prompt

## Misogyny
```
MISOGYNY_PROMPT = """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

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
Caption: [Generated caption here]"""
```

## Sarcasm
```
SARCASM_PROMPT = """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If the text and image create a contrast, contradiction, reversal, exaggeration, or other observable incongruity between what the text claims and what the image shows, describe it explicitly
- Describe linguistic cues such as hyperbole, rhetorical questions, exaggerated praise, or obvious understatement when they are explicitly present in the text
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'sarcastic', 'ironic') — describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]"""
```

## Offensiveness 
```
OFFENSIVENESS_PROMPT = """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If the meme directs its text or imagery at a person or group (individual, profession, nationality, appearance, belief), name the target; if no target is identifiable, state that
- If the text or imagery contains insults, profanity, derogatory comparisons, threats, wishes of harm, or negative generalizations toward the identified target, describe those elements and how the image and text reinforce each other, noting whether profanity is used as general emphasis or directed at the target and whether the phrasing is aggressive or neutral; if the tone is benign or self-directed, state that
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'offensive', 'harmless') — describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]"""
```

## Humour
```
HUMOUR_PROMPT = """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

The caption should:
- Describe the main visual elements (people, facial expressions, gestures, objects, setting, and their actions)
- Summarize the text overlay (if short) or explain its meaning concisely
- If the content creates an expectation and then subverts, exaggerates, or twists it, describe the setup and the turn; if the content is presented straight with no comedic turn, state that
- If a comedic device is present (exaggeration, absurdity, wordplay), name it and describe how it operates here; if no such device is present, do not invent one
- State whether the image reinforces, contradicts, exaggerates, or recontextualizes the text, using only observable evidence
- Mention recognizable meme templates when identifiable
- Avoid judgmental labels (e.g., 'funny', 'not funny') — describe the content and mechanism, not a verdict

If a listed signal is absent, do not invent it.
Do not speculate about intent or meaning beyond what is visibly present.

Format the response as:
Caption: [Generated caption here]"""
```

# Caption Enriched Prompt
```
UNIFIED_PROMPT = """Task: Analyze this meme image using the above grounding information and generate a **single caption** suitable for CLIP fine-tuning. Keep the caption to NO MORE THAN 40 words -- it must fit in a 64-token text encoder without being cut off, so be terse and prioritise the mechanism over scene detail.

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
Caption: [Generated caption here]"""
```