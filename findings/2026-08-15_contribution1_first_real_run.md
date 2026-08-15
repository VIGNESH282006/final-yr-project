# Contribution #1 (confidence-aware signals): first real-model run

**Date:** 2026-08-15
**Setup:** `notebooks/02_real_pipeline.ipynb` on Colab (T4 GPU), single shared
Qwen2.5-7B-Instruct (4-bit) for all three agent roles, local E5-base retrieval, same 8
HotpotQA questions as the baseline run. Code includes contribution #1's changes
(`pyrag/utils.py::extract_confidence_tag`, `pyrag/tools.py` confidence-tag prompts,
`pyrag/runner.py` confidence-aware retry trigger).

## Result: 62.5% EM (5/8), up from 50.0% (4/8) baseline

| # | Question | Predicted | Gold | Baseline verdict | New verdict |
|---|----------|-----------|------|----|----|
| 1 | Derrickson/Wood nationality | yes | yes | Correct | Correct |
| 2 | Corliss Archer government position | "United States ambassador to Ghana, ... Chief of Protocol ..." | Chief of Protocol | Wrong (unknown) | **EM-wrong but factually correct & complete** -- verbose phrasing doesn't match gold string |
| 3 | Companion-books YA series | Animorphs | Animorphs | Wrong (hallucinated) | **Fixed** |
| 4 | Laleli Mosque / Esma Sultan Mansion | No | no | Correct | Correct |
| 5 | Big Stone Gap director's NY city | New York City | Greenwich Village, New York City | Correct | Regressed (right city, lost specificity) |
| 6 | 2014 S/S boy group founder | YG Entertainment | YG Entertainment | Correct | Correct |
| 7 | Stage name Aladin consultant | Eenasul Fateh | Eenasul Fateh | Wrong (circular "Aladin") | **Fixed** |
| 8 | Lewiston Maineiacs arena capacity | 4,000 people | 3,677 seated | Close, wrong | Close, wrong (unchanged) |

Net: 2 questions fixed (Q3, Q7), 1 regressed (Q5), 1 unchanged-wrong (Q8), 1 arguably-improved-but-still-EM-wrong (Q2), 3 unchanged-correct (Q1, Q4, Q6).

## IMPORTANT correction to the design assumption: the real model never emits `<confidence>` at all

Checked every `raw:` block in the actual Colab output (not simulated) -- **not one
`answer()` call anywhere in this run contains a `<confidence>` tag**, despite both
`ANSWER_SYSTEM_PROMPT_WITH_DOCS` and `ANSWER_SYSTEM_PROMPT_NO_DOCS` explicitly instructing
the model to output one. Verified locally: `extract_confidence_tag()` on Q1 Step 2's exact
raw text (no tag present) returns `"low"` -- the safe fallback, not a genuine signal.

**Consequence: every single answer this run was logged with `confidence="low"`,
unconditionally**, because the parser never found a real tag to read. This means the
confidence-aware retry in `pyrag/runner.py` (`_answer_step_is_insufficient`,
`final_answer_is_insufficient`) fired far more liberally than designed -- effectively
"always allow a retry with boosted top-k if boost hasn't already happened", rather than
"retry specifically when the model is uncertain." Q3 and Q7 likely improved because they
got a free boosted-retrieval retry pass, similar in effect to just raising the default
`topk` or always doing one retry -- NOT because the model's own self-assessed confidence
discriminated between reliable and unreliable answers, which was the actual mechanism
contribution #1 was designed to test.

This is still a genuine, real result (62.5% > 50.0%, and Q7 -- the flagship reproduction
case -- did get fixed), but the causal story needs to be accurate in the report: **this run
demonstrates that "always retry once with more documents" helps**, not yet that
**"the model's structured self-reported confidence reliably identifies wrong answers"**,
because we don't yet have a case where the model reported anything other than the
default. The mechanism as designed is UNTESTED until the model actually emits the tag.

## Why might Qwen2.5-7B-Instruct not be emitting the tag?

Plausible reasons, not yet confirmed:
1. **Prompt is being followed inconsistently** -- the instruction is fairly deep in a long
   system prompt; the model may be prioritizing the `<answer>` format (reinforced by the
   one-shot example) over the newer `<confidence>` instruction.
2. **The few-shot example matters more than the instruction text.** Note
   `ANSWER_SYSTEM_PROMPT_WITH_DOCS`'s example block DOES show `<confidence> high
   </confidence>` after `<answer>`, so the format is demonstrated -- but the model may
   still be dropping it under certain conditions (e.g. shorter answers, different
   `<redacted_thinking>` phrasing patterns).
3. **max_tokens truncation** -- `LocalLLM`'s default `max_new_tokens=1024` should be ample
   for these short answers, but worth double-checking generation isn't being cut off
   before the confidence tag would appear (unlikely given `<answer>` always closes first,
   but should verify with a raw token count check if the fix below doesn't work).

## Next step (before doing anything else with contribution #1)

Need to make the model reliably emit the tag, then re-run, before this contribution can be
called validated. Options to try, roughly in order of first thing to attempt:
1. Move the confidence tag requirement earlier/more prominently in the system prompt
   (currently after several other rule blocks -- may be getting deprioritized).
2. Add 1-2 additional few-shot examples showing a LOW-confidence case specifically (the
   current example only shows a high-confidence case), so the model has seen the pattern
   it's actually supposed to use most often for this test's harder questions.
3. Consider a stricter output-format instruction (e.g. explicitly "you MUST include all
   three tags in this exact order, this is machine-parsed, do not omit any tag").
4. If the tag still doesn't appear reliably after prompt changes, that itself becomes a
   documented, reportable limitation of few-shot prompting a 7B instruct model for
   structured multi-field output without fine-tuning -- which is a legitimate, honest
   finding for the report (ties back to why the paper's own RL variant exists: prompting
   alone has real limits on smaller open models).
