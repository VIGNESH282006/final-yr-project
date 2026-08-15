# Contribution #1 (confidence-aware signals): VALIDATED real-model run

**Date:** 2026-08-15 (session 5, final run of the day)
**Status: Contribution #1's core mechanism is now confirmed working on real models.**
This is the first run where we directly verified (not inferred) that confidence values
genuinely vary per-answer, using the fixed notebook that force-cleans the Colab clone and
prints the running commit hash for verification (see "How this run differs" below).

## Result: 75.0% EM (6/8), up from 50.0% baseline (unmodified official PyRAG)

| # | Question | Predicted | Gold | Verdict |
|---|----------|-----------|------|---------|
| 1 | Derrickson/Wood nationality | yes | yes | Correct |
| 2 | Corliss Archer government position | "United States ambassador to Ghana, ... Chief of Protocol ..." | Chief of Protocol | EM-wrong, but factually complete and correct -- verbose phrasing vs. terse gold string |
| 3 | Companion-books YA series | Animorphs | Animorphs | **Correct** (fixed vs. baseline's hallucinated "The Lost Colony series") |
| 4 | Laleli Mosque / Esma Sultan Mansion | no | no | Correct |
| 5 | Big Stone Gap director's NY city | Greenwich Village, New York City | Greenwich Village, New York City | Correct |
| 6 | 2014 S/S boy group founder | YG Entertainment | YG Entertainment | Correct |
| 7 | Stage name Aladin consultant | Eenasul Fateh | Eenasul Fateh | **Correct** (fixed vs. baseline's circular "Aladin") |
| 8 | Lewiston Maineiacs arena capacity | 4,000 | 3,677 seated | Close, wrong (unrelated extraction/precision nuance, not a target of any contribution) |

Both originally-flagged before/after cases (Q3, Q7) are fixed. Only Q2 and Q8 remain
EM-wrong, and neither is a framework-logic bug: Q2 found genuinely correct, complete
information but phrased more verbosely than HotpotQA's terse gold answer; Q8 is a
number-extraction precision nuance (the source doc literally contains both "4,000" and
"3,677 seated" and the model picked the rounded one).

## Confidence values DIRECTLY observed (not inferred) across all 21 answer() calls

**19 answers: `confidence: high`. Exactly 2 answers: `confidence: low`** (Q3 step 4 +
step 5 final synthesis; Q7 step 5 final synthesis). This is genuine, non-constant
variation -- the mechanism is discriminating, not defaulting to one value everywhere
(which is what happened in all three earlier failed attempts this session).

## THE key trace: Question 7's final synthesis step

```
[Step 4] answer('Who is a consultant that helped organizations improve their performance?')
    raw: <redacted_thinking>Doc [7] mentions Mick Batyske ... advisor and consultant for
    Localeur ...</redacted_thinking>
    <answer> Mick Batyske </answer>
    -> Mick Batyske  [confidence: high]      <- WRONG person, but reasonably read from evidence, so high is defensible here

[Step 5] answer('Given: Eenasul Fateh is known by the stage name Aladin, Mick Batyske is
    a consultant ... Answer the question: ...')
    raw: <redacted_thinking> ... Eenasul Fateh fits both criteria.</redacted_thinking>
    <answer> Eenasul Fateh </answer>
    -> Eenasul Fateh  [confidence: low]      <- CORRECT answer, but model itself is not
                                                 confident about it -- exactly the
                                                 discriminating signal we wanted
```

This is the clearest possible demonstration of the mechanism: the final answer text
("Eenasul Fateh") is correct and contains no sentinel word, yet `rate_confidence()`
independently flagged it as low-confidence -- because the reasoning chain leading to it
(conflating two different people from two separate lookups) is genuinely shaky, even
though it happened to land on the right name this time. The OLD sentinel-string-only
check would have had no way to flag this at all. This is the single best before/after
trace to use in the project report and viva.

## How this run differs from the 3 earlier (unreliable) runs this session

Three earlier attempts this session ALL showed zero confidence variation / zero
`[confidence: ...]` output at all. Root cause, fully diagnosed: Colab's `PyRAG` folder
persisted across "Restart session" in the user's environment, so `!git clone` failed
SILENTLY inside a shell-magic cell (no visible cell error) and `%cd PyRAG` entered the
STALE old folder -- meaning the notebook kept running old code with zero indication
anything was wrong, across multiple "fixes" that were never actually tested.

**Fixed** in `notebooks/02_real_pipeline.ipynb`:
1. Clone cell now does `shutil.rmtree("PyRAG", ignore_errors=True)` before cloning --
   guarantees a fresh checkout every time regardless of leftover state.
2. A new cell right after clone prints `git log -1 --oneline` so the running commit is
   directly visible and checkable against GitHub before trusting any result.

**This run's sanity-check cell printed `f82eea0` and the user visually confirmed it
matched GitHub's latest commit before running the rest** -- this is why THIS run's
results (and the confidence values in it) are trustworthy where the previous three
were not.

## What contribution #1 actually is, as shipped (for the report)

Two architectural pieces, both in `pyrag/`:
1. **`pyrag/tools.py::rate_confidence()`** -- after `answer()` generates its answer, a
   SEPARATE, small follow-up LLM call rates how well the evidence supports that answer,
   replying with exactly one word (high/medium/low). This exists because two rounds of
   trying to get the model to emit an in-response `<confidence>` tag (alongside
   `<redacted_thinking>` and `<answer>` in one generation) completely failed across 16+
   real answer calls -- a confirmed, reproducible limit of single-pass structured output
   on a 7B instruct model, not something more forceful prompt wording could fix. This
   negative result is itself worth a paragraph in the report.
2. **`pyrag/runner.py`** -- `_answer_step_is_insufficient()` and the final-answer check
   in `run()` now branch on `entry["confidence"] != "high"` (in addition to keeping the
   original sentinel-string check as a safety net), instead of ONLY string-matching
   "unknown"/"cannot answer" in the answer text. This closes the exact blind spot shown
   in Q7 above: a confidently-WORDED final answer that the model itself, when asked
   directly, does not actually trust.

**Cost:** confidence rating doubles LLM calls per `answer()` (2 instead of 1) -- roughly
doubles generation time/compute per question. A deliberate, documented trade-off.

## Honest gaps still open (do NOT claim these are resolved)

1. **Sample size is tiny (n=8).** 75.0% vs 50.0% on 8 questions is suggestive, not
   statistically strong. A proper before/after comparison needs a larger sample (30-50+
   questions minimum) across more than one benchmark before this can be reported as a
   real accuracy improvement rather than an anecdote.
2. **No controlled ablation yet.** We changed the mechanism AND happened to also change
   which questions got retried (since confidence now varies) -- we have NOT yet isolated
   "how much of the improvement is from confidence-awareness specifically" vs. "some
   questions just got a lucky retry regardless of trigger reason." Contribution #1's
   real validation needs a same-question, same-model, same-seed comparison: baseline
   trigger logic vs. confidence-aware trigger logic, holding everything else constant.
3. **`rate_confidence()`'s own reliability is unverified beyond this one run.** We saw
   good discrimination on Q3 and Q7 here, but haven't checked: does it ever wrongly rate
   a GOOD answer as low (costing an unnecessary retry/compute), or a BAD answer as high
   (missing a case it should have caught)? Need more runs, ideally with intentionally
   planted bad-evidence cases, to characterize its false-positive/false-negative rate.
4. **Q2 and Q8 are NOT contribution #1's job.** Q2 needs answer-format/verbosity
   handling (arguably a scoring-normalization improvement, or a Plan Agent instruction to
   keep the final answer terse); Q8 is retrieval/extraction precision. Neither should be
   folded into contribution #1's story.
5. **This was all done with a SINGLE shared Qwen2.5-7B-Instruct model** for all three
   agent roles (Plan/Decompose/Answer), NOT the paper's two-model setup, due to a GPU-OOM
   constraint on the free Colab T4. This is a documented, reasonable substitution (the
   paper's own README allows it) but should be stated plainly as a limitation of this
   experimental setup, not silently assumed equivalent to the two-model configuration.

## Recommended next steps (in order)

1. **Scale up the sample size.** Before doing anything else, re-run on 30-50 HotpotQA
   questions (not just 8) to get an EM number worth citing. `notebooks/02_real_pipeline.ipynb`
   already supports this -- just raise `N_SAMPLES`. Expect proportionally longer runtime
   (roughly 2x per question already from the confidence-rating call, times however many
   more questions).
2. **Build a controlled ablation.** Run the SAME sample twice: once with
   `_answer_step_is_insufficient` forced to use ONLY the old sentinel-string check
   (ignore confidence), once with the full confidence-aware check as currently shipped.
   Compare EM directly -- this isolates contribution #1's actual causal effect from
   noise/luck.
3. Once (1) and (2) are done and the improvement holds up at scale, write the final
   report section for contribution #1 using Q7's trace above as the flagship example.
4. Only after that: move to contribution #2 (anti under-decomposition checker).

## Where everything lives
- Code: `pyrag/tools.py` (rate_confidence, CONFIDENCE_RATING_SYSTEM_PROMPT),
  `pyrag/runner.py` (_answer_step_is_insufficient, confidence-aware final check),
  `pyrag/utils.py` (extract_confidence_word).
- Notebook: `notebooks/02_real_pipeline.ipynb` (has the clean-clone fix + commit-hash
  sanity check -- ALWAYS verify the printed commit matches GitHub before trusting a run).
- Local regression tests (no GPU needed): `notebooks/01_smoke_test.py`,
  `notebooks/04_test_confidence_fix.py` -- both pass as of this commit.
- EM scoring: `pyrag/eval.py`, `notebooks/03_score_baseline.py` (hardcoded to the
  original 8-question baseline; will need updating or generalizing for a larger sample).
- Full session history: `PROGRESS.md` at the repo root -- read this top to bottom for
  the full session-by-session log, this file is the summary/handoff of just today's
  final validated result.
- Repo: https://github.com/VIGNESH282006/final-yr-project (public). `origin` remote
  still points at the official upstream `GasolSun36/PyRAG` for reference -- never push
  there.

## Instructions for whoever picks this up next (tomorrow)

Read this file plus `PROGRESS.md` in full before writing any code. The immediate next
task is recommended step 1 above (scale up sample size) or step 2 (build the ablation) --
ask the user which they'd rather do first if not already decided. Do NOT assume the
75.0% number is final or publication-worthy yet -- it is a promising n=8 result on a
single run, not a validated benchmark result.
