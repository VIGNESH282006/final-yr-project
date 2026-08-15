# Baseline run: 8 HotpotQA questions, training-free PyRAG, real models

**Date:** 2026-08-15
**Setup:** `notebooks/02_real_pipeline.ipynb` on Colab (T4 GPU). Qwen2.5-7B-Instruct
(Decompose + Answer), Qwen2.5-Coder-7B-Instruct (Plan), E5-base local retrieval over
each question's own HotpotQA `distractor`-split context paragraphs (topk=3, adaptive
boost to topk=10). No modifications to `pyrag/` source yet — this is the **unmodified
official PyRAG baseline**, first real end-to-end run.

This is the first working run of the whole framework on real data, and the first
evidence of where the baseline actually breaks vs. works, straight from HotpotQA's
validation split (not cherry-picked).

## Scorecard

| # | Question | Predicted | Gold | Verdict |
|---|----------|-----------|------|---------|
| 1 | Were Scott Derrickson and Ed Wood of the same nationality? | yes | yes | Correct |
| 2 | What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell? | unknown | Chief of Protocol | Wrong — retrieval-recall gap (see analysis) |
| 3 | What science fantasy YA series... has companion books about enslaved worlds and alien species? | The Lost Colony series | Animorphs | Wrong — hallucinated final answer |
| 4 | Are the Laleli Mosque and Esma Sultan Mansion in the same neighborhood? | No | no | Correct (case difference only) |
| 5 | The director of "Big Stone Gap" is based in what NY city? | Greenwich Village, NYC | Greenwich Village, NYC | Correct |
| 6 | 2014 S/S is the debut album of a South Korean boy group formed by who? | YG Entertainment | YG Entertainment | Correct |
| 7 | Who was known by his stage name Aladin and helped organizations as a consultant? | Aladin | Eenasul Fateh | Wrong — false-confidence / circular answer |
| 8 | The arena where the Lewiston Maineiacs played can seat how many people? | 4,000 | 3,677 seated | Close but not exact match |

**Raw: 4/8 exact-ish correct, 1/8 case-only mismatch, 3/8 wrong.** Not a formal EM score
(no answer normalization yet — see Next Steps) but a first honest read.

## Key finding: Q2 vs Q7 are two DIFFERENT failure modes, both worth keeping

It would be easy to lump "predicted unknown/wrong" together, but reading the full traces
shows two distinct things happening, and only one of them is what contribution #1 targets.

### Q2 — adaptive retrieval mechanism itself worked correctly

Trace shows: `answer()` for the sub-query legitimately said `unknown` because none of the
10 context documents (checked all 10 titles in the trace) mention Shirley Temple's real
post-acting government career as U.S. Chief of Protocol — that fact simply is not present
in this question's HotpotQA `distractor` context split. The runner correctly detected the
"unknown" sentinel, retried with `topk` boosted 3->10 (`[Retry] ... boost: {2: 10}` in the
log), pulled in more documents, and the Answer Agent still correctly said "unknown" because
the evidence genuinely isn't there. Final answer propagated as "unknown" cleanly — no
corruption, no silent misuse as fact.

**This is a genuine retrieval-recall gap** (out of scope per the paper's own Figure 4b and
our project scope — see `PROGRESS.md` / project memory), NOT the sentinel-string bug.
Actually a good example of the adaptive-retrieval mechanism working as designed.

### Q7 — the actual target flaw for contribution #1, reproduced

Trace shows the Answer Agent's own reasoning for the second retrieve/answer step:
> "None of the documents mention Aladin helping any organization improve their
> performance as a consultant... unknown"

So THIS answer correctly said `unknown`, boost fired (`{2: 10, 3: 10}`), and a third
`answer()` call (searching organizations specifically) ALSO reasoned "None of the
documents mention Aladin helping any organization..." and again said `unknown`. But then
the FINAL synthesis step, given `"...Aladin helped improve the performance of unknown."`,
produced:
> "the answer is Aladin" (i.e. it circularly answered the question with the question's
> own premise — the person's stage name — instead of the person's real name, Eenasul Fateh,
> which one of the RETRIEVED documents in step 1 (`Doc 4: Eenasul Fateh`) actually names
> correctly)

This is the real bug: the model had the correct name (`Eenasul Fateh`, visible in Doc 1's
retrieval at Step 1 and even used correctly as `consultant_info` mid-program) sitting right
there, but the LAST synthesis call — the one whose output becomes `final_answer` — produced
a confidently-phrased WRONG answer instead of surfacing an "unknown" signal, DESPITE having
just reasoned through why the evidence was insufficient. A plain string check on
`final_answer` (`_answer_indicates_insufficient_info`) would not catch this, because the
text "Aladin" doesn't contain any sentinel phrase — this is exactly the blind spot: a
confidently-worded wrong answer with no linguistic marker of low confidence.

This is the shape of the paper's own Failure Case F2 (sentinel/confidence conflation) and
directly motivates a structured confidence score from the Answer Agent instead of
string-matching its output text.

### Q3 — a softer version of the same pattern

Similar shape to Q7: sub-answer legitimately says "not have such companion books" (i.e.
effectively "no"/uncertain), boost fires, still can't find it, and the final synthesis step
invents `"The Lost Colony series"` out of nowhere rather than surfacing low confidence.
Worth keeping as a second example in the report, weaker/less clean than Q7 since there's no
literal "unknown" text to point to in the sub-step, just an implied negative.

### Q8 — a different, smaller issue: precision loss, not framework logic

Predicted "4,000" vs. gold "3,677 seated" — the retrieved document actually contains BOTH
numbers ("4,000 capacity (3,677 seated)"), and the Answer Agent picked the rounded
marketing figure over the specific seated count HotpotQA's gold answer uses. This is an
answer-extraction/precision nuance, not a decomposition or type or confidence bug — flag it
as a scoring/normalization note, not a target for any of the 4 contributions.

## Implication for contribution #1's design

Confidence needs to be captured **at every `answer()` call**, not just checked as a string
on the literal word "unknown" — including the FINAL synthesis call, which today is exactly
the one place `runner.py`'s `_answer_indicates_insufficient_info()` still checks (see
`result.get("final_answer", "")` in `runner.py`), but only via substring match. Q7 shows
that check firing on `"unknown"` at an intermediate step and STILL not preventing a
confidently-wrong final answer, because the intermediate "unknown" gets silently
interpolated into the next prompt as if it were a plain fact string, and the model just
talks past it. A structured confidence score would need to also propagate forward (e.g. "if
any input fact had low confidence, that should suppress high confidence in anything built
on top of it") — not just gate on the final text.

## Next steps arising from this run
1. Build a proper EM/F1 scoring harness (needed regardless of what we do next — right now
   we can only eyeball correctness, and Q4's "No" vs "no" needs normalization to score fairly).
2. Use Q7 (primary) and Q3 (secondary) as the worked examples when implementing and later
   demonstrating contribution #1 (confidence-aware signals) — show the SAME questions
   re-run after the fix, side by side with this baseline trace.
3. Q8 flagged separately as a scoring-normalization note, not a contribution target.
