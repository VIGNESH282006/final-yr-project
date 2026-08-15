# PyRAG++ Progress Log

Running log of what's been built, in session order. Update this at the end of each work session.

## 2026-08-15 — Session 1: Clone + understand the pipeline

**Done:**
- Cloned official PyRAG repo (https://github.com/GasolSun36/PyRAG) into this directory.
- Read the full README and every file in `pyrag/` (`runner.py`, `decompose_agent.py`,
  `plan_agent.py`, `code_executor.py`, `tools.py`, `retrieval_agent.py`, `llm.py`, `utils.py`)
  plus `main.py`, to understand the Decompose → Plan → Execute flow end to end
  (see "Pipeline architecture" note below).
- Confirmed, by reading the actual code (not just the paper), that the flaw behind
  contribution #1 (sentinel-string adaptive retrieval) is real and lives in
  `pyrag/runner.py` lines 15-34: `_INSUFFICIENT_ANSWER_MARKERS` is a hardcoded tuple of
  strings like `"unknown"`, `"cannot answer"`, etc., and `_answer_indicates_insufficient_info()`
  does plain substring matching on the answer text.
- Identified infrastructure gaps between the README's assumed setup (2x vLLM servers,
  multi-GPU tensor parallelism, full Wikipedia-2018 E5 retrieval server) and our free-tier
  constraints. `requirements.txt` in the repo is actually the VERL **training** dependency
  list (flash-attn, ray, pinned vllm==0.8.4), not a lightweight inference list, and there's
  no `setup.py`/`pyproject.toml` despite the README's `pip install -e .` instruction.

**Decisions made:**
- All actual model/LLM execution happens in Google Colab notebooks (free GPU tier). This
  local machine is used only for editing code, docs, and notebooks — no local GPU runs.
- Model serving: skip vLLM. Load `Qwen/Qwen2.5-7B-Instruct` (Decompose + Answer agents) and
  `Qwen/Qwen2.5-Coder-7B-Instruct` (Plan agent) directly via Hugging Face `transformers` +
  `bitsandbytes` 4-bit quantization, in-process, no server. We'll write a small class with
  the same `.generate(system_prompt, user_prompt)` interface as `pyrag.llm.OpenAILLM` so the
  rest of the pipeline (`runner.py`, `tools.py`, agents) needs zero changes.
- Retrieval: skip the full Wikipedia E5 server. For the first 5-10 HotpotQA sample questions,
  embed each question's own shipped context paragraphs (HotpotQA ships gold + distractor
  paragraphs per question) locally with E5-base and do in-memory cosine similarity search.
  Real E5 model, real dense retrieval, zero server — and this same approach extends
  naturally to the Tamil/Hindi multilingual work later (contribution #4).

**Pipeline architecture (for viva reference):**
```
RAGProgramRunner.run(query)                              [pyrag/runner.py]
  1. DecomposeAgent.decompose(query)                      [pyrag/decompose_agent.py]
       -> LLM returns JSON list of atomic sub-questions
  2. PlanAgent.generate_code(query, sub_queries)           [pyrag/plan_agent.py]
       -> code-LLM writes Python using only retrieve()/answer()
       -> compile()-checked immediately (catches SyntaxError pre-execution)
  3. CodeExecutor.execute(code, retrieve_fn, answer_fn)    [pyrag/code_executor.py]
       -> exec()'s the generated Python string directly
       -> retrieve_fn/answer_fn (built in pyrag/tools.py::make_tools) log every
          call into execution_log -> this is the "inspectable trace"
       -> any Python exception -> RuntimeError -> fed back into
          PlanAgent.fix_code() for up to 3 repair rounds
          = "compiler-grounded self-repair"
  4. Adaptive retrieval (back in runner.py, after execution):
       scans execution_log for answers containing hardcoded strings like
       "unknown"/"cannot answer" -> re-runs that retrieve() step with topk boosted
       to 10. THIS is the exact mechanism contribution #1 replaces.
```

**Done (cont.):**
- Built and ran `notebooks/01_smoke_test.py` locally (no GPU/model needed): a `FakeLLM`
  class with canned responses stands in for `pyrag.llm.OpenAILLM`, paired with the repo's
  real `MockRetrievalAgent`. Confirmed full Decompose -> Plan -> compile() -> exec() ->
  retrieve()/answer() tool-call logging -> final answer wiring all works correctly.
- Gotcha found + fixed: Windows console default codepage (cp1252) can't print the `→`
  arrow character `runner.py` uses in its trace output -> crashes with UnicodeEncodeError.
  Fix: always run scripts with `PYTHONIOENCODING=utf-8` set (Colab/Linux won't have this
  problem, but note it for any local Windows runs).

**Done (cont.):**
- Verified locally (CPU, no GPU needed) that `datasets.load_dataset("hotpot_qa",
  "distractor", split=...)` works and confirmed its schema: each question ships
  `context.title` (list of article titles) + `context.sentences` (list of sentence
  lists) for 10 paragraphs -- a mix of gold-relevant and distractor text. No Wikipedia
  dump needed; this is our retrieval corpus.
- Built `notebooks/02_real_pipeline.ipynb` (17 cells, Colab-ready): clone+install ->
  load N=8 HotpotQA sample -> `E5LocalRetrievalAgent` (real E5-base embeddings +
  cosine similarity over each question's own 10 context paragraphs, formatted to match
  `HttpRetrievalAgent`'s doc string format so `pyrag/tools.py` needs no changes) ->
  `LocalLLM` (transformers + 4-bit bitsandbytes wrapper around Qwen2.5-7B-Instruct /
  Qwen2.5-Coder-7B-Instruct, matching `OpenAILLM`'s `.generate(system, user)` interface
  exactly so `runner.py`/agents need ZERO changes) -> full `RAGProgramRunner.run()` on
  each question -> predicted vs. gold answer comparison.
- Validated the E5 retrieval cell (cell 3 logic) locally end-to-end on real HotpotQA
  example #1 ("Were Scott Derrickson and Ed Wood of the same nationality?") -- it
  correctly ranked the two person-biography paragraphs above a same-titled distractor
  (the film "Ed Wood (film)"), confirming the retrieval logic is correct before
  spending any Colab GPU time on it. Could NOT test the model-loading/full-pipeline
  cells locally (need GPU, ~10GB downloads) -- those are still unverified until run in
  Colab.

## 2026-08-15 — Session 2: First real baseline run + trace analysis

**Done:**
- Fixed a real bug hit during the Colab run: `datasets.load_dataset("hotpot_qa", ...)`
  fails on current `datasets`/`huggingface_hub` versions because the Hugging Face Hub
  now requires namespaced dataset repo ids. Correct id is `hotpotqa/hotpot_qa` (verified
  locally, same schema). Fixed in `notebooks/02_real_pipeline.ipynb` cell 5.
- User ran the full notebook successfully end-to-end on Colab (T4 GPU, free tier): real
  Qwen2.5-7B-Instruct + Qwen2.5-Coder-7B-Instruct (4-bit) + real E5-base local retrieval,
  on 8 real HotpotQA validation questions. **This is the first working, unmodified
  official-PyRAG baseline run of the whole project.**
- Read and analyzed all 8 traces in full. Raw output + analysis saved to
  `findings/2026-08-15_baseline_run_8q_raw_output.txt` (primary source, full traces) and
  `findings/2026-08-15_baseline_run_8q.md` (write-up/analysis) -- read those files for
  full detail, summary below.

**Baseline scorecard:** 4/8 correct, 1/8 correct-but-case-mismatch (needs EM
normalization), 3/8 wrong.

**Key finding -- two DIFFERENT failure modes found, don't conflate them:**
- **Q2** ("...government position...Corliss Archer..."): answer correctly says
  "unknown" -- adaptive retry fires correctly (topk 3->10), still correctly says
  "unknown" because the fact genuinely isn't in any of the 10 shipped context docs.
  This is a **retrieval-recall gap** (out of scope per project decisions), and actually a
  clean example of adaptive retrieval WORKING as designed, not a bug.
- **Q7** ("...stage name Aladin...consultant..."): this IS a real, clean reproduction of
  contribution #1's target flaw (paper's F2-style false-confidence blind spot), but it's
  actually TWO compounding bugs in one trace:
  1. Plan Agent's generated code computes `consultant_info = "Eenasul Fateh"` (the
     correct answer!) at an intermediate step, but never interpolates that variable into
     the final synthesis prompt -- only `stage_name` and `organizations` get used. Code-
     generation bug, not a confidence-signal bug.
  2. The final synthesis answer comes back as `"Aladin"` -- confidently phrased, no
     sentinel word anywhere in it -- despite being built on an input fact that WAS
     literally the string "unknown" one step earlier. `runner.py`'s substring check only
     fires on literal sentinel text, so this confidently-wrong final answer sails through
     undetected. THIS is the precise blind spot a structured confidence score needs to
     close.
  - **Use Q7 as the primary worked example for contribution #1's before/after demo.**
- **Q3** is a softer/secondary version of the same pattern (hallucinated final answer
  after an implied-negative intermediate step) -- keep as a secondary example, weaker
  than Q7 since there's no literal "unknown" text mid-trace to point to.
- **Q8** (seating capacity 4,000 vs gold 3,677) is a separate, minor issue: the source
  doc contains both numbers and the Answer Agent picked the rounded one. This is an
  extraction/scoring-normalization nuance, NOT a target for any of the 4 contributions --
  noted for when we build EM scoring, not for framework fixes.

**Next up (do this next session):**
1. Build a proper EM/F1 scoring harness with answer normalization (lowercasing,
   article/punctuation stripping per the paper's convention) -- needed regardless of
   what else we do, since right now we can only eyeball correctness (e.g. Q4's
   "No"/"no" case mismatch would currently misscore as wrong under naive exact-string
   comparison).
2. Start contribution #1 (confidence-aware signals): modify `pyrag/tools.py`'s
   `answer()` to request and return a structured confidence score, and update
   `pyrag/runner.py`'s adaptive-retrieval trigger to branch on that score instead of
   `_INSUFFICIENT_ANSWER_MARKERS` string matching. Use Q7 as the concrete before/after
   demo case once implemented -- re-run the SAME question and compare traces side by
   side with `findings/2026-08-15_baseline_run_8q_raw_output.txt`.
3. Still need to explain 4-bit quantization, cosine similarity, and `exec()`-based
   execution to the user interactively (only explained in chat prose so far, not
   walked through live against running code/output).

## 2026-08-15 — Session 3: EM scoring harness + official baseline number

**Done:**
- Built `pyrag/eval.py`: `normalize_answer()` (lowercase, strip punctuation, strip
  "a"/"an"/"the" -- the standard SQuAD/HotpotQA normalization convention) +
  `exact_match()` + `exact_match_score()`.
- Built `notebooks/03_score_baseline.py`, a runnable script scoring the 8-question
  baseline run from `findings/2026-08-15_baseline_run_8q.md`.
- User ran it locally (no Colab needed -- pure Python, no GPU): confirmed output
  matches exactly.

**OFFICIAL BASELINE NUMBER: 50.0% Exact Match (4/8) on unmodified official PyRAG**,
Qwen2.5-7B-Instruct/Qwen2.5-Coder-7B-Instruct (4-bit), local E5-base retrieval over
HotpotQA's own shipped context, 8 validation questions. This is the number every future
contribution gets compared against. Note Q4's "No"/"no" case-mismatch now correctly
scores as a PASS after normalization (it looked like a miss before eval.py existed).

**Scope decision RESOLVED:** user wants (a) the Tamil/Hindi multilingual extension
(already contribution #4) AND (b) an interactive UI for typing a custom question and
seeing the answer. Decisions locked in:
- UI = a **Streamlit** app (simple Python-only web UI, no separate JS/React stack to
  learn or defend in viva). NOT a full React/JS frontend -- would add an unnecessary
  second tech stack for a beginner to maintain/explain.
- UI is a **demo/deliverable layer**, explicitly NOT a 5th research contribution --
  doesn't fix a documented paper flaw, so it doesn't compete with #1-4 for "novelty"
  credit, it's just the presentation layer on top of them.
- **Sequencing: finish contributions #1, #2, #3 (logic fixes) and #4 (Tamil/Hindi)
  FIRST, build the Streamlit UI LAST**, once the pipeline already supports all 3
  languages -- avoids rebuilding the UI mid-way as multilingual support lands under it.
- Custom single-question support (no UI, just calling `runner.run("your question")`
  directly instead of looping over benchmark questions) needs no new code at all --
  it's already exposed by `RAGProgramRunner.run()`. Can be demonstrated any time,
  including right now, in a notebook cell.

**Updated project roadmap (high level, in order):**
1. ~~Understand baseline pipeline~~ (done, session 1)
2. ~~Get real end-to-end run working on Colab~~ (done, session 2)
3. ~~Build EM scoring harness, establish baseline number~~ (done, session 3 -- 50.0% EM)
4. Contribution #1: confidence-aware signals (replace sentinel-string matching in
   `runner.py`) -- next up
5. Contribution #2: anti under-decomposition checker
6. Contribution #3: type-validation layer feeding self-repair
7. Contribution #4: Tamil/Hindi multilingual extension (translated benchmarks +
   multilingual-e5/LaBSE retrieval)
8. Streamlit UI: custom question input, language picker (EN/TA/HI), live trace +
   answer display -- built last, wraps the finished multilingual pipeline
9. Final evaluation: EM/F1 across all 5 benchmarks x all fixes x all 3 languages,
   ablation study, write-up

## 2026-08-15 — Session 4: Contribution #1, step 1 (confidence tag capture)

**Done:**
- Added `extract_confidence_tag()` to `pyrag/utils.py`: parses a `<confidence>high|
  medium|low</confidence>` tag, defaults safely to `"low"` if the tag is missing or
  contains anything else (never silently treats a parse failure as high confidence).
  Tested standalone against 4 cases (normal, uppercase, missing tag, garbage value) --
  all correct.
- Updated `pyrag/tools.py`: both `ANSWER_SYSTEM_PROMPT_WITH_DOCS` and
  `ANSWER_SYSTEM_PROMPT_NO_DOCS` now instruct the model to also output a
  `<confidence>` tag, with explicit guidance that a fluent-sounding answer built on
  an uncertain input fact should still be marked low confidence (directly targets the
  Q7 failure pattern). `make_tools()`'s `answer()` closure now parses this tag and
  logs it into `execution_log` as a new `"confidence"` field per answer step.
  **Design choice:** `answer()`'s RETURN VALUE (the plain answer string) is completely
  unchanged -- confidence is captured out-of-band in execution_log only. This matters
  because Plan-Agent-generated code does things like `x = answer(...)` and later
  f-string-interpolates `x` directly; changing the return type would break every
  existing generated program and violates the Plan Agent's own prompt rule against
  parsing answer() return values.
- Tested in isolation: `make_tools()` + a fake LLM, both with and without a
  `<confidence>` tag in the model's raw output -- confirmed the answer text is
  unaffected either way, and confidence logs correctly / defaults to "low" safely
  when absent.
- Regression-checked `notebooks/01_smoke_test.py` (fake LLM with NO confidence tag
  at all, simulating an older-style response) -- output is byte-identical to before
  this change. No breakage.

**Done (cont.) -- step 2, runner.py confidence-aware trigger:**
- Added `_answer_step_is_insufficient()` to `pyrag/runner.py`: an answer step now
  counts as insufficient if EITHER the old sentinel-string check fires (kept as a
  safety net for models that don't reliably emit the new tag) OR its logged
  `confidence` is not `"high"`. Used by `_retrieve_indices_for_insufficient_answers`
  (intermediate steps, decides which retrieve() calls get topk-boosted).
- Added `_last_answer_entry()` helper (finds the final `answer()` call's log entry).
- Updated `run()`'s final-answer check: now retries if EITHER the final answer text
  has a sentinel word OR the final answer's own logged confidence is not `"high"` --
  this is the fix for the exact Q7 blind spot (confidently-wrong final text, no
  sentinel word, previously slipped through the check entirely).
- Tested `_answer_step_is_insufficient` / `_build_retrieve_topk_boost` /
  `_last_answer_entry` standalone against a hand-built execution_log shaped like
  Q7's real trace -- confirmed the step-5 "Aladin"/low-confidence entry is now
  correctly flagged as insufficient (old code would have returned False here, since
  "Aladin" contains no sentinel word).

**Done (cont.) -- step 3, full pipeline reproduction test:**
- Built `notebooks/04_test_confidence_fix.py`: a `ScriptedLLM` replays a fixed
  sequence of responses reproducing Q7's shape -- correct real name found early
  (high confidence), "organizations" sub-answer genuinely unknown (low confidence),
  and a first-attempt final answer that's confidently wrong ("Aladin", low
  confidence, NO sentinel word). Asserts that (a) the confidence-aware retry fires
  (`retried_with_topk10 == True`) and (b) after retry, final_answer recovers the
  correct name ("Eenasul Fateh").
- **Ran it: PASSED.** Confirmed end-to-end: the runner now retries on Q7's exact
  failure shape where the OLD code would have silently accepted "Aladin" as final.
- **Caveat, important:** this is a SCRIPTED/simulated proof of the runner's LOGIC,
  not proof the real Qwen models will self-report confidence honestly and
  consistently. That can only be confirmed by re-running the real question through
  real models on Colab.

**Next up:**
1. Re-run `notebooks/02_real_pipeline.ipynb` on Colab with the updated `pyrag/`
   source (need to re-clone or re-upload the modified files -- Colab currently has
   session 2's clone, which predates this fix) on the SAME 8 questions, especially
   Q7, and compare against the saved baseline in `findings/2026-08-15_baseline_run_8q*`.
   Score with `pyrag/eval.py` / update `notebooks/03_score_baseline.py` with the new
   predictions to get a new EM number to compare against the 50.0% baseline.
2. If the real model doesn't self-report confidence reliably (e.g. always says
   "high"), that's a finding in itself -- document it and consider whether the
   confidence prompt needs few-shot examples of low-confidence cases to calibrate
   the model better.
3. Once contribution #1 is validated against real models, write up a short
   before/after section (old trace vs new trace on Q7) for the project report,
   then move to contribution #2 (anti under-decomposition checker).

## 2026-08-15 — Session 5: own GitHub repo + Colab GPU-OOM fix

**Done:**
- Set up user's own GitHub repo (`https://github.com/VIGNESH282006/final-yr-project`,
  public) as the project's real remote, separate from `origin`
  (`GasolSun36/PyRAG`, kept as read-only reference to the official repo -- NEVER
  push there). Local git remote `myrepo` points at the user's repo.
  Pushed all work so far (contribution #1 code, eval.py, notebooks, findings,
  PROGRESS.md) there.
- User hit a REAL GPU out-of-memory error on Colab's free T4 trying to load BOTH
  Qwen2.5-7B-Instruct AND Qwen2.5-Coder-7B-Instruct in 4-bit at once -- two 7B
  models' combined memory (weights + activations + KV cache + framework overhead)
  exceeded what's actually free on a T4 after Colab's own reservations, despite
  earlier back-of-envelope math suggesting it would "just fit."
- **Fix: switched to ONE shared Qwen2.5-7B-Instruct model for both the Plan role
  and Decompose/Answer roles** (`shared_llm` used for both `instruct_llm` and
  `plan_llm` in the notebook). This matches the paper's own README, which
  explicitly documents single-model mode as valid ("simpler", vs. running two
  vLLM instances). Updated `notebooks/02_real_pipeline.ipynb` cells 1, 9/10-11
  accordingly, and fixed the clone cell to point at the user's own repo instead of
  the upstream one. Committed + pushed.

**Update, same session:** user successfully re-ran the full notebook after the OOM
fix. Result: **62.5% EM (5/8), up from 50.0% (4/8) baseline.** Q3 and Q7 (our two
flagged before/after cases) both flipped from wrong to correct; Q5 regressed
slightly (right city, lost "Greenwich Village" specificity); Q8 unchanged; Q2 got
factually richer/more correct information than baseline but is still EM-wrong due
to verbose phrasing not matching the terse gold string. Full comparison table and
traces in `findings/2026-08-15_contribution1_first_real_run.md`.

**IMPORTANT, must read before claiming success on contribution #1:** checked the
real raw model output line by line -- **the real Qwen2.5-7B-Instruct model is NOT
emitting the `<confidence>` tag at all**, in ANY of the 8 questions' answer() calls,
despite the prompt explicitly instructing it to. Verified locally that
`extract_confidence_tag()` on the real (tagless) raw text correctly falls back to
its safe default, `"low"`. Consequence: every single answer this run was logged
with `confidence="low"` UNCONDITIONALLY, so the new retry trigger effectively
became "always allow one retry with boosted retrieval," not "retry specifically
when the model signals genuine uncertainty." **The mechanism as originally
designed (discriminating between the model's genuinely high vs. low confidence)
is therefore still UNTESTED** -- this run is real evidence that boosted-retry
helps, but not yet evidence that structured self-reported confidence is what's
doing the work. Full analysis + candidate fixes (move the tag instruction earlier
in the prompt, add a low-confidence few-shot example, stricter format enforcement)
in `findings/2026-08-15_contribution1_first_real_run.md`.

**Done -- prompt fix attempt:** rewrote `CONFIDENCE_INSTRUCTIONS` and both
`ANSWER_SYSTEM_PROMPT_WITH_DOCS` / `ANSWER_SYSTEM_PROMPT_NO_DOCS` in `pyrag/tools.py`:
1. Confidence rule now appears FIRST in the system prompt (was previously buried
   after several other rule blocks), framed as "checked by an automated system,
   not optional" / "a response missing this tag is treated as FAILED."
2. Added a SECOND few-shot example to each prompt demonstrating a LOW-confidence
   case (previously only a high-confidence example was shown, which may have read
   as "just copy this pattern" rather than genuine per-case reasoning).
3. `ANSWER_SYSTEM_PROMPT_NO_DOCS` (used for the final synthesis call) now
   explicitly calls out: if any given background fact is itself the literal word
   "unknown", confidence must be capped at medium -- directly targets the Q7-style
   failure where an "unknown" input fact got silently treated as solid.
- Verified `extract_confidence_tag()` only ever runs on the model's OWN reply
  text (`result` from `llm.generate()`), never on the system prompt string --
  confirmed the new example blocks embedded in the prompt can't leak into
  parsing (architecturally impossible, not just incidentally fine).
- Regression-checked: `notebooks/01_smoke_test.py` and
  `notebooks/04_test_confidence_fix.py` both still pass unchanged after the
  prompt rewrite.

**Update, same session -- the prompt fix did NOT work either.** User re-ran on
Colab: still 62.5% EM (5/8, same as before), and inspection of every `raw:` block
in the output confirmed **the model STILL never emitted a `<confidence>` tag**,
despite the stronger "REQUIRED / checked by an automated system" wording and the
new low-confidence few-shot example. This round also broke Q7's Decompose Agent
(3x JSON parse failures, fell back to using the original question as its own
single sub-query) -- likely collateral damage from the system prompt getting
longer/more demanding. **Conclusion: prompt wording alone cannot reliably get
Qwen2.5-7B-Instruct to fill in a 3rd structured field (confidence) alongside
`<redacted_thinking>` and `<answer>` in one generation pass.** This is a real,
reproducible negative result (2 attempts, 16+ answer calls, zero tags emitted),
not noise -- documented as a finding in itself.

**Fix (architectural, not prompt wording): split confidence rating into a
SEPARATE, dedicated LLM call.** Rather than asking for 3 fields in one response,
`pyrag/tools.py::answer()` now:
1. Generates the answer exactly as before (reverted prompts back to their
   original simpler form -- `ANSWER_SYSTEM_PROMPT_WITH_DOCS`/`_NO_DOCS` no longer
   mention confidence at all).
2. Makes a SECOND, small follow-up call via the new `rate_confidence()` function:
   shown the question, the evidence, and the answer that was given, and asked to
   reply with ONE WORD (high/medium/low) rating how well the evidence supports
   the answer. Much smaller ask for a 7B model than a multi-field structured
   response.
- Added `pyrag/utils.py::extract_confidence_word()`: a more forgiving parser than
  `extract_confidence_tag()` (which expects an exact `<confidence>` tag) -- finds
  the first high/medium/low word ANYWHERE in a short free-text reply (tolerates
  "High.", "I'd say low, since...", etc.), still defaults safely to "low" if none
  of the three words appear at all.
- **Trade-off, must document in the report:** this doubles LLM calls (2 per
  `answer()` instead of 1), meaning roughly double generation time/compute per
  question on Colab. A deliberate, reasonable cost for a working signal, but
  worth being upfront about.
- Updated `notebooks/01_smoke_test.py`'s FakeLLM and
  `notebooks/04_test_confidence_fix.py`'s ScriptedLLM to supply a response for
  each of the now-2-calls-per-answer() sequence. Both regression tests re-run and
  PASS with the new architecture; `notebooks/03_score_baseline.py` unaffected
  (doesn't touch tools.py).

**Update, same session -- third real run, BEST result yet: 75.0% EM (6/8),** up
from 50.0% baseline. Q2 (Corliss Archer) regressed to "unknown" (down from a
factually-correct-but-verbose miss last run -- still EM-wrong either way, no
real change there). Q5 (Big Stone Gap) recovered full "Greenwich Village, New
York City" specificity, now correct again. Q3 and Q7 (our flagged before/after
cases) both remain fixed. Q1/Q4/Q6/Q8 stable. Retry lines (`[Retry] ...`) fired
for Q2, Q6, Q7 only -- NOT for Q1, Q3, Q4, Q5, Q8 -- meaning confidence
plausibly varied across questions rather than defaulting to one constant value
everywhere (unlike the earlier in-prompt-tag attempts where it silently
defaulted to "low" every time). This is the first run with real signal that the
two-call rate_confidence() architecture might be discriminating correctly.

**Caveat/gap found while reviewing this run:** `runner.py::_print_trace()` never
actually PRINTED the `confidence` field, so this conclusion had to be inferred
indirectly from which questions got a `[Retry]` line -- not verified directly
from the trace. **Fixed:** trace now prints `[confidence: high/medium/low]`
after every answer's arrow line. Regression-tested (`01_smoke_test.py`,
`04_test_confidence_fix.py`) -- both still pass. This fix is display-only, does
not change any pipeline behavior, but is necessary before the NEXT re-run can
actually confirm (not infer) that confidence varies per-answer.

**Update, same session -- 4th real run, STILL no `[confidence: ...]` visible
anywhere in the output**, despite user confirming they DID do Runtime -> Restart
session first. Diagnosed: this points to a Colab-specific gotcha, not a logic
bug -- if a `PyRAG` folder persists in the Colab filesystem across a session
restart (can happen depending on runtime type/state), `!git clone ... PyRAG`
fails silently inside a shell-magic cell (no cell error, execution just
continues), and `%cd PyRAG` then enters the STALE existing folder instead of a
fresh clone -- so the notebook keeps running old code with zero visible
indication anything is wrong. This fully explains why the trace-print fix from
earlier in this session appeared to have no effect: it was never actually
running.

**Fixed:** clone cell now `shutil.rmtree("PyRAG", ignore_errors=True)` before
cloning, guaranteeing a fresh checkout every single run regardless of leftover
state. Added a new verification cell right after clone that prints `git log -1
--oneline` -- compare this commit hash against
https://github.com/VIGNESH282006/final-yr-project/commits/main before trusting
any run's results. This class of bug (silently stale code) is worth remembering
for ANY future Colab notebook in this project, not just this one.

**Next up:** re-run `notebooks/02_real_pipeline.ipynb` on Colab once more (fresh
clone/pull + runtime restart), and FIRST check the printed commit hash matches
`2fc3ef9` (or whatever is newest on GitHub) before reading anything else. Then
read the actual `[confidence: ...]` value on every single answer line across all
8 questions -- that direct evidence (not inference from retry timing) is what
finally validates or refutes the confidence mechanism. Once confirmed, write the
before/after report section (Q7 full before/after trace comparison) and move to
contribution #2 (anti under-decomposition checker).

## How to resume next session
Read this file top to bottom, then check `notebooks/` for the latest numbered notebook to
see how far we got. Ask the user to confirm current status before proceeding if unclear.
