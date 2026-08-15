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

## How to resume next session
Read this file top to bottom, then check `notebooks/` for the latest numbered notebook to
see how far we got. Ask the user to confirm current status before proceeding if unclear.
