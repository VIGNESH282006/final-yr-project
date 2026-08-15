"""
Reproduces the Question 7 failure from findings/2026-08-15_baseline_run_8q.md
("Who was known by his stage name Aladin and helped organizations improve their
performance as a consultant?") with a scripted FakeLLM, to prove contribution #1
(confidence-aware signals) changes the runner's behavior on exactly this case.

IMPORTANT (revised after real-model testing): the real Qwen2.5-7B-Instruct model
never reliably emitted an in-response <confidence> tag, even after two rounds of
stronger prompting (see findings/2026-08-15_contribution1_first_real_run.md). So
confidence is now rated by a SEPARATE, dedicated follow-up call
(pyrag.tools.rate_confidence) -- meaning each answer() call now makes TWO LLM
generate() calls instead of one: the main answer, then a one-word confidence
rating. This script's ScriptedLLM must supply a response for each of those calls,
in order.

This test does not prove the real model will always self-report low confidence
correctly (that can only be confirmed by re-running on Colab) -- it proves the
RUNNER + rate_confidence() plumbing reacts correctly WHEN the model does.

Run with:  python notebooks/04_test_confidence_fix.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrag import MockRetrievalAgent, RAGProgramRunner


class ScriptedLLM:
    """Replays a fixed sequence of responses, one per call to .generate()."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt[:30], user_prompt[:60]))
        if not self.responses:
            raise AssertionError(f"ScriptedLLM ran out of responses at call {len(self.calls)}")
        return self.responses.pop(0)


def main():
    responses = [
        # 1. DecomposeAgent.decompose()
        '["What is Aladin\'s real name?", "Which organizations did he consult for?"]',

        # 2. PlanAgent.generate_code()
        '''```python
docs1 = retrieve("What is Aladin's real name?")
real_name = answer("What is Aladin's real name?", docs1)

docs2 = retrieve(f"Which organizations did {real_name} consult for?")
organizations = answer(f"Which organizations did {real_name} consult for?", docs2)

final_answer = answer(
    f"Given: Aladin's real name is {real_name}, and he consulted for {organizations}. "
    f"Answer the question: Who was known by his stage name Aladin and helped organizations "
    f"improve their performance as a consultant?"
)
```''',

        # --- FIRST execution attempt (topk=3) ---
        # 3. answer() main call for "What is Aladin's real name?" -- correct
        '<redacted_thinking>Doc 1 names him.</redacted_thinking><answer>Eenasul Fateh</answer>',
        # 4. rate_confidence() follow-up for that answer -- well-supported, high
        'high',

        # 5. answer() main call for "Which organizations..." -- doesn't know
        '<redacted_thinking>None of the documents name specific organizations.</redacted_thinking>'
        '<answer>unknown</answer>',
        # 6. rate_confidence() follow-up -- evidence doesn't support it, low
        'low',

        # 7. FIRST final synthesis attempt: confidently wrong, no sentinel word
        '<redacted_thinking>Based on the given facts, the consultant is Aladin.</redacted_thinking>'
        '<answer>Aladin</answer>',
        # 8. rate_confidence() follow-up -- this is the crucial one: the evidence
        #    ("organizations: unknown") does NOT actually support "Aladin" as an answer,
        #    so a reasonable rater says low, even though the answer text has no sentinel word.
        'low',

        # --- runner.py should now trigger a retry because confidence="low" ---
        # The retry re-executes the SAME generated code from scratch, so all THREE
        # answer() calls run again, each now making 2 LLM calls (answer + confidence).

        # 9-10. retry: "What is Aladin's real name?" -- same as before
        '<redacted_thinking>Doc 1 names him.</redacted_thinking><answer>Eenasul Fateh</answer>',
        'high',

        # 11-12. retry: "Which organizations..." at boosted topk -- still unknown
        '<redacted_thinking>Still no organizations named in the retrieved documents.</redacted_thinking>'
        '<answer>unknown</answer>',
        'low',

        # 13-14. SECOND final synthesis attempt: this time falls back to the real name
        '<redacted_thinking>Organizations are unverifiable, but the person\'s real name is known.</redacted_thinking>'
        '<answer>Eenasul Fateh</answer>',
        'medium',
    ]

    llm = ScriptedLLM(responses)
    retrieval_agent = MockRetrievalAgent()
    runner = RAGProgramRunner(llm=llm, retrieval_agent=retrieval_agent)

    result = runner.run(
        "Who was known by his stage name Aladin and helped organizations improve "
        "their performance as a consultant?",
        topk=3,
    )

    print("\n\n=== TEST CHECK ===")
    print("retried_with_topk10:", result["retried_with_topk10"])
    print("final_answer:", result["final_answer"])
    assert result["retried_with_topk10"] is True, "Expected the confidence-aware retry to fire"
    assert result["final_answer"] == "Eenasul Fateh", "Expected the retry to recover the correct answer"
    print("PASS: confidence-aware retry fired, and recovered the correct answer.")


if __name__ == "__main__":
    main()
