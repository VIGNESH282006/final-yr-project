"""
Reproduces the Question 7 failure from findings/2026-08-15_baseline_run_8q.md
("Who was known by his stage name Aladin and helped organizations improve their
performance as a consultant?") with a scripted FakeLLM, to prove contribution #1
(confidence-aware signals) changes the runner's behavior on exactly this case.

The FakeLLM below is scripted to reproduce the ORIGINAL baseline's exact wrong
final answer ("Aladin", gold: "Eenasul Fateh") with LOW confidence on both the
insufficient sub-answer AND the wrong final answer -- mirroring what a real model
would very plausibly emit under the new confidence-aware prompt. This test does not
prove the real model will always self-report low confidence correctly (that can only
be confirmed by re-running on Colab) -- it proves the RUNNER now reacts correctly
WHEN the model does.

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

        # 3. answer() call for "What is Aladin's real name?" -- correct, high confidence
        '<redacted_thinking>Doc 1 names him.</redacted_thinking>'
        '<answer>Eenasul Fateh</answer><confidence>high</confidence>',

        # 4. answer() call for "Which organizations..." -- model correctly signals it
        #    doesn't know, LOW confidence, matching Q7's real trace
        '<redacted_thinking>None of the documents name specific organizations.</redacted_thinking>'
        '<answer>unknown</answer><confidence>low</confidence>',

        # 5. FIRST final synthesis attempt (topk=3 round): confidently wrong,
        #    NO sentinel word -- this is what slipped past the OLD runner.py entirely.
        '<redacted_thinking>Based on the given facts, the consultant is Aladin.</redacted_thinking>'
        '<answer>Aladin</answer><confidence>low</confidence>',

        # --- runner.py should now trigger a retry because confidence="low" ---
        # The retry re-executes the SAME generated code from scratch, so all THREE
        # answer() calls run again, in order: real_name, organizations, final_answer.

        # 6. answer() retry for "What is Aladin's real name?" -- same as before, high confidence
        '<redacted_thinking>Doc 1 names him.</redacted_thinking>'
        '<answer>Eenasul Fateh</answer><confidence>high</confidence>',

        # 7. answer() retry for "Which organizations..." at boosted topk -- still unknown
        '<redacted_thinking>Still no organizations named in the retrieved documents.</redacted_thinking>'
        '<answer>unknown</answer><confidence>low</confidence>',

        # 8. SECOND final synthesis attempt (after retry): this time the model
        #    (correctly) falls back to the real name it already knows, still flagging
        #    low confidence about the "organizations" part it can't verify.
        '<redacted_thinking>Organizations are unverifiable, but the person\'s real name is known.</redacted_thinking>'
        '<answer>Eenasul Fateh</answer><confidence>medium</confidence>',
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
