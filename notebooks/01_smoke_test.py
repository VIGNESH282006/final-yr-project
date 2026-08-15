"""
Step 1: Prove the Decompose -> Plan -> Execute wiring works, with ZERO downloads
and ZERO GPU. No real LLM is called here -- we use a fake "canned response" LLM
so we can inspect exactly how the pipeline behaves before we introduce the
complexity of a real model.

Run with:  python notebooks/01_smoke_test.py   (from the PyRAG/ repo root)
"""

import sys
import os

# Make sure Python can find the `pyrag` package (it lives at PyRAG/pyrag,
# and we're running this script from PyRAG/notebooks/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrag import MockRetrievalAgent, RAGProgramRunner


class FakeLLM:
    """
    Stands in for pyrag.llm.OpenAILLM. The real class sends a system_prompt +
    user_prompt to a running model server and returns its text reply. Since we
    don't have a model yet, this class just returns pre-written text depending
    on WHICH agent is calling it (we can tell by a keyword in the prompt).

    This is only for testing plumbing -- it is NOT part of the real pipeline.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # DecomposeAgent's system prompt asks for a JSON list of sub-queries.
        if "decomposition agent" in system_prompt:
            return '["Who directed Inception?", "When was that person born?"]'

        # PlanAgent's system prompt asks for a Python program.
        if "planning agent" in system_prompt:
            return '''```python
docs1 = retrieve("Who directed Inception?")
director = answer("Who directed Inception?", docs1)

docs2 = retrieve(f"When was {director} born?")
birth = answer(f"When was {director} born?", docs2)

final_answer = answer(
    f"Given: {director} directed Inception and was born {birth}. "
    f"Answer the question: When was the director of Inception born?"
)
```'''

        # tools.py's rate_confidence() sends a small dedicated follow-up prompt
        # (identifiable by "ANSWER GIVEN") asking for one word: high/medium/low.
        # Every answer in this smoke test is well-supported, so always say "high".
        if "ANSWER GIVEN" in user_prompt:
            return "high"

        # tools.py's answer() function asks the LLM to answer using <answer> tags.
        # We tell these three answer() calls apart by the exact question text,
        # not loose keyword guesses (that's what tripped us up on the first run).
        if "Given:" in user_prompt:  # final synthesis call (no docs)
            return "<redacted_thinking>Combining the two facts.</redacted_thinking><answer>30 July 1970</answer>"
        if "When was" in user_prompt:  # second sub-question (birth date lookup)
            return "<redacted_thinking>Doc 2 gives the birth date.</redacted_thinking><answer>30 July 1970</answer>"
        return "<redacted_thinking>Doc 1 names the director.</redacted_thinking><answer>Christopher Nolan</answer>"


def main():
    fake_llm = FakeLLM()
    retrieval_agent = MockRetrievalAgent()  # tiny hardcoded 6-sentence corpus, keyword overlap search

    runner = RAGProgramRunner(
        llm=fake_llm,
        retrieval_agent=retrieval_agent,
    )

    result = runner.run("When was the director of Inception born?", topk=3)

    print("\n\n=== RESULT DICT KEYS ===")
    print(list(result.keys()))


if __name__ == "__main__":
    main()
