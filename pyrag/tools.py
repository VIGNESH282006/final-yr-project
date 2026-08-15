from typing import Any, Callable, Dict, List, Optional, Tuple

from pyrag.llm import OpenAILLM
from pyrag.retrieval_agent import RetrievalAgent
from pyrag.utils import extract_answer_tag, extract_confidence_word, format_docs_for_prompt

# NOTE (PyRAG++ contribution #1, revised): we tried asking the model to emit a
# <confidence> tag INSIDE the same response as <redacted_thinking>/<answer>, across two
# rounds of increasingly forceful prompt wording. Qwen2.5-7B-Instruct never once produced
# the tag in either attempt (see findings/2026-08-15_contribution1_first_real_run.md) --
# a 7B instruct model reliably filling in a 3rd structured field in one pass turned out to
# be a real, reproducible limit of prompting alone, not something more forceful wording
# could fix. So confidence is now a SEPARATE, dedicated follow-up call (rate_confidence()
# below) with a much smaller, single-field task -- easier for a small model to get right,
# at the cost of one extra LLM call per answer().
ANSWER_SYSTEM_PROMPT_WITH_DOCS = (
    "You are given a question and retrieved documents.\n"
    "You MUST answer using ONLY information from the retrieved documents.\n"
    "Even for yes/no questions, decide yes or no by reasoning from facts in the documents.\n\n"
    "Output format (STRICT):\n"
    "<redacted_thinking> ... </redacted_thinking>\n"
    "<answer> ... </answer>\n\n"
    "Evidence citation rule:\n"
    "- Whenever you use evidence from the documents in your reasoning, you MUST cite it inline as Doc [i] "
    "(matching the document indices shown in the retrieved block, e.g. [Doc 1] → Doc [1]).\n"
    "- Only cite documents that are actually relevant.\n"
    "- Keep <redacted_thinking> concise (1–3 sentences).\n\n"
    "Answer rules:\n"
    "- The <answer> should be a short phrase, preferably taken directly from the documents when possible.\n"
    "- Match the answer TYPE to the QUESTION: WHO / which person / 谁先 / born first / earlier → a person's "
    "NAME in <answer>, not only a date; WHEN / 何时 → date or time; yes/no → exactly yes / no / unknown when "
    "the documents do not support a definite answer.\n"
    "- Do NOT output anything outside <redacted_thinking> and <answer>.\n\n"
    "Example (do NOT copy the content, only follow the style):\n"
    "<redacted_thinking>Doc [1] states that Future Ted serves as the narrator, and Doc [4] confirms the voice actor.</redacted_thinking>\n"
    "<answer> Ted Mosby </answer>\n"
)

ANSWER_SYSTEM_PROMPT_NO_DOCS = (
    "There are NO retrieved documents. The question text itself contains background facts "
    "(after 'Given:') and the actual question to answer (after 'Answer the question:').\n"
    "You MUST use the provided facts to answer the ACTUAL QUESTION.\n\n"
    "CRITICAL: Your job is to ANSWER the question, NOT to confirm whether the facts are true.\n"
    "- If the question asks WHO / WHICH person → reply with a person's NAME.\n"
    "- If the question asks WHEN → reply with a date or time.\n"
    "- If the question asks WHERE → reply with a place.\n"
    "- ONLY answer yes/no when the question is explicitly a yes/no question (e.g. 'Are both ...?', 'Is it true ...?').\n"
    "- NEVER answer 'yes' or 'no' to a WHO/WHICH/WHEN/WHERE question.\n\n"
    "Output format (STRICT):\n"
    "<redacted_thinking> ... </redacted_thinking>\n"
    "<answer> ... </answer>\n\n"
    "- In <redacted_thinking>, identify the actual question type and combine the given facts to produce the answer (1–2 sentences).\n"
    "- The <answer> must directly answer the question — a name, date, place, etc. — NOT 'yes' or 'no' unless the question is truly yes/no.\n"
    "- Do NOT output anything outside <redacted_thinking> and <answer>.\n"
)

ANSWER_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT_WITH_DOCS

CONFIDENCE_RATING_SYSTEM_PROMPT = (
    "You will be shown a QUESTION, the EVIDENCE that was available to answer it (documents, "
    "or background facts), and the ANSWER that was given. Your ONLY job is to rate how well the "
    "EVIDENCE actually supports the ANSWER.\n\n"
    "Reply with EXACTLY ONE WORD: high, medium, or low. Nothing else -- no explanation, no "
    "punctuation, no extra text.\n\n"
    "- high = the evidence directly and unambiguously supports the answer.\n"
    "- medium = the answer is a reasonable inference from the evidence, but not stated outright.\n"
    "- low = the evidence does NOT actually support the answer (e.g. the answer says "
    "'unknown', or the evidence doesn't mention the answer at all, or the evidence itself "
    "contains the word 'unknown').\n"
)


def _build_confidence_rating_prompt(query: str, evidence: str, answer_text: str) -> str:
    return (
        f"=== QUESTION ===\n{query}\n=== END QUESTION ===\n\n"
        f"=== EVIDENCE ===\n{evidence}\n=== END EVIDENCE ===\n\n"
        f"=== ANSWER GIVEN ===\n{answer_text}\n=== END ANSWER ===\n\n"
        "One word only: high, medium, or low."
    )


def rate_confidence(llm: OpenAILLM, query: str, docs: List[str], answer_text: str) -> str:
    """
    A small, dedicated follow-up call whose only job is rating confidence -- separate
    from the main answer() call, since a single generation asking for both an answer AND
    a confidence tag proved unreliable in practice (see the note above ANSWER_SYSTEM_PROMPT).
    """
    evidence = format_docs_for_prompt(docs) if docs else "(no documents -- background facts were embedded in the question above)"
    user_prompt = _build_confidence_rating_prompt(query, evidence, answer_text)
    raw = llm.generate(CONFIDENCE_RATING_SYSTEM_PROMPT, user_prompt)
    return extract_confidence_word(raw)


def answer_system_prompt_for_docs(docs: List[str]) -> str:
    return ANSWER_SYSTEM_PROMPT_WITH_DOCS if docs else ANSWER_SYSTEM_PROMPT_NO_DOCS


def make_tools(
    retrieval_agent: RetrievalAgent,
    llm: OpenAILLM,
    default_topk: int = 5,
    retrieve_topk_boost: Optional[Dict[int, int]] = None,
) -> Tuple[Callable, Callable, List[Dict[str, Any]]]:
    execution_log: List[Dict[str, Any]] = []
    boost = retrieve_topk_boost or {}
    retrieve_call_idx = [0]

    def retrieve(query: str, topk: int = default_topk) -> List[str]:
        retrieve_call_idx[0] += 1
        idx = retrieve_call_idx[0]
        if idx in boost:
            topk = max(topk, boost[idx])
        docs = retrieval_agent.retrieve(query, topk=topk)
        execution_log.append({
            "step": len(execution_log) + 1,
            "type": "retrieve",
            "query": query,
            "topk": topk,
            "docs": docs,
        })
        return docs

    def answer(query: str, docs: Optional[List[str]] = None) -> str:
        if docs is None:
            docs = []
        user_prompt = (
            f"=== QUESTION ===\n"
            f"{query}\n"
            f"=== END QUESTION ===\n\n"
            f"=== RETRIEVED DOCUMENTS ===\n"
            f"{format_docs_for_prompt(docs)}\n"
            f"=== END DOCUMENTS ==="
        )
        result = llm.generate(answer_system_prompt_for_docs(docs), user_prompt)
        returned = extract_answer_tag(result)
        confidence = rate_confidence(llm, query, docs, returned)
        execution_log.append({
            "step": len(execution_log) + 1,
            "type": "answer",
            "query": query,
            "docs": docs,
            "answer_raw": result,
            "answer_returned": returned,
            "confidence": confidence,
        })
        return returned

    return retrieve, answer, execution_log
