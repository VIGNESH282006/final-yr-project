import json
import re
from typing import Any, Dict, List


def extract_json_block(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1).strip())
    obj = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(1).strip())
    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


VALID_CONFIDENCE_LEVELS = ("high", "medium", "low")


def extract_confidence_tag(text: str) -> str:
    """
    Pulls the model's self-reported confidence out of a <confidence>...</confidence>
    tag. Falls back to "low" (the safe default) if the tag is missing or contains
    something other than high/medium/low -- we never want a parsing failure to look
    like high confidence.
    """
    m = re.search(r"<confidence>\s*(.*?)\s*</confidence>", text, re.DOTALL | re.IGNORECASE)
    if m:
        level = m.group(1).strip().lower()
        if level in VALID_CONFIDENCE_LEVELS:
            return level
    return "low"


def extract_confidence_word(text: str) -> str:
    """
    Parses a short free-text reply to a "reply with one word: high/medium/low" style
    question. Unlike extract_confidence_tag (which expects an exact <confidence> tag),
    this tolerates the model adding punctuation or a short surrounding sentence (e.g.
    "High." or "I'd say low.") by searching for the FIRST valid confidence word anywhere
    in the reply. Falls back to "low" if none of the three words appear at all -- we
    never want an unparseable reply to look like high confidence.
    """
    t = text.strip().lower()
    m = re.search(r"\b(high|medium|low)\b", t)
    if m:
        return m.group(1)
    return "low"


def normalize_python_source(code: str) -> str:
    if not code:
        return code
    trans = str.maketrans({
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    })
    code = code.translate(trans)
    code = re.sub(r"[\u200b-\u200d\ufeff]", "", code)
    return code


def extract_python_block(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return normalize_python_source(fence.group(1).strip())
    fence = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        return normalize_python_source(fence.group(1).strip())
    return normalize_python_source(text)


def format_docs_for_prompt(docs: List[str]) -> str:
    if not docs:
        return "No retrieved documents."
    return "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(docs)])
