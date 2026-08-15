import re
import string
from typing import List


def normalize_answer(text: str) -> str:
    """
    Turn an answer string into a canonical form so trivial differences
    (capitalization, "the"/"a", punctuation, extra spaces) don't count as
    a mismatch. This is the standard normalization used by SQuAD/HotpotQA-style
    Exact Match scoring -- we're following the same convention the PyRAG paper
    itself uses to evaluate on these benchmarks.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())  # collapse repeated whitespace
    return text


def exact_match(prediction: str, gold: str) -> bool:
    """True if prediction and gold are identical after normalization."""
    return normalize_answer(prediction) == normalize_answer(gold)


def exact_match_score(predictions: List[str], golds: List[str]) -> float:
    """Fraction of predictions that exactly match their gold answer (0.0-1.0)."""
    if len(predictions) != len(golds):
        raise ValueError(
            f"predictions and golds must be the same length, got {len(predictions)} vs {len(golds)}"
        )
    if not predictions:
        return 0.0
    matches = sum(exact_match(p, g) for p, g in zip(predictions, golds))
    return matches / len(predictions)
