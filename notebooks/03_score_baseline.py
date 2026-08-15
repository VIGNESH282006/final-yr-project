"""
Score the 8-question baseline run from findings/2026-08-15_baseline_run_8q.md
using pyrag/eval.py's Exact Match metric.

Run with:  python notebooks/03_score_baseline.py   (from the PyRAG/ repo root)

Once we build a proper results-saving step in the Colab notebook (so results get
written to a JSON file instead of copy-pasted), this script will read that file
directly instead of using the hardcoded lists below.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrag.eval import exact_match, exact_match_score

QUESTIONS = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
    "What science fantasy young adult series, told in first person, has a set of companion books "
    "narrating the stories of enslaved worlds and alien species?",
    "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
    'The director of the romantic comedy "Big Stone Gap" is based in what New York city?',
    "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
    "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
    "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
]

PREDICTIONS = [
    "yes",
    "unknown",
    "The Lost Colony series",
    "No",
    "Greenwich Village, New York City",
    "YG Entertainment",
    "Aladin",
    "4,000",
]

GOLDS = [
    "yes",
    "Chief of Protocol",
    "Animorphs",
    "no",
    "Greenwich Village, New York City",
    "YG Entertainment",
    "Eenasul Fateh",
    "3,677 seated",
]


def main():
    print("=== Per-question results ===")
    for q, p, g in zip(QUESTIONS, PREDICTIONS, GOLDS):
        mark = "PASS" if exact_match(p, g) else "FAIL"
        print(f"[{mark}] {q}")
        print(f"       predicted: {p!r}   gold: {g!r}")

    score = exact_match_score(PREDICTIONS, GOLDS)
    n_correct = round(score * len(PREDICTIONS))
    print(f"\n=== Exact Match: {score:.1%} ({n_correct}/{len(PREDICTIONS)}) ===")


if __name__ == "__main__":
    main()
