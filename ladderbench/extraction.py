"""Answer extraction and checking for probe problems.

Deliberately dependency-free and heuristic: probe answers are exact numbers
or multiple-choice letters, so a small extractor with a strict comparator is
enough and stays identical across every model and server.
"""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_IS = re.compile(
    r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?(-?\d[\d,]*(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MC_LETTER = re.compile(r"\b([A-D])\b(?![\w])")


def extract_answer(text: str, kind: str = "exact") -> str | None:
    """Pull the predicted answer out of a model response.

    kind="exact" targets a numeric answer (\\boxed, "answer is X", or the last
    number in the text). kind="mcq" targets a standalone A-D letter.
    Returns None when nothing plausible is found (counted as incorrect).
    """
    if not text:
        return None
    if kind == "mcq":
        m = _MC_LETTER.search(text)
        return m.group(1) if m else None

    m = _BOXED.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    matches = _NUMBER.findall(text)
    if matches:
        return matches[-1].replace(",", "").strip()
    return None


def check_answer(predicted: str | None, gold: str) -> bool:
    """Strict comparator: numeric when both sides parse as numbers."""
    if predicted is None:
        return False
    p, g = predicted.strip(), gold.strip()
    try:
        return abs(float(p) - float(g)) <= 1e-6
    except ValueError:
        return p.lower() == g.lower()
