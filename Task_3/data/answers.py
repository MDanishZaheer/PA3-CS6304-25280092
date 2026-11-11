# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Answer equivalence and correctness without gold-informed candidate grouping."""

import re
from functools import lru_cache

import pandas as pd
from Levenshtein import distance
from math_verify import parse, verify
from math_verify.errors import TimeoutException


NO_ANSWER = "NO_ANSWER"
GROUP_COMPARISON_TIMEOUT = 15
GOLD_COMPARISON_TIMEOUT = 120
RETRY_COMPARISON_TIMEOUT = None
_comparison_counts = {"calls": 0, "retries": 0, "timeouts": 0,
                      "errors": 0, "parse_timeouts": 0}


def comparison_counts(reset=False):
    counts = _comparison_counts.copy()
    if reset:
        for key in _comparison_counts:
            _comparison_counts[key] = 0
    return counts


def missing(answer):
    return bool(pd.isna(answer)) or str(answer).strip() in ("", NO_ANSWER)


def _clean_text(answer):
    return " ".join(str(answer).strip().lower().split()).rstrip(". ")


@lru_cache(maxsize=100000)
def _parsed_math(answer):
    answer = str(answer).strip()
    # The export stores the contents of \boxed{} without a LaTeX environment.
    # Math-Verify needs one to parse LaTeX rather than an inner number alone.
    attempts = [answer] if "$" in answer else [f"${answer}$", answer]
    for expression in attempts:
        try:
            parsed = parse(expression, fallback_mode="no_fallback")
            if parsed:
                return parsed
        except TimeoutException:
            _comparison_counts["parse_timeouts"] += 1
        except Exception:
            continue
    return []


@lru_cache(maxsize=300000)
def _math_equivalent(left, right, timeout_seconds=GOLD_COMPARISON_TIMEOUT):
    if str(left).strip() == str(right).strip():
        return True
    a, b = _parsed_math(left), _parsed_math(right)
    if not a or not b:
        return False
    _comparison_counts["calls"] += 1
    try:
        # The reference (gold or earlier group representative) comes first.
        return bool(verify(b, a, timeout_seconds=timeout_seconds, raise_on_error=True))
    except TimeoutException:
        _comparison_counts["timeouts"] += 1
        return False
    except Exception:
        _comparison_counts["errors"] += 1
        return False


def _choice_letter(answer, choices):
    text = str(answer).strip()
    parenthesized = re.findall(r"\(([a-zA-Z])\)", text)
    if parenthesized:
        text = parenthesized[0].upper()
    found = re.fullmatch(r"[A-Z]", text)
    if found and ord(text) - ord("A") < len(choices):
        return text
    nearest = min(range(len(choices)), key=lambda index: distance(text, str(choices[index])))
    return chr(ord("A") + nearest)


def canonical_mathvista(answer, metadata):
    if missing(answer):
        return NO_ANSWER
    choices = metadata.get("choices")
    if metadata.get("question_type") == "multi_choice":
        if not isinstance(choices, list) or not choices:
            raise ValueError("MathVista multiple-choice metadata needs choices.")
        return _choice_letter(answer, choices)
    answer_type = str(metadata.get("answer_type", "text")).lower()
    text = str(answer).strip()
    if answer_type in ("integer", "float"):
        try:
            if answer_type == "integer":
                return str(int(float(text)))
            precision = metadata.get("precision")
            return str(round(float(text), int(precision)))
        except (TypeError, ValueError, OverflowError):
            return NO_ANSWER
    if answer_type == "list":
        return str(answer)
    return _clean_text(text)


def equivalent(left, right, dataset_name, metadata, timeout_seconds=GOLD_COMPARISON_TIMEOUT):
    if missing(left) or missing(right):
        return False
    if dataset_name == "mathvista":
        a = canonical_mathvista(left, metadata)
        b = canonical_mathvista(right, metadata)
        return a != NO_ANSWER and a == b
    return _math_equivalent(str(left), str(right), timeout_seconds)


def answer_groups(answers, dataset_name, metadata):
    """Return group IDs based only on candidate answers, never the reference."""
    representatives = []
    groups = []
    for answer in answers:
        if missing(answer) or (dataset_name == "mathvista" and
                               canonical_mathvista(answer, metadata) == NO_ANSWER):
            groups.append(NO_ANSWER)
            continue
        match = next((index for index, representative in enumerate(representatives)
                      if equivalent(answer, representative, dataset_name, metadata,
                                    GROUP_COMPARISON_TIMEOUT)), None)
        if match is None:
            match = len(representatives)
            representatives.append(answer)
        groups.append(f"answer_{match}")
    return groups


def correctness(answers, gold, dataset_name, metadata):
    return [bool(equivalent(answer, gold, dataset_name, metadata)) for answer in answers]


def last_boxed(response):
    """Extract the final balanced-brace boxed answer; return None on missing extraction."""
    start = str(response).rfind(r"\boxed{")
    if start < 0:
        return None
    depth = 1
    begin = start + len(r"\boxed{")
    for index in range(begin, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[begin:index].strip() or None
    return None
