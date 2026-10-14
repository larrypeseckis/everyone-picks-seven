"""Deterministic integer extraction. Failures are counted, never dropped.

Rule (frozen into the manifest as PARSE_RULE):
  1. Remove any <think>...</think> block.
  2. Remove echoes of the range itself ("1 and N", "1 to N", "1-N", "1 through N")
     and markdown list enumerators at the start of a line ("1. ", "2) ").
  3. Collect every numeric token (integers and decimals).
  4. If exactly one distinct numeric token remains, it is an integer, and it is
     in 1..N, that is the answer. Repeats of the same value are fine ("7. It's 7!").
  5. If several distinct numeric tokens remain but exactly one distinct number
     appears in markdown bold (**17**), that bolded number is the answer.
  6. If there are no numeric tokens, apply rule 4 to English number words
     one..twenty.
  7. Otherwise the response is a failure with a reason code:
     no_number | multiple_numbers | out_of_range | non_integer.
A lenient variant (first in-range integer after steps 1-2) is used only for the
reported parse-rule sensitivity check, never for the headline.
"""
from __future__ import annotations

import re

PARSE_RULE = __doc__.split("Rule (frozen into the manifest as PARSE_RULE):", 1)[1].strip()
_BOLD = re.compile(r"\*\*\s*(-?\d+(?:\.\d+)?)\s*[!.]?\s*\*\*")

WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_NUM = re.compile(r"(?<!\w)(?<!\d\.)-?\d+(?:\.\d+)?(?!\w)")
_LIST = re.compile(r"^\s*\d+[.)]\s+", re.M)
_WORD = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.I)


def _range_echo(n: int) -> re.Pattern:
    return re.compile(rf"(?<!\d)1\s*(?:and|to|through|thru|-|–|—)\s*{n}(?!\d)", re.I)


def parse_answer(text: str, n: int) -> tuple[int | None, str]:
    """Return (value, status). status is 'ok' or a failure reason."""
    t = _THINK.sub(" ", text or "")
    t = _LIST.sub(" ", _range_echo(n).sub(" ", t))
    tokens = _NUM.findall(t)
    if tokens:
        distinct = set(tokens)
        if len(distinct) > 1:
            bold = set(_BOLD.findall(t))
            if len(bold) != 1:
                return None, "multiple_numbers"
            distinct = bold
        tok = distinct.pop()
        if "." in tok:
            return None, "non_integer"
        v = int(tok)
        return (v, "ok") if 1 <= v <= n else (None, "out_of_range")
    words = {WORDS[w.lower()] for w in _WORD.findall(t)}
    if len(words) == 1:
        v = words.pop()
        return (v, "ok") if 1 <= v <= n else (None, "out_of_range")
    if len(words) > 1:
        return None, "multiple_numbers"
    return None, "no_number"


def parse_answer_lenient(text: str, n: int) -> tuple[int | None, str]:
    """Sensitivity-check rule only: the first in-range integer after think/echo removal."""
    t = _LIST.sub(" ", _range_echo(n).sub(" ", _THINK.sub(" ", text or "")))
    for tok in _NUM.findall(t):
        if "." not in tok and 1 <= int(tok) <= n:
            return int(tok), "ok"
    for w in _WORD.findall(t):
        if 1 <= WORDS[w.lower()] <= n:
            return WORDS[w.lower()], "ok"
    return None, "no_number"
