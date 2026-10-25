"""Text normalization and error rates for TTS -> STT round trips.

Both the reference and the transcript go through the same normalizer, so
"Rs 12,450" and "twelve thousand four hundred fifty" can compare equal.
Heuristics are English-only and deliberately small; read them before
trusting a number.
"""

from __future__ import annotations

import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ABBREV = {"dr": "doctor", "mr": "mister", "mrs": "missus", "st": "street", "rs": "rupees", "inr": "rupees"}


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + int_to_words(r))
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            q, r = divmod(n, size)
            return int_to_words(q) + " " + name + ("" if r == 0 else " " + int_to_words(r))
    raise AssertionError("unreachable")


def _digits(s: str) -> str:
    return " ".join(_ONES[int(d)] for d in s)


def _number(tok: str) -> str:
    # OTPs, PINs, phone fragments and zero-led codes are read digit by digit.
    if len(tok) >= 5 or (len(tok) > 1 and tok.startswith("0")):
        return _digits(tok)
    return int_to_words(int(tok))


def normalize(text: str) -> list[str]:
    t = text.lower()
    t = t.replace("₹", " rupees ").replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"(?<=\d),(?=\d{2,3}\b)", "", t)  # 12,450 / 1,00,000 -> plain digits
    t = re.sub(r"\b(\d{1,2})[.:](\d{2})\s*(am|pm|a\.m\.|p\.m\.)", lambda m: f"{m.group(1)}:{m.group(2)} {m.group(3)[0]}m", t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", t)  # 5th -> 5: ordinal vs cardinal is not a hearing error
    t = re.sub(r"(\d+)\.(\d+)",lambda m: f"{m.group(1)} point {_digits(m.group(2))}", t)
    t = re.sub(r"(\d+):(\d{2})", lambda m: f"{m.group(1)} {'' if m.group(2) == '00' else m.group(2)}", t)
    t = re.sub(r"(?<=\w)@(?=\w)", " at ", t)
    t = re.sub(r"(?<=[a-z])\.(?=[a-z])", " dot ", t)
    t = t.replace("'", "")
    t = re.sub(r"[^\w\s]", " ", t)
    out: list[str] = []
    for tok in t.split():
        if tok.isdigit():
            out.extend(_number(tok).split())
        elif re.search(r"\d", tok) and re.search(r"[a-z]", tok):
            # alphanumeric codes (PNRs, booking refs) are spelled out
            for part in re.findall(r"\d+|[a-z]", tok):
                out.extend((_digits(part) if part.isdigit() else part).split())
        else:
            out.append(_ABBREV.get(tok, tok))
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (edits, reference_word_count) so callers can pool a corpus WER."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp), len(ref)


def missing_phrases(must_hear: list[str], hypothesis: str) -> list[str]:
    """Phrases from `must_hear` that do not appear, in order, in the transcript."""
    hyp = " " + " ".join(normalize(hypothesis)) + " "
    return [p for p in must_hear if " " + " ".join(normalize(p)) + " " not in hyp]
