"""Relative day words in a fact's text, made absolute when the agent stores it (0.32.0, absolute_dates).

"The meeting is tomorrow at 10:00" stored on 2026-11-27 reads, a month later, as a meeting that is always tomorrow.
The word stays and the day follows it: "tomorrow (2026-11-28)". Only words that name one day whatever the sentence
says around them; "in 3 days" (after what?), weekdays ("on Friday": which one?), "next week" and months are left as
they are. Nothing inside quotes, inline code or a URL changes, and a word already followed by a date is left alone,
so storing the same text again adds nothing.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

# (pattern, offset in days); the longer phrases first, so "the day after tomorrow" is not read as "tomorrow"
_WORDS = (
    (r"the day after tomorrow", 2), (r"the day before yesterday", -2),
    (r"tomorrow", 1), (r"yesterday", -1), (r"today", 0), (r"tonight", 0),
    (r"послезавтра", 2), (r"позавчера", -2), (r"завтра", 1), (r"вчера", -1), (r"сегодня", 0),
)
_ISO = r"\d{4}-\d{2}-\d{2}"
# Not part of a longer word ("сегодняшний", "todays"), not a possessive ("today's", "today’s": the day would split
# the phrase), not already followed by a date ("tomorrow (2026-11-28)", "tomorrow 2026-11-28").
_RE_DAY = re.compile(
    r"(?<![\w-])(" + "|".join(w for w, _ in _WORDS) + r")(?![\w-])(?!['’]s\b)(?!\s*\(?" + _ISO + ")",
    re.IGNORECASE)
_OFFSETS = {w.lower(): d for w, d in _WORDS}
# Spans the text quotes rather than says: «…», "…", “…”, „…“, `…`, and URLs
_RE_KEEP = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”|„[^“”]*[“”]|`[^`]*`|https?://\S+|www\.\S+", re.IGNORECASE)


def absolutize(text: str, today: date | None) -> tuple[str, list[str]]:
    """(*text* with the day after each relative day word, what was added as "tomorrow → 2026-11-28"). *today*: the
    person's day (None: not known, nothing is added)."""
    if today is None or not text:
        return text, []
    kept = [m.span() for m in _RE_KEEP.finditer(text)]
    added: list[str] = []

    def day(m: re.Match) -> str:
        if any(start <= m.start() < end for start, end in kept):
            return m.group(0)
        word = m.group(1)
        iso = (today + timedelta(days=_OFFSETS[re.sub(r"\s+", " ", word.lower())])).isoformat()
        added.append(f"{word} → {iso}")
        return f"{word} ({iso})"

    return _RE_DAY.sub(day, text), added


__all__ = ["absolutize"]
