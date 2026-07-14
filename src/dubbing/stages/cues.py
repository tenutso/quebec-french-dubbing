"""Stage 5: cue construction. ASR words -> standards-compliant caption/dub cues.

Segments the word stream into cues that respect subtitle standards (speaker boundaries,
pauses, sentence boundaries, max duration, and a character budget). Because French runs
~15-20% longer than English, cues are segmented against a source-side character budget
derived from the target two-line limit and an expansion factor, leaving room for the
translation to still fit two 42-char lines.

Timing/CPS are finalized after translation (the subtitle stage wraps + validates the
French text), but building conservative cues here keeps that step nearly always compliant.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from dubbing import subtitle_rules as rules
from dubbing.models import Cue, Word

if TYPE_CHECKING:
    from dubbing.pipeline import Context

# French expands over English; budget source text smaller so the translation fits.
EXPANSION_FACTOR = 1.2
# Max source chars per cue so the French rendering still fits two lines.
MAX_SOURCE_CHARS = int((rules.MAX_LINE_CHARS * rules.MAX_LINES) / EXPANSION_FACTOR)  # 70
# A silence longer than this forces a cue break (natural phrase boundary).
PAUSE_BREAK = 0.6  # seconds
# Approximate natural speaking rate for sizing the dub char budget.
DUB_CHARS_PER_SEC = 15.0
# Minimum inter-cue gap enforced in seconds when fps is unknown.
DEFAULT_MIN_GAP = 0.08

_SENTENCE_END = re.compile(r"[.!?…]$")


def _flush(cue_words: list[Word], index: int) -> Cue:
    text = " ".join(w.text for w in cue_words).strip()
    return Cue(
        index=index,
        start=cue_words[0].start,
        end=cue_words[-1].end,
        speaker_id=cue_words[0].speaker_id or "S?",
        source_text=text,
    )


def build_cues(words: list[Word]) -> list[Cue]:
    """Group ``words`` into cues honoring speaker/pause/sentence/length boundaries."""
    cues: list[Cue] = []
    current: list[Word] = []

    def running_len(extra: Word) -> int:
        joined = " ".join(w.text for w in current + [extra])
        return len(joined)

    for w in words:
        if current:
            prev = current[-1]
            speaker_changed = (w.speaker_id or prev.speaker_id) != (
                prev.speaker_id or w.speaker_id
            )
            long_pause = (w.start - prev.end) >= PAUSE_BREAK
            too_long_chars = running_len(w) > MAX_SOURCE_CHARS
            too_long_time = (w.end - current[0].start) > rules.MAX_DURATION
            sentence_boundary = bool(_SENTENCE_END.search(prev.text)) and len(
                " ".join(x.text for x in current)
            ) >= MAX_SOURCE_CHARS // 2

            if (
                speaker_changed
                or long_pause
                or too_long_chars
                or too_long_time
                or sentence_boundary
            ):
                cues.append(_flush(current, len(cues)))
                current = []

        current.append(w)

    if current:
        cues.append(_flush(current, len(cues)))

    _enforce_timing(cues)
    return cues


def _enforce_timing(cues: list[Cue]) -> None:
    """Extend too-short cues toward the CPS floor and keep a minimum inter-cue gap.

    Uses the *source* text as a proxy for length; the subtitle stage does the final
    French-text validation after translation.
    """
    for i, cue in enumerate(cues):
        need = rules.min_duration_for(cue.source_text)
        if cue.duration < need:
            limit = cues[i + 1].start - DEFAULT_MIN_GAP if i + 1 < len(cues) else math.inf
            cue.end = min(cue.start + need, max(cue.end, min(cue.start + need, limit)))
        # Keep a gap to the next cue.
        if i + 1 < len(cues):
            nxt = cues[i + 1]
            if nxt.start - cue.end < DEFAULT_MIN_GAP:
                cue.end = max(cue.start, nxt.start - DEFAULT_MIN_GAP)


def dub_char_budget(cue: Cue) -> int:
    """Soft char budget for the dub translation so TTS roughly fits the cue slot."""
    return max(10, int(cue.duration * DUB_CHARS_PER_SEC))


def run(ctx: "Context") -> None:
    ctx.cues = build_cues(ctx.words)
