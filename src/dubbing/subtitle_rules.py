"""Subtitle standards engine (Quebec French, course/broadcast).

Encodes the Netflix Timed Text Style Guide baseline for Canadian French so both the cue
builder (stage 5) and subtitle authoring (stage 7) enforce the same rules. Pure logic,
no I/O — fully unit-testable.

References: Netflix Timed Text Style Guide — Canadian French; Netflix General
Requirements (reading speed, line treatment).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Standards constants -----------------------------------------------------
MAX_LINE_CHARS = 42  # chars per line (incl. spaces/punct)
MAX_LINES = 2
MAX_CPS = 17.0  # reading speed, adult
MIN_DURATION = 5.0 / 6.0  # ~0.833 s
MAX_DURATION = 7.0  # s
MIN_GAP_FRAMES = 2  # frames between consecutive cues

# French function words we avoid stranding at the end of a line (keep them with the
# following word: articles, short prepositions, conjunctions, clitic pronouns).
_NO_BREAK_AFTER = {
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d",
    "à", "au", "aux", "et", "ou", "où", "que", "qu", "qui", "ne",
    "se", "ce", "je", "tu", "il", "on", "en", "y", "ma", "ta", "sa",
    "mon", "ton", "son", "mes", "tes", "ses", "nos", "vos", "leur",
    "par", "pour", "sur", "dans", "avec", "sans",
}


def char_count(text: str) -> int:
    """Displayed character count: newlines don't count, everything else does."""
    return len(text.replace("\n", ""))


def cps(text: str, duration: float) -> float:
    """Characters per second over ``duration`` seconds."""
    if duration <= 0:
        return float("inf")
    return char_count(text) / duration


def min_duration_for(text: str) -> float:
    """Shortest on-screen duration that keeps ``text`` at/under the CPS limit."""
    return max(MIN_DURATION, char_count(text) / MAX_CPS)


def apply_french_typography(text: str) -> str:
    """Normalize to fr-CA typographic conventions.

    * straight double quotes -> guillemets « » with inner non-breaking spaces
    * narrow/normal non-breaking space before : ; ! ?
    * ASCII "..." -> ellipsis glyph
    """
    text = text.replace("...", "…")

    # Guillemets: replace pairs of straight quotes left-to-right.
    out: list[str] = []
    open_quote = True
    for ch in text:
        if ch == '"':
            out.append("« " if open_quote else " »")
            open_quote = not open_quote
        else:
            out.append(ch)
    text = "".join(out)

    # Non-breaking space before two-part punctuation (Quebec/French convention).
    for p in (":", ";", "!", "?"):
        # collapse an existing normal space, then insert NBSP
        text = text.replace(" " + p, p).replace(p, " " + p)
    return text


def _token_ok_to_end_line(token: str) -> bool:
    cleaned = token.strip(".,;:!?…»«").lower().rstrip("'’")
    return cleaned not in _NO_BREAK_AFTER


def wrap_two_lines(text: str, max_chars: int = MAX_LINE_CHARS) -> list[str]:
    """Balance ``text`` into at most two lines on word boundaries.

    Prefers the break nearest the midpoint that does not strand a function word at the
    end of the first line. Returns 1 line if the whole thing fits.
    """
    text = " ".join(text.split())  # normalize whitespace
    if char_count(text) <= max_chars:
        return [text]

    words = text.split(" ")
    # Candidate break after word i (line1 = words[:i+1]); score by balance + no-strand.
    best_idx: int | None = None
    best_score = float("inf")
    for i in range(len(words) - 1):
        line1 = " ".join(words[: i + 1])
        line2 = " ".join(words[i + 1 :])
        if len(line1) > max_chars or len(line2) > max_chars:
            continue
        imbalance = abs(len(line1) - len(line2))
        penalty = 0 if _token_ok_to_end_line(words[i]) else max_chars  # avoid stranding
        score = imbalance + penalty
        if score < best_score:
            best_score, best_idx = score, i

    if best_idx is None:
        # Can't satisfy both lines <= max_chars; fall back to a hard midpoint split.
        mid = len(words) // 2
        return [" ".join(words[:mid]), " ".join(words[mid:])]
    return [" ".join(words[: best_idx + 1]), " ".join(words[best_idx + 1 :])]


@dataclass
class CueTiming:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def validate(
    text: str,
    timing: CueTiming,
    *,
    next_start: float | None = None,
    fps: float | None = None,
) -> list[str]:
    """Return a list of standards violations for one cue (empty == compliant).

    ``next_start``/``fps`` enable the inter-cue gap check when available.
    """
    problems: list[str] = []
    lines = text.split("\n")

    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines > {MAX_LINES}")
    for ln in lines:
        if len(ln) > MAX_LINE_CHARS:
            problems.append(f"line {len(ln)} chars > {MAX_LINE_CHARS}: {ln!r}")

    dur = timing.duration
    if dur < MIN_DURATION - 1e-6:
        problems.append(f"duration {dur:.3f}s < {MIN_DURATION:.3f}s")
    if dur > MAX_DURATION + 1e-6:
        problems.append(f"duration {dur:.3f}s > {MAX_DURATION}s")

    rate = cps(text, dur)
    if rate > MAX_CPS + 1e-6:
        problems.append(f"reading speed {rate:.1f} CPS > {MAX_CPS}")

    if next_start is not None and fps:
        gap = next_start - timing.end
        min_gap = MIN_GAP_FRAMES / fps
        if gap < min_gap - 1e-6:
            problems.append(f"gap {gap*1000:.0f}ms < {MIN_GAP_FRAMES} frames")

    return problems
