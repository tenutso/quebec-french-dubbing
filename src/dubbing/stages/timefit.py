"""Stage 9: time-fit synthesized clips to the timeline.

French runs ~15-20% longer than English, so a spoken clip often overflows its cue slot.
Rather than chop the tail (which cuts words off mid-sentence), each clip is fit to a
*window* that starts at the cue and extends into the following silence up to the next
cue's clip. Within that window the clip keeps its natural timing; only if it is still
too long is it compressed (up to ~15%) and, as a last resort, trimmed. Uses ffmpeg
``rubberband`` when available (better quality) or ``atempo``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dubbing.pipeline import Context

MAX_SPEEDUP = 1.15  # a clip may be compressed by at most ~15% to fit its window
MIN_GAP = 0.06  # keep a small gap before the next cue's clip starts
MAX_BORROW = 2.5  # seconds of following silence a clip may extend into
FIT_SR = 48_000


def _wav_dur(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg time-fit failed:\n{' '.join(cmd)}\n{p.stderr}")


def fit_clip(src: Path, dst: Path, max_len: float) -> float:
    """Fit ``src`` into a window of at most ``max_len`` seconds; return the speed factor.

    If the clip already fits the window its natural timing is preserved (no stretch, no
    padding — the mix stage lays clips on a silent bus, so shorter is fine). If it is
    longer, it is compressed up to :data:`MAX_SPEEDUP`, then trimmed only for whatever
    overflow remains.
    """
    native = _wav_dur(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if native <= 0 or max_len <= 0:
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i",
              f"anullsrc=r={FIT_SR}:cl=mono", "-t", f"{max(max_len, 0.05):.3f}", str(dst)])
        return 1.0

    if native <= max_len + 1e-3:
        # Fits the window: keep natural timing, just normalize the sample rate.
        _run(["ffmpeg", "-y", "-i", str(src), "-af", f"aresample={FIT_SR}",
              "-ac", "1", str(dst)])
        return 1.0

    # Too long: compress up to the cap, then trim any residual overflow.
    speed = min(MAX_SPEEDUP, native / max_len)
    af = f"rubberband=tempo={speed}" if shutil.which("rubberband") else f"atempo={speed}"
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-af", f"{af},aresample={FIT_SR}",
        "-ac", "1", "-t", f"{max_len:.3f}", str(dst),
    ])
    return speed


def _window_end(cue, next_start: float) -> float:
    """The latest time this cue's clip may run to: into the following silence, but not
    over the next clip and not swallowing more than ``MAX_BORROW`` seconds of silence."""
    return min(next_start - MIN_GAP, cue.end + MAX_BORROW)


def run(ctx: "Context") -> None:
    cues_sorted = sorted(ctx.cues, key=lambda c: c.start)
    src_dur = ctx.source.duration if ctx.source else max(
        (c.end for c in ctx.cues), default=0.0
    )
    next_start = {
        c.index: (cues_sorted[i + 1].start if i + 1 < len(cues_sorted) else src_dur)
        for i, c in enumerate(cues_sorted)
    }
    by_index = {c.index: c for c in ctx.cues}

    for clip in ctx.clips:
        cue = by_index[clip.cue_index]
        # Window: at least the original slot, extended into following silence.
        max_len = max(cue.duration, _window_end(cue, next_start[cue.index]) - cue.start)
        fitted = ctx.job.work_dir / f"fit_{cue.index:05d}.wav"
        fit_clip(clip.wav_path, fitted, max_len)
        clip.wav_path = fitted
        clip.fitted_duration = _wav_dur(fitted)
