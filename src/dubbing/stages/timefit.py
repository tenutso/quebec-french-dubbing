"""Stage 9: time-fit synthesized clips to their cue slots.

Each clip is stretched/compressed toward its cue duration, then padded/trimmed to exactly
that length so it drops cleanly onto the timeline. The stretch factor is bounded (default
±8%) so prosody isn't wrecked; the LLM's length-controlled dub text keeps most clips well
within that band. Uses ffmpeg ``atempo`` (rubberband if available for better quality).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dubbing.pipeline import Context

MAX_SPEEDUP = 1.08  # clip may be shortened by at most 8%
MAX_SLOWDOWN = 0.92  # ...or lengthened by at most ~8%
FIT_SR = 48_000


def _wav_dur(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg time-fit failed:\n{' '.join(cmd)}\n{p.stderr}")


def fit_clip(src: Path, dst: Path, target: float) -> float:
    """Fit ``src`` to exactly ``target`` seconds; return the applied speed factor."""
    native = _wav_dur(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if native <= 0 or target <= 0:
        # Degenerate: emit exactly `target` of silence.
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i",
              f"anullsrc=r={FIT_SR}:cl=mono", "-t", str(max(target, 0.01)), str(dst)])
        return 1.0

    speed = native / target
    speed = min(MAX_SPEEDUP, max(MAX_SLOWDOWN, speed))

    use_rb = shutil.which("rubberband") is not None
    if use_rb:
        af = f"rubberband=tempo={speed}"
    else:
        af = f"atempo={speed}"

    # Stretch, resample, then pad-or-trim to exactly `target` (apad + -t).
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-af", f"{af},aresample={FIT_SR},apad",
        "-ac", "1", "-t", f"{target:.3f}", str(dst),
    ])
    return speed


def run(ctx: "Context") -> None:
    for clip in ctx.clips:
        cue = next(c for c in ctx.cues if c.index == clip.cue_index)
        fitted = ctx.job.work_dir / f"fit_{cue.index:05d}.wav"
        fit_clip(clip.wav_path, fitted, cue.duration)
        clip.wav_path = fitted
        clip.fitted_duration = cue.duration
