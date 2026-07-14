"""Stage 10: mix & master. Dub clips + background stem -> R128-normalized track.

Places each time-fitted clip on a silent timeline at its cue start, sums the dub voice
bus, mixes it with the preserved background stem (music/SFX) for a full-replacement dub,
then loudness-normalizes the master to the job's target (EBU R128 / LUFS).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from dubbing.models import DubStyle, MediaAsset

if TYPE_CHECKING:
    from dubbing.pipeline import Context

SR = 48_000
VOICEOVER_DUCK = 0.25  # original bed level under a UN-style voice-over


def _read_mono(path: Path, sr: int = SR) -> np.ndarray:
    data, file_sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        # simple linear resample; fine for a mono voice bus
        n = int(round(len(mono) * sr / file_sr))
        mono = np.interp(np.linspace(0, len(mono), n, endpoint=False),
                         np.arange(len(mono)), mono)
    return mono.astype(np.float32)


def _place_clips(ctx: "Context", total_samples: int) -> np.ndarray:
    """Sum time-fitted clips onto a silent bus at their cue start times."""
    bus = np.zeros(total_samples, dtype=np.float32)
    by_index = {c.index: c for c in ctx.cues}
    for clip in ctx.clips:
        cue = by_index[clip.cue_index]
        audio = _read_mono(clip.wav_path)
        start = int(round(cue.start * SR))
        end = min(start + len(audio), total_samples)
        if start < total_samples:
            bus[start:end] += audio[: end - start]
    return bus


def _bed(ctx: "Context", total_samples: int) -> np.ndarray:
    """The non-voice bed to mix under the dub, per dub style."""
    job = ctx.job
    if job.dub_style is DubStyle.FULL_REPLACEMENT and ctx.background_stem is not None:
        bed = _read_mono(ctx.background_stem.path)  # music/SFX only
    elif job.dub_style is DubStyle.VOICE_OVER and ctx.full_audio is not None:
        bed = _read_mono(ctx.full_audio.path) * VOICEOVER_DUCK  # ducked original
    else:
        bed = np.zeros(total_samples, dtype=np.float32)
    if len(bed) < total_samples:
        bed = np.pad(bed, (0, total_samples - len(bed)))
    return bed[:total_samples]


def run(ctx: "Context") -> None:
    duration = ctx.source.duration if ctx.source else (
        max((c.end for c in ctx.cues), default=0.0)
    )
    total = int(round(duration * SR))
    if total <= 0:
        return

    voice = _place_clips(ctx, total)
    mix = voice + _bed(ctx, total)

    # Guard against clipping before loudness measurement.
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix = mix / peak

    meter = pyln.Meter(SR)
    loudness = meter.integrated_loudness(mix)
    if np.isfinite(loudness):
        mix = pyln.normalize.loudness(mix, loudness, ctx.job.loudness.lufs)
    # Final true-peak safety limit.
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 0.99:
        mix = mix * (0.99 / peak)

    out = ctx.job.work_dir / "dub_track.wav"
    sf.write(str(out), mix.astype(np.float32), SR)
    ctx.dub_track = MediaAsset(
        path=out, duration=duration, sample_rate=SR, channels=1
    )
