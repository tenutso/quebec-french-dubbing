"""Tests for time-fit (stage 9) and mix/master (stage 10)."""

from __future__ import annotations

import shutil
import wave

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

from dubbing.models import Cue, DubStyle, Job, LoudnessTarget, MediaAsset, SynthClip
from dubbing.pipeline import Context
from dubbing.stages import mix, timefit

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _tone_wav(path, seconds, sr=24000, freq=200.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr)


def _wav_dur(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def test_fit_clip_hits_target_within_stretch_bounds(tmp_path):
    src = tmp_path / "src.wav"
    _tone_wav(src, 1.0)
    dst = tmp_path / "dst.wav"
    # Ask for 0.95s: a 5% compression, inside the ±8% band.
    speed = timefit.fit_clip(src, dst, 0.95)
    assert _wav_dur(dst) == pytest.approx(0.95, abs=0.03)
    assert 1.0 <= speed <= timefit.MAX_SPEEDUP


def test_fit_clip_pads_when_target_longer(tmp_path):
    src = tmp_path / "src.wav"
    _tone_wav(src, 1.0)
    dst = tmp_path / "dst.wav"
    # Ask for 2.0s: beyond slowdown bound, so it stretches to the cap then pads.
    timefit.fit_clip(src, dst, 2.0)
    assert _wav_dur(dst) == pytest.approx(2.0, abs=0.03)


def _ctx_with_clips(tmp_path):
    job = Job(input_path=tmp_path / "in.mp4", work_dir=tmp_path / "w",
              dub_style=DubStyle.FULL_REPLACEMENT, loudness=LoudnessTarget.WEB)
    job.work_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(job=job)
    ctx.source = MediaAsset(path=job.input_path, duration=5.0, fps=25.0)

    ctx.cues = [
        Cue(index=0, start=0.5, end=1.5, speaker_id="S1", source_text="a",
            target_text_dub="un"),
        Cue(index=1, start=2.0, end=3.0, speaker_id="S2", source_text="b",
            target_text_dub="deux"),
    ]
    ctx.clips = []
    for c in ctx.cues:
        w = job.work_dir / f"tts_{c.index}.wav"
        _tone_wav(w, c.duration)
        ctx.clips.append(SynthClip(cue_index=c.index, wav_path=w,
                                   native_duration=c.duration))
    return ctx


def test_mix_places_clips_and_normalizes_loudness(tmp_path):
    ctx = _ctx_with_clips(tmp_path)
    timefit.run(ctx)
    mix.run(ctx)

    assert ctx.dub_track is not None and ctx.dub_track.path.exists()
    data, sr = sf.read(str(ctx.dub_track.path))
    assert sr == mix.SR
    # Track length matches source duration.
    assert len(data) / sr == pytest.approx(5.0, abs=0.05)

    # Loudness is within ~1.5 LU of the web target (-16 LUFS).
    loud = pyln.Meter(sr).integrated_loudness(data)
    assert loud == pytest.approx(LoudnessTarget.WEB.lufs, abs=1.5)

    # Silence before the first cue (0..0.5s) stays near-silent; energy appears in-cue.
    pre = data[: int(0.4 * sr)]
    incue = data[int(0.6 * sr): int(1.4 * sr)]
    assert np.abs(pre).mean() < np.abs(incue).mean()
