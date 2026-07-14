"""Regression tests for the GPU-stage <-> model mapping boundary.

The GPU runners speak the pyannote/whisperx dict shape (``{"speaker", ...}``); the stages
must translate that into the pipeline's ``SpeakerSegment``/``Word`` models. These map
mismatches only surface on a real run, so pin them here with fakes (no models needed).
"""

from __future__ import annotations

from pathlib import Path

from dubbing import gpu_runners
from dubbing.models import Job, MediaAsset, SpeakerSegment
from dubbing.pipeline import Context
from dubbing.stages import asr, diarize


def _ctx(tmp_path):
    job = Job(input_path=tmp_path / "in.mp4", work_dir=tmp_path / "w")
    ctx = Context(job=job)
    ctx.asr_audio = MediaAsset(path=tmp_path / "a.wav", duration=5.0, sample_rate=16000)
    ctx.vocals_stem = ctx.asr_audio
    return ctx


def test_diarize_stage_maps_speaker_key(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_runners, "diarize_audio",
                        lambda p: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5}])
    ctx = _ctx(tmp_path)
    diarize.run(ctx)
    assert ctx.segments == [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=1.5)]


def test_asr_stage_passes_speaker_shape_and_maps_words(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.segments = [SpeakerSegment(speaker_id="SPEAKER_01", start=0.0, end=2.0)]

    captured = {}

    def fake_transcribe(path: Path, segments: list[dict]):
        captured["segments"] = segments  # assert the runner receives its dict shape
        return [{"text": "Bonjour", "start": 0.1, "end": 0.6, "speaker": "SPEAKER_01"}]

    monkeypatch.setattr(gpu_runners, "transcribe_words", fake_transcribe)
    asr.run(ctx)

    assert captured["segments"] == [{"speaker": "SPEAKER_01", "start": 0.0, "end": 2.0}]
    assert len(ctx.words) == 1
    assert ctx.words[0].text == "Bonjour" and ctx.words[0].speaker_id == "SPEAKER_01"
