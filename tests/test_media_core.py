"""End-to-end smoke test for the CPU media core using a synthetic clip.

Generates a tiny video with ffmpeg (no GPU/premium deps), then exercises ingest ->
subtitle authoring -> mux. Skipped automatically if ffmpeg is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pysubs2
import pytest

from dubbing import subtitle_rules as rules
from dubbing.config import load_job  # noqa: F401 (import smoke)
from dubbing.models import Cue, DubStyle, Job
from dubbing.pipeline import Context
from dubbing.stages import ingest, mux, subtitles

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _make_clip(path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=320x240:d=3:r=25",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-shortest", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True, check=True,
    )


def _sample_cues():
    return [
        Cue(index=0, start=0.2, end=2.0, speaker_id="S1",
            source_text="Welcome to this webinar.",
            target_text_sub="Bienvenue dans ce webinaire."),
        Cue(index=1, start=2.1, end=2.95, speaker_id="S2",
            source_text="Let's begin.",
            target_text_sub="Commencons."),
    ]


def test_ingest_probes_and_extracts_audio(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    job = Job(input_path=clip, work_dir=tmp_path / "work",
              dub_style=DubStyle.SUBTITLES_ONLY)
    ctx = Context(job=job)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    ingest.run(ctx)

    assert ctx.source.fps == pytest.approx(25.0, abs=0.5)
    assert ctx.source.duration == pytest.approx(3.0, abs=0.3)
    assert ctx.asr_audio.path.exists() and ctx.asr_audio.sample_rate == 16_000
    assert ctx.full_audio.path.exists() and ctx.full_audio.channels == 2


def test_subtitles_authoring_is_standards_compliant(tmp_path):
    paths = subtitles.write_subtitles(_sample_cues(), tmp_path, "clip")
    assert paths["srt"].exists() and paths["vtt"].exists()

    subs = pysubs2.load(str(paths["srt"]))
    assert len(subs) == 2
    for ev in subs:
        text = ev.text.replace(r"\N", "\n")
        problems = rules.validate(
            text, rules.CueTiming(ev.start / 1000, ev.end / 1000)
        )
        assert problems == [], problems


def test_full_pipeline_subtitles_only(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    job = Job(input_path=clip, work_dir=tmp_path / "work",
              dub_style=DubStyle.SUBTITLES_ONLY)
    ctx = Context(job=job)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    ingest.run(ctx)
    ctx.cues = _sample_cues()
    subtitles.run(ctx)
    mux.run(ctx)

    assert ctx.output_path.exists()
    assert ctx.subtitle_paths["srt"].exists()
