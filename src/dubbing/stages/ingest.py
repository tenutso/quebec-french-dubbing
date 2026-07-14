"""Stage 1: ingest & probe.

Extracts two audio renditions from the source video:
  * 16 kHz mono WAV for ASR/diarization (what Whisper/pyannote expect)
  * 48 kHz stereo WAV for source separation and final mixing (full fidelity)
and records probed metadata (duration, fps, sample rate) on the context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dubbing import ffmpeg_utils as ff
from dubbing.models import MediaAsset

if TYPE_CHECKING:
    from dubbing.pipeline import Context

ASR_SAMPLE_RATE = 16_000
FULL_SAMPLE_RATE = 48_000


def run(ctx: "Context") -> None:
    job = ctx.job
    info = ff.probe(job.input_path)
    duration = ff.duration_seconds(info)
    fps = ff.video_fps(info)

    ctx.source = MediaAsset(path=job.input_path, duration=duration, fps=fps)

    asr_wav = job.work_dir / "audio.asr.16k.wav"
    ff.extract_audio(job.input_path, asr_wav, sample_rate=ASR_SAMPLE_RATE, channels=1)
    ctx.asr_audio = MediaAsset(
        path=asr_wav, duration=duration, sample_rate=ASR_SAMPLE_RATE, channels=1
    )

    full_wav = job.work_dir / "audio.full.48k.wav"
    ff.extract_audio(job.input_path, full_wav, sample_rate=FULL_SAMPLE_RATE, channels=2)
    ctx.full_audio = MediaAsset(
        path=full_wav, duration=duration, sample_rate=FULL_SAMPLE_RATE, channels=2
    )
