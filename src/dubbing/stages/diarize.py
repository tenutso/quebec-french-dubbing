"""Stage 3: speaker diarization (pyannote.audio) on the vocals stem.

Dispatches local vs. Modal via ``DUBBING_GPU_BACKEND``. Fills ``ctx.segments``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dubbing.models import SpeakerSegment

if TYPE_CHECKING:
    from dubbing.pipeline import Context


def run(ctx: "Context") -> None:
    # Diarize the vocals stem when available (cleaner), else the ASR audio.
    audio = ctx.vocals_stem or ctx.asr_audio
    assert audio is not None, "ingest must run before diarize"
    backend = os.environ.get("DUBBING_GPU_BACKEND", "local")

    if backend == "modal":
        from dubbing.modal_app import diarize as diar_fn

        raw = diar_fn.remote(audio.path.read_bytes())
    else:
        from dubbing.gpu_runners import diarize_audio

        raw = diarize_audio(audio.path)

    ctx.segments = [
        SpeakerSegment(speaker_id=s["speaker"], start=s["start"], end=s["end"])
        for s in raw
    ]
