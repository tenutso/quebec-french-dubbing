"""Stage 4: ASR + word alignment (WhisperX large-v3) on the vocals stem.

Dispatches local vs. Modal via ``DUBBING_GPU_BACKEND``. Produces word-level timestamps
merged with diarization speaker labels into ``ctx.words``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dubbing.models import Word

if TYPE_CHECKING:
    from dubbing.pipeline import Context


def run(ctx: "Context") -> None:
    audio = ctx.vocals_stem or ctx.asr_audio
    assert audio is not None, "ingest must run before asr"
    # gpu_runners speak the diarization dict shape ({"speaker", "start", "end"}).
    segments = [
        {"speaker": s.speaker_id, "start": s.start, "end": s.end} for s in ctx.segments
    ]
    backend = os.environ.get("DUBBING_GPU_BACKEND", "local")

    if backend == "modal":
        from dubbing.modal_app import asr as asr_fn

        raw = asr_fn.remote(audio.path.read_bytes(), segments)
    else:
        from dubbing.gpu_runners import transcribe_words

        raw = transcribe_words(audio.path, segments)

    ctx.words = [
        Word(text=w["text"], start=w["start"], end=w["end"],
             speaker_id=w.get("speaker"))
        for w in raw
        if w.get("text")
    ]
