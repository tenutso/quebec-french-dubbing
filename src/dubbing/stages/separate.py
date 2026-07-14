"""Stage 2: source separation (Demucs) — vocals vs. background (music/SFX).

Dispatches to the local runner or the Modal GPU function based on
``DUBBING_GPU_BACKEND`` (``local`` | ``modal``, default ``local``). Produces the vocals
stem (for diarization/ASR/cloning) and the background stem (mixed under a full dub).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dubbing.models import MediaAsset

if TYPE_CHECKING:
    from dubbing.pipeline import Context


def run(ctx: "Context") -> None:
    src = ctx.full_audio
    assert src is not None, "ingest must run before separate"
    work = ctx.job.work_dir
    backend = os.environ.get("DUBBING_GPU_BACKEND", "local")

    if backend == "modal":
        from dubbing.modal_app import separate as sep_fn

        res = sep_fn.remote(src.path.read_bytes(), src.path.name)
        vocals = work / "vocals.wav"
        background = work / "background.wav"
        vocals.write_bytes(res["vocals"])
        background.write_bytes(res["background"])
    else:
        from dubbing.gpu_runners import separate_audio

        v, b = separate_audio(src.path, work / "sep")
        vocals, background = v, b

    ctx.vocals_stem = MediaAsset(path=vocals, duration=src.duration)
    ctx.background_stem = MediaAsset(path=background, duration=src.duration)
