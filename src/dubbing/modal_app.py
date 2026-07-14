"""Modal app for the on-demand GPU stages (Demucs, pyannote, WhisperX).

Deploy with ``modal deploy src/dubbing/modal_app.py``. Each function takes and returns
bytes/JSON so it can run in an isolated container; the local stage wrappers upload the
audio, call the function, and write results back into the job work dir.

The container installs the OSS GPU stack once (baked into the image), so at run time we
only pay GPU-seconds — the core of the cost strategy.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "demucs>=4.0",
        "pyannote.audio>=3.1",
        "whisperx>=3.1",
        "soundfile>=0.12",
        "numpy>=1.26",
        "torch",
    )
    # Bundle our runner module into the image.
    .add_local_python_source("dubbing")
)

app = modal.App("dubbing-gpu")

# HF token (pyannote) provided as a Modal secret named "huggingface".
GPU = "A10G"


@app.function(image=image, gpu=GPU, timeout=1800, secrets=[modal.Secret.from_name("huggingface")])
def separate(audio_bytes: bytes, name: str) -> dict:
    import tempfile
    from pathlib import Path

    from dubbing.gpu_runners import separate_audio

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / name
        src.write_bytes(audio_bytes)
        vocals, background = separate_audio(src, Path(d) / "sep")
        return {"vocals": vocals.read_bytes(), "background": background.read_bytes()}


@app.function(image=image, gpu=GPU, timeout=1800, secrets=[modal.Secret.from_name("huggingface")])
def diarize(vocals_bytes: bytes) -> list[dict]:
    import tempfile
    from pathlib import Path

    from dubbing.gpu_runners import diarize_audio

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vocals.wav"
        p.write_bytes(vocals_bytes)
        return diarize_audio(p)


@app.function(image=image, gpu=GPU, timeout=1800)
def asr(vocals_bytes: bytes, segments: list[dict]) -> list[dict]:
    import tempfile
    from pathlib import Path

    from dubbing.gpu_runners import transcribe_words

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vocals.wav"
        p.write_bytes(vocals_bytes)
        return transcribe_words(p, segments)
