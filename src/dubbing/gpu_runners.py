"""Open-source GPU model runners: Demucs, pyannote, WhisperX.

These are the free, self-hosted stages. Heavy imports live inside each function so the
package imports on a machine without the GPU stack (e.g. a laptop authoring subtitles).
The same functions run locally or inside the Modal GPU image (see ``modal_app.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

WHISPER_MODEL = "large-v3"


def separate_audio(in_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Split ``in_path`` into (vocals, background) stems with Demucs (htdemucs).

    Returns paths to a vocals-only WAV and a background (music/SFX) WAV. The background
    is the sum of the non-vocal stems, which is what a full-replacement dub mixes under.
    """
    import subprocess
    import sys

    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    # Demucs writes stems under out_dir/htdemucs/<track>/{vocals,drums,bass,other}.wav.
    # Use the current interpreter so the venv's demucs is found (bare "python" may be
    # a different, demucs-less interpreter).
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals",
         "-n", "htdemucs", "-o", str(out_dir), str(in_path)],
        check=True,
    )
    track = in_path.stem
    stem_dir = out_dir / "htdemucs" / track
    vocals = stem_dir / "vocals.wav"
    # With --two-stems vocals, Demucs emits vocals.wav + no_vocals.wav (the background).
    background = stem_dir / "no_vocals.wav"
    if not background.exists():  # fall back to summing individual stems
        import numpy as np

        parts = [sf.read(str(stem_dir / f"{s}.wav")) for s in ("drums", "bass", "other")]
        data = np.sum([p[0] for p in parts], axis=0)
        background = stem_dir / "background.wav"
        sf.write(str(background), data, parts[0][1])
    return vocals, background


DIARIZATION_PIPELINE = "pyannote/speaker-diarization-community-1"


def diarize_audio(vocals_path: Path, hf_token: str | None = None) -> list[dict]:
    """Return speaker segments ``[{speaker, start, end}]`` via pyannote.audio 4.x.

    Model gated on Hugging Face — set ``HF_TOKEN`` and accept the pipeline's conditions.
    Override the pipeline with ``PYANNOTE_PIPELINE`` (e.g. ``pyannote/speaker-diarization-3.1``).
    """
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    name = os.environ.get("PYANNOTE_PIPELINE", DIARIZATION_PIPELINE)
    # pyannote.audio 4.x uses `token=`; older 3.x used `use_auth_token=`.
    try:
        pipeline = Pipeline.from_pretrained(name, token=token)
    except TypeError:  # pragma: no cover - fallback for 3.x
        pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    output = pipeline(str(vocals_path))
    # pyannote.audio 4.x returns a DiarizeOutput wrapper; prefer the exclusive (non-
    # overlapping) diarization for clean word->speaker assignment. 3.x returns an
    # Annotation directly.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    segments = [
        {"speaker": speaker, "start": float(turn.start), "end": float(turn.end)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s["start"])
    del pipeline, output
    free_cuda()
    return segments


def transcribe_words(vocals_path: Path, segments: list[dict]) -> list[dict]:
    """Transcribe + word-align with WhisperX, then attach speaker labels.

    Returns ``[{text, start, end, speaker}]`` word dicts. Language is auto-detected
    (English source expected) and word timestamps come from WhisperX forced alignment.
    """
    import whisperx

    device = "cuda" if _cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(vocals_path))
    result = model.transcribe(audio, batch_size=16)

    align_model, meta = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, meta, audio, device,
        return_char_alignments=False,
    )

    # Attach speakers by overlapping each word midpoint with a diarization segment.
    words: list[dict] = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            mid = (w["start"] + w["end"]) / 2
            speaker = _speaker_at(segments, mid)
            words.append({
                "text": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "speaker": speaker,
            })
    # Release WhisperX/alignment models so downstream GPU users (Ollama) have VRAM.
    del model, align_model
    free_cuda()
    return words


def _speaker_at(segments: list[dict], t: float) -> str | None:
    best = None
    for s in segments:
        if s["start"] <= t <= s["end"]:
            return s["speaker"]
        # nearest fallback
        dist = min(abs(t - s["start"]), abs(t - s["end"]))
        if best is None or dist < best[0]:
            best = (dist, s["speaker"])
    return best[1] if best else None


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def free_cuda() -> None:
    """Release the process's cached CUDA memory so a co-resident model (e.g. the local
    Ollama LLM used for translation) has room. Safe to call on CPU-only hosts."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
