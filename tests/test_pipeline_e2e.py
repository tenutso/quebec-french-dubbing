"""End-to-end orchestration test for pipeline.run().

The GPU stages (separate/diarize/asr) and premium providers (translation/TTS) are
replaced with lightweight fakes so the *wiring* — ingest -> cues -> subtitles -> tts ->
time-fit -> mix -> mux — runs for real through ffmpeg and produces real deliverables.
This is the harness the plan's `make sample` uses; swapping the fakes for real backends
is the only change needed for a production run.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from dubbing import pipeline, providers
from dubbing.config import load_glossary  # noqa: F401 (import smoke)
from dubbing.models import (
    Job,
    MediaAsset,
    ProviderSelection,
    SpeakerSegment,
    VoiceStrategy,
    Word,
)
from dubbing.providers.registry import (
    register_translation_provider,
    register_tts_provider,
)
from dubbing.providers.tts import VoiceRef

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


# --- Fake premium providers ------------------------------------------------
class _FakeTTS:
    name = "faketts"
    locale_support = {"fr-CA"}
    supports_cloning = True

    def register_voice(self, speaker_id, samples):
        return VoiceRef(provider=self.name, voice_id=f"clone-{speaker_id}", is_clone=True)

    def preset_voice(self, speaker_id, index):
        return VoiceRef(provider=self.name, voice_id=f"preset-{index}")

    def synthesize(self, text, voice, out_path, *, prev_text=None, next_text=None,
                   target_duration=None):
        # Emit a short tone; time-fit will stretch/pad it to the cue length.
        sr, dur = 24000, 0.7
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sf.write(str(out_path), (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr)
        return out_path


class _FakeTranslate:
    name = "faketrans"

    def translate(self, cues, *, glossary, register, max_chars_per_cue):
        for c in cues:
            c.target_text_sub = f"[fr] {c.source_text}"
            c.target_text_dub = f"[fr] {c.source_text}"
        return cues


@pytest.fixture(autouse=True)
def _register_fakes():
    register_tts_provider("faketts")(lambda **k: _FakeTTS())
    register_translation_provider("faketrans")(lambda **k: _FakeTranslate())


# --- Fake GPU stages via monkeypatch --------------------------------------
def _fake_gpu(monkeypatch):
    from dubbing.stages import asr, diarize, separate

    def fake_separate(ctx):
        # Reuse the full audio as both stems (no real separation in the test).
        ctx.vocals_stem = MediaAsset(path=ctx.full_audio.path, duration=ctx.source.duration)
        ctx.background_stem = MediaAsset(path=ctx.full_audio.path, duration=ctx.source.duration)

    def fake_diarize(ctx):
        ctx.segments = [
            SpeakerSegment(speaker_id="S1", start=0.2, end=1.4),
            SpeakerSegment(speaker_id="S2", start=1.6, end=2.8),
        ]

    def fake_asr(ctx):
        ctx.words = [
            Word(text="Welcome", start=0.2, end=0.7, speaker_id="S1"),
            Word(text="everyone.", start=0.7, end=1.4, speaker_id="S1"),
            Word(text="Let's", start=1.6, end=2.0, speaker_id="S2"),
            Word(text="begin.", start=2.0, end=2.8, speaker_id="S2"),
        ]

    monkeypatch.setattr(separate, "run", fake_separate)
    monkeypatch.setattr(diarize, "run", fake_diarize)
    monkeypatch.setattr(asr, "run", fake_asr)


def _make_clip(path):
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=3:r=25",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=3",
         "-shortest", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def test_full_pipeline_produces_dub_and_subtitles(tmp_path, monkeypatch):
    clip = tmp_path / "webinar.mp4"
    _make_clip(clip)
    _fake_gpu(monkeypatch)

    job = Job(
        input_path=clip,
        work_dir=tmp_path / "work",
        voice_strategy=VoiceStrategy.CLONE,
        providers=ProviderSelection(translation="faketrans", tts="faketts"),
    )

    ctx = pipeline.run(job)

    # Deliverables exist.
    assert ctx.output_path.exists()
    assert ctx.subtitle_paths["srt"].exists()
    assert ctx.subtitle_paths["vtt"].exists()
    assert ctx.dub_track is not None and ctx.dub_track.path.exists()

    # Cues were built and translated.
    assert len(ctx.cues) >= 2
    assert all(c.target_text_dub for c in ctx.cues)
    # Two speakers -> two cloned voice profiles.
    assert set(ctx.voices) == {"S1", "S2"}
    assert all(v.provider == "faketts" for v in ctx.voices.values())

    # Output MP4 carries a French audio track.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream_tags=language", "-of", "csv=p=0", str(ctx.output_path)],
        capture_output=True, text=True, check=True,
    )
    assert "fra" in probe.stdout


def test_provider_swap_to_azure_skips_cloning(tmp_path, monkeypatch):
    clip = tmp_path / "webinar.mp4"
    _make_clip(clip)
    _fake_gpu(monkeypatch)

    # Azure is non-cloning + fr-CA: pipeline must use preset voices, not clones.
    job = Job(
        input_path=clip,
        work_dir=tmp_path / "work",
        voice_strategy=VoiceStrategy.CLONE,  # requested, but Azure can't clone
        providers=ProviderSelection(translation="faketrans", tts="azure"),
    )

    # Patch Azure synth to write a tone instead of calling the cloud.
    from dubbing.providers import tts_azure

    def fake_factory(voice_name, out_path):
        class _S:
            def speak_ssml_async(self, ssml):
                class _R:
                    def get(self_inner):
                        sr, dur = 24000, 0.6
                        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                        sf.write(str(out_path),
                                 (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
                        return object()
                return _R()
        return _S()

    monkeypatch.setattr(
        providers, "get_tts_provider",
        lambda name, **k: tts_azure.AzureTTS(synthesizer_factory=fake_factory)
        if name == "azure" else providers.get_tts_provider(name, **k),
    )

    ctx = pipeline.run(job)
    assert ctx.output_path.exists()
    # Preset (non-clone) voices were assigned.
    assert all(not v.sample_paths for v in ctx.voices.values())
    assert all(v.voice_ref.startswith("fr-CA-") for v in ctx.voices.values())
