"""Contract tests for the pluggable provider layer."""

from __future__ import annotations

import pytest

from dubbing import providers
from dubbing.models import Cue
from dubbing.providers.tts import TTSProvider, VoiceRef, assert_supports_target


def test_registered_tts_providers_declare_fr_ca():
    for name in ("chatterbox", "cosyvoice", "elevenlabs", "azure"):
        p = providers.get_tts_provider(
            name, client=object(), synthesizer_factory=lambda *a: None, model=object()
        )
        assert isinstance(p, TTSProvider)  # runtime Protocol check
        assert "fr-CA" in p.locale_support
        assert_supports_target(p)  # must not raise


def test_chatterbox_is_default_and_supports_cloning():
    from dubbing.models import ProviderSelection

    assert ProviderSelection().tts == "chatterbox"
    p = providers.get_tts_provider("chatterbox", model=object())
    assert p.supports_cloning is True
    # The reference clip is the "voice" (zero-shot); empty ref -> built-in default voice.
    assert p.register_voice("S1", [__import__("pathlib").Path("ref.wav")]).is_clone
    assert p.preset_voice("S1", 0).voice_id == ""


def test_chatterbox_uses_from_local_when_model_dir_set(monkeypatch):
    """CHATTERBOX_MODEL_DIR routes model loading to from_local (a fine-tuned checkpoint);
    otherwise the pretrained base is used. Verified without loading real weights."""
    mtl = pytest.importorskip("chatterbox.mtl_tts")
    from dubbing.providers.tts_chatterbox import ChatterboxTTS

    calls: dict[str, object] = {}

    class _FakeModel:
        sr = 24000

    class _FakeMTL:
        @classmethod
        def from_local(cls, ckpt_dir, device):
            calls["from_local"] = (ckpt_dir, device)
            return _FakeModel()

        @classmethod
        def from_pretrained(cls, device):
            calls["from_pretrained"] = device
            return _FakeModel()

    monkeypatch.setattr(mtl, "ChatterboxMultilingualTTS", _FakeMTL)

    # Unset -> pretrained base.
    monkeypatch.delenv("CHATTERBOX_MODEL_DIR", raising=False)
    ChatterboxTTS(device="cpu")._ensure_model()
    assert "from_pretrained" in calls and "from_local" not in calls

    # Set -> from_local with that directory.
    calls.clear()
    monkeypatch.setenv("CHATTERBOX_MODEL_DIR", "/tmp/qc-ckpt")
    ChatterboxTTS(device="cpu")._ensure_model()
    assert calls.get("from_local") == ("/tmp/qc-ckpt", "cpu")
    assert "from_pretrained" not in calls


class _FakeCosy:
    """Stand-in for CosyVoice's AutoModel; records the cross-lingual call."""

    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def inference_cross_lingual(self, text, prompt, stream=False):
        import torch

        self.calls.append((text, prompt, stream))
        yield {"tts_speech": torch.zeros(1, 2400)}  # two chunks -> exercise concat
        yield {"tts_speech": torch.zeros(1, 1200)}


def test_cosyvoice_cross_lingual_synthesis(tmp_path):
    import wave

    from dubbing.providers.tts import VoiceRef
    from dubbing.providers.tts_cosyvoice import CosyVoiceTTS

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")  # only existence is checked; the model call is faked
    fake = _FakeCosy()
    provider = CosyVoiceTTS(model=fake)
    out = tmp_path / "o.wav"

    provider.synthesize(
        "Bonjour le monde",
        VoiceRef(provider="cosyvoice", voice_id=str(ref), is_clone=True),
        out,
        locale="fr-CA",
    )

    # CosyVoice3 requires an instruct prefix ending in <|endofprompt|> ahead of the text,
    # else its LLM asserts and yields a silent clip. Reference is the filepath, not a tensor.
    assert fake.calls == [
        ("You are a helpful assistant.<|endofprompt|>Bonjour le monde", str(ref), False)
    ]
    # Both yielded chunks were concatenated (2400 + 1200) at the model sample rate.
    assert out.exists()
    with wave.open(str(out)) as w:
        assert w.getnframes() == 3600 and w.getframerate() == 24000


class _KernelFailCosy:
    """Always raises CosyVoice's too-few-frames vocoder error (to exercise the fallback)."""

    sample_rate = 24000

    def inference_cross_lingual(self, text, prompt, stream=False):
        raise RuntimeError(
            "Calculated padded input size per channel: (3). Kernel size: (4). "
            "Kernel size can't be greater than actual input size"
        )
        yield  # pragma: no cover — makes this a generator


def test_cosyvoice_vocoder_underrun_falls_back_to_silence(tmp_path):
    """A cue the vocoder can't render emits silence sized to the cue, not a crash."""
    import wave

    from dubbing.providers.tts import VoiceRef
    from dubbing.providers.tts_cosyvoice import CosyVoiceTTS

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    provider = CosyVoiceTTS(model=_KernelFailCosy())
    out = tmp_path / "o.wav"

    provider.synthesize(
        "Oh", VoiceRef(provider="cosyvoice", voice_id=str(ref), is_clone=True), out,
        target_duration=0.5,
    )
    assert out.exists()
    with wave.open(str(out)) as w:
        assert w.getnframes() == int(24000 * 0.5) and w.getframerate() == 24000


def test_cosyvoice_trims_overlong_prompt(tmp_path):
    """CosyVoice asserts prompt audio <=30s; over-long clone refs must be trimmed before
    being handed to the model (else it crashes mid-synth)."""
    import torch
    import torchaudio

    from dubbing.providers.tts import VoiceRef
    from dubbing.providers.tts_cosyvoice import CosyVoiceTTS

    ref = tmp_path / "long_ref.wav"
    torchaudio.save(str(ref), torch.zeros(1, 16000 * 3), 16000)  # 3s clip

    fake = _FakeCosy()
    provider = CosyVoiceTTS(model=fake)
    provider._max_prompt_s = 1.0  # force a trim
    provider.synthesize(
        "Bonjour", VoiceRef(provider="cosyvoice", voice_id=str(ref), is_clone=True),
        tmp_path / "o.wav", locale="fr-CA",
    )

    used_prompt = fake.calls[0][1]
    assert used_prompt != str(ref)  # a trimmed copy, not the original
    info = torchaudio.info(used_prompt)
    assert info.num_frames / info.sample_rate <= 1.0 + 1e-3


def test_cosyvoice_resolves_paths_from_root_when_model_dir_unset(tmp_path, monkeypatch):
    """The reported bug: with COSYVOICE_ROOT/MODEL_DIR unset the import/model paths must
    still resolve. COSYVOICE_ROOT alone should derive the model dir under it."""
    from dubbing.providers import tts_cosyvoice as cv

    monkeypatch.delenv("COSYVOICE_MODEL_DIR", raising=False)
    monkeypatch.setenv("COSYVOICE_ROOT", str(tmp_path))
    assert cv._resolve_root() == tmp_path.resolve()
    assert cv._resolve_model_dir() == str(tmp_path / "pretrained_models/Fun-CosyVoice3-0.5B")

    # An explicit COSYVOICE_MODEL_DIR wins over the derived path.
    monkeypatch.setenv("COSYVOICE_MODEL_DIR", "/models/cv")
    assert cv._resolve_model_dir() == "/models/cv"


def test_cosyvoice_installer_registered_and_noop_when_installed(monkeypatch):
    """`cosyvoice` registers an installer; it must not shell out when already installed."""
    import subprocess

    from dubbing import providers
    from dubbing.providers import tts_cosyvoice as cv

    assert "cosyvoice" in providers.registry._TTS_INSTALLERS
    monkeypatch.setattr(cv, "_is_installed", lambda: True)
    # If the no-op short-circuit regresses, this makes the errant shell-out fail loudly.
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("installer must not shell out when already installed"),
    )
    providers.ensure_tts_ready("cosyvoice")  # no-op, must not raise


def test_ensure_ready_is_noop_for_providers_without_installer():
    """Providers that ship with `make install` have no installer -> a safe no-op."""
    from dubbing import providers

    providers.ensure_tts_ready("chatterbox")  # no installer registered
    providers.ensure_tts_ready("elevenlabs")  # must not construct the provider
    providers.ensure_translation_ready("ollama")


def test_cosyvoice_requires_a_reference(tmp_path):
    from dubbing.providers.tts import VoiceRef
    from dubbing.providers.tts_cosyvoice import CosyVoiceTTS

    provider = CosyVoiceTTS(model=_FakeCosy())
    # No clone reference and no COSYVOICE_FR_REF -> a clear error, not a crash mid-synth.
    with pytest.raises(RuntimeError, match="reference clip"):
        provider.synthesize(
            "Bonjour", VoiceRef(provider="cosyvoice", voice_id=""), tmp_path / "o.wav"
        )


def test_provider_without_fr_ca_is_rejected():
    class Bad:
        name = "bad"
        locale_support = {"en-US"}
        supports_cloning = False

    with pytest.raises(ValueError, match="does not support 'fr-CA'"):
        assert_supports_target(Bad())


def test_azure_is_non_cloning_and_rotates_voices():
    az = providers.get_tts_provider("azure")
    assert az.supports_cloning is False
    with pytest.raises(NotImplementedError):
        az.register_voice("S1", [])
    # distinct speakers get distinct voices, wrapping around the catalogue
    v0 = az.preset_voice("S1", 0).voice_id
    v1 = az.preset_voice("S2", 1).voice_id
    assert v0 != v1


def test_unknown_provider_raises():
    with pytest.raises(KeyError):
        providers.get_tts_provider("does-not-exist")


class _FakeClaude:
    """Minimal fake of anthropic.Anthropic().messages.parse for translation tests."""

    def __init__(self):
        self.messages = self
        self.calls = 0

    def parse(self, *, model, output_format, messages, **kwargs):
        import json
        import re

        self.calls += 1
        # Echo back a translation for every index present in the user payload.
        text = messages[0]["content"]
        indices = [int(i) for i in re.findall(r'"index":\s*(\d+)', text)]
        units = [
            {"index": i, "target_text_sub": f"sub-{i}", "target_text_dub": f"dub-{i}"}
            for i in indices
        ]

        class R:
            parsed_output = output_format.model_validate({"units": units})

        return R()


def test_claude_translation_fills_both_variants():
    cues = [
        Cue(index=0, start=0, end=2, speaker_id="S1", source_text="Hello."),
        Cue(index=1, start=2, end=4, speaker_id="S2", source_text="Welcome."),
    ]
    tr = providers.get_translation_provider("claude", client=_FakeClaude())
    out = tr.translate(cues, glossary={"email": "courriel"}, register="québécois",
                       max_chars_per_cue={0: 30, 1: 30})
    assert out[0].target_text_sub == "sub-0" and out[0].target_text_dub == "dub-0"
    assert out[1].target_text_sub == "sub-1" and out[1].target_text_dub == "dub-1"


def test_ollama_translation_fills_both_variants():
    import json
    import re

    from dubbing.providers import translation_common as tc

    def fake_chat(messages, schema):
        # Echo a translation for each index present in the user message.
        indices = [int(i) for i in re.findall(r'"index":\s*(\d+)', messages[-1]["content"])]
        units = [
            {"index": i, "target_text_sub": f"qc-{i}", "target_text_dub": f"qc-{i}"}
            for i in indices
        ]
        assert schema == tc.Batch.model_json_schema()  # structured-output schema wired
        return json.dumps({"units": units})

    cues = [
        Cue(index=0, start=0, end=2, speaker_id="S1", source_text="Hello."),
        Cue(index=1, start=2, end=4, speaker_id="S2", source_text="Welcome."),
    ]
    tr = providers.get_translation_provider("ollama", chat=fake_chat)
    out = tr.translate(cues, glossary={"email": "courriel"}, register="québécois",
                       max_chars_per_cue={0: 30, 1: 30})
    assert out[0].target_text_dub == "qc-0" and out[1].target_text_sub == "qc-1"


def test_ollama_is_the_default_translation_provider():
    from dubbing.models import ProviderSelection

    assert ProviderSelection().translation == "ollama"


def test_claude_translation_batches_large_input():
    from dubbing.providers.translation_common import BATCH_SIZE

    cues = [
        Cue(index=i, start=i, end=i + 1, speaker_id="S1", source_text=f"line {i}")
        for i in range(BATCH_SIZE + 5)
    ]
    fake = _FakeClaude()
    tr = providers.get_translation_provider("claude", client=fake)
    out = tr.translate(cues, glossary={}, register="q",
                       max_chars_per_cue={c.index: 20 for c in cues})
    assert fake.calls == 2  # split into two batches
    assert all(c.target_text_dub for c in out)
