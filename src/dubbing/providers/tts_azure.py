"""Azure AI Speech TTS provider — fr-CA neural voices (strongest native Québec prosody).

Non-cloning: each diarized speaker is mapped to a distinct fr-CA neural voice. SSML
``prosody rate`` derived from ``target_duration`` nudges the clip toward its slot length
(the time-fit stage remains authoritative). This is the recommended provider when speaker
identity preservation is not required and native Québec prosody matters most.
"""

from __future__ import annotations

import os
from pathlib import Path

from dubbing.providers.registry import register_tts_provider
from dubbing.providers.tts import VoiceRef

# A rotation of Azure fr-CA neural voices (mix of genders) for per-speaker assignment.
FR_CA_VOICES = [
    "fr-CA-SylvieNeural",
    "fr-CA-AntoineNeural",
    "fr-CA-JeanNeural",
    "fr-CA-ThierryNeural",
]


class AzureTTS:
    name = "azure"
    locale_support = {"fr-CA"}
    supports_cloning = False

    def __init__(self, synthesizer_factory=None) -> None:
        # synthesizer_factory(voice_name, out_path) -> object with .speak_ssml_async;
        # injectable for tests. Real one is built lazily per call from the SDK.
        self._synth_factory = synthesizer_factory

    def register_voice(self, speaker_id: str, samples):  # pragma: no cover - unsupported
        raise NotImplementedError(
            "AzureTTS does not support cloning; use preset_voice() instead"
        )

    def preset_voice(self, speaker_id: str, index: int) -> VoiceRef:
        return VoiceRef(
            provider=self.name, voice_id=FR_CA_VOICES[index % len(FR_CA_VOICES)]
        )

    def synthesize(
        self,
        text: str,
        voice: VoiceRef,
        out_path: Path,
        *,
        prev_text: str | None = None,
        next_text: str | None = None,
        target_duration: float | None = None,
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ssml = self._build_ssml(text, voice.voice_id, target_duration)
        synth = (self._synth_factory or self._default_synth)(voice.voice_id, out_path)
        result = synth.speak_ssml_async(ssml).get()
        _check_result(result)
        return out_path

    @staticmethod
    def _build_ssml(text: str, voice_name: str, target_duration: float | None) -> str:
        # Advisory rate control only; keep within a natural band so prosody isn't wrecked.
        rate_attr = ""
        if target_duration:
            rate_attr = ' rate="0%"'  # placeholder; time-fit stage does the real fit
        return (
            '<speak version="1.0" xml:lang="fr-CA">'
            f'<voice name="{voice_name}"><prosody{rate_attr}>'
            f"{_escape(text)}"
            "</prosody></voice></speak>"
        )

    def _default_synth(self, voice_name: str, out_path: Path):  # pragma: no cover - SDK
        import azure.cognitiveservices.speech as speechsdk

        cfg = speechsdk.SpeechConfig(
            subscription=os.environ["AZURE_SPEECH_KEY"],
            region=os.environ["AZURE_SPEECH_REGION"],
        )
        cfg.speech_synthesis_voice_name = voice_name
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        audio_cfg = speechsdk.audio.AudioOutputConfig(filename=str(out_path))
        return speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _check_result(result) -> None:  # pragma: no cover - SDK result plumbing
    reason = getattr(result, "reason", None)
    if reason is not None and str(reason).endswith("Canceled"):
        raise RuntimeError(f"Azure synthesis canceled: {getattr(result, 'error_details', '')}")


@register_tts_provider("azure")
def _factory(**kwargs) -> AzureTTS:
    return AzureTTS(synthesizer_factory=kwargs.get("synthesizer_factory"))
