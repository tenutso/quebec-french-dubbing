"""ElevenLabs TTS provider — default, with per-speaker instant voice cloning.

Preserves each original speaker's identity by cloning their voice from isolated vocal
samples, then synthesizes the French dub text. Cross-cue prosody continuity uses the
``previous_text`` / ``next_text`` stitching parameters. The Quebec register is carried by
the translated text (the clone reproduces the speaker's own accent/timbre).
"""

from __future__ import annotations

import os
from pathlib import Path

from dubbing.providers.registry import register_tts_provider
from dubbing.providers.tts import VoiceRef

MODEL = "eleven_multilingual_v2"  # multilingual model; French supported
OUTPUT_FORMAT = "pcm_24000"  # raw PCM we wrap into WAV; avoids mp3 re-decode for time-fit


class ElevenLabsTTS:
    name = "elevenlabs"
    locale_support = {"fr-CA", "fr-FR", "fr"}
    supports_cloning = True

    def __init__(self, client=None) -> None:
        if client is None:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
        self._client = client

    def register_voice(self, speaker_id: str, samples: list[Path]) -> VoiceRef:
        """Create an instant voice clone from the speaker's isolated vocal samples."""
        voice = self._client.voices.ivc.create(
            name=f"dub-{speaker_id}",
            files=[open(p, "rb") for p in samples],
        )
        return VoiceRef(provider=self.name, voice_id=voice.voice_id, is_clone=True)

    def preset_voice(self, speaker_id: str, index: int) -> VoiceRef:
        """Fallback to a configured multilingual preset voice (no cloning)."""
        ids = [v for v in os.environ.get("ELEVENLABS_FR_VOICES", "").split(",") if v]
        if not ids:
            raise RuntimeError(
                "No ELEVENLABS_FR_VOICES configured for preset (non-clone) synthesis"
            )
        return VoiceRef(provider=self.name, voice_id=ids[index % len(ids)])

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
        audio = self._client.text_to_speech.convert(
            voice_id=voice.voice_id,
            model_id=MODEL,
            text=text,
            output_format=OUTPUT_FORMAT,
            previous_text=prev_text,  # cross-cue prosody continuity
            next_text=next_text,
        )
        _write_pcm_wav(b"".join(audio), out_path, sample_rate=24_000)
        return out_path


def _write_pcm_wav(pcm: bytes, out_path: Path, *, sample_rate: int) -> None:
    import wave

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)


@register_tts_provider("elevenlabs")
def _factory(**kwargs) -> ElevenLabsTTS:
    return ElevenLabsTTS(client=kwargs.get("client"))
