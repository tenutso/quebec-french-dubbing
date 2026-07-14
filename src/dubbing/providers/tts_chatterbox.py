"""Chatterbox Multilingual TTS provider — local, free, GPU, with zero-shot cloning.

Resemble AI's open-source Chatterbox (MIT) runs entirely on the local GPU: no API, no
account, no per-minute cost. It supports French and clones a speaker's voice zero-shot
from a short reference clip (``audio_prompt_path``) — so it preserves speaker identity like
a premium cloning vendor while keeping the dub free. Québec register is carried by the
translated text (the model speaks generic French with the reference speaker's timbre).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dubbing.providers.registry import register_tts_provider
from dubbing.providers.tts import VoiceRef

logger = logging.getLogger(__name__)

# fr-CA -> Chatterbox language id ("fr"); the QC register comes from the text.
_LOCALE_TO_LANG = {"fr-CA": "fr", "fr-FR": "fr", "fr": "fr"}


def _genf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class ChatterboxTTS:
    name = "chatterbox"
    locale_support = set(_LOCALE_TO_LANG)
    supports_cloning = True

    def __init__(self, device: str | None = None, model=None) -> None:
        self._device = device
        self._model = model  # lazy: heavy weights load on first synth, injectable for tests
        self._sr: int | None = None
        # Generation controls. cfg_weight is the key cross-lingual lever: lower it to
        # reduce adherence to an English reference's delivery so French prosody surfaces.
        # A neutral fr-CA reference clip (CHATTERBOX_FR_REF) can steer prosody without
        # losing too much identity when blended by the user.
        self._exaggeration = _genf("CHATTERBOX_EXAGGERATION", 0.5)
        # 0.5 is the most natural/stable default; lower it (~0.2-0.3) to reduce an
        # English reference's prosody bleed if you prefer that trade-off.
        self._cfg_weight = _genf("CHATTERBOX_CFG_WEIGHT", 0.5)
        self._temperature = _genf("CHATTERBOX_TEMPERATURE", 0.8)
        self._fr_ref = os.environ.get("CHATTERBOX_FR_REF")  # optional fr reference clip

    def _ensure_model(self):
        if self._model is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        if self._sr is None:
            self._sr = int(getattr(self._model, "sr", 24000))
        return self._model

    def register_voice(self, speaker_id: str, samples: list[Path]) -> VoiceRef:
        """No server-side clone: the reference clip *is* the voice (zero-shot at synth)."""
        if not samples:
            raise ValueError(f"no clone reference audio for {speaker_id}")
        return VoiceRef(provider=self.name, voice_id=str(samples[0]), is_clone=True)

    def preset_voice(self, speaker_id: str, index: int) -> VoiceRef:
        # Empty voice_id -> Chatterbox's built-in default voice (no cloning).
        return VoiceRef(provider=self.name, voice_id="")

    def synthesize(
        self,
        text: str,
        voice: VoiceRef,
        out_path: Path,
        *,
        prev_text: str | None = None,
        next_text: str | None = None,
        target_duration: float | None = None,
        locale: str = "fr-CA",
    ) -> Path:
        import torchaudio

        model = self._ensure_model()
        lang = _LOCALE_TO_LANG.get(locale, "fr")
        ref = voice.voice_id if voice.voice_id and Path(voice.voice_id).exists() else None
        # A neutral fr-CA reference overrides the speaker clone for prosody, if provided.
        if self._fr_ref and Path(self._fr_ref).exists():
            ref = self._fr_ref

        wav = model.generate(
            text, language_id=lang, audio_prompt_path=ref,
            exaggeration=self._exaggeration,
            cfg_weight=self._cfg_weight,
            temperature=self._temperature,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # wav is a (1, N) float tensor (possibly on GPU); torchaudio wants CPU.
        torchaudio.save(str(out_path), wav.detach().cpu(), self._sr)
        return out_path


@register_tts_provider("chatterbox")
def _factory(**kwargs) -> ChatterboxTTS:
    return ChatterboxTTS(device=kwargs.get("device"), model=kwargs.get("model"))
