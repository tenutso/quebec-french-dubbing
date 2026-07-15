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
import re
from pathlib import Path

from dubbing.providers.registry import register_tts_provider
from dubbing.providers.tts import VoiceRef

logger = logging.getLogger(__name__)

# fr-CA -> Chatterbox language id ("fr"); the QC register comes from the text.
_LOCALE_TO_LANG = {"fr-CA": "fr", "fr-FR": "fr", "fr": "fr"}

# Chatterbox's alignment/EOS logic gets unstable on long inputs — it can force an early
# EOS (cutting a sentence) or run away repeating tokens. Synthesizing in shorter chunks
# and concatenating keeps each generation well-behaved. Override with CHATTERBOX_MAX_CHARS.
_DEFAULT_MAX_CHARS = 180
# A short gap inserted between concatenated chunks so words don't butt together.
_CHUNK_GAP_S = 0.06

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def _geni(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _split_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """Split ``text`` into <=``max_chars`` chunks on sentence, then clause, boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    def pack(pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        cur = ""
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if len(p) > max_chars:  # still too long: break on clause boundaries
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks.extend(pack(_CLAUSE_SPLIT.split(p)) if _CLAUSE_SPLIT.search(p)
                              else _hard_wrap(p, max_chars))
            elif not cur:
                cur = p
            elif len(cur) + 1 + len(p) <= max_chars:
                cur = f"{cur} {p}"
            else:
                chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks

    return pack(_SENT_SPLIT.split(text)) or [text]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Last-resort wrap of a long clause with no punctuation, on word boundaries."""
    out, cur = [], ""
    for word in text.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur = f"{cur} {word}"
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


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
        self._max_chars = _geni("CHATTERBOX_MAX_CHARS", _DEFAULT_MAX_CHARS)

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
        import torch
        import torchaudio

        model = self._ensure_model()
        lang = _LOCALE_TO_LANG.get(locale, "fr")
        ref = voice.voice_id if voice.voice_id and Path(voice.voice_id).exists() else None
        # A neutral fr-CA reference overrides the speaker clone for prosody, if provided.
        if self._fr_ref and Path(self._fr_ref).exists():
            ref = self._fr_ref

        # Synthesize in sentence-sized chunks (Chatterbox is unstable on long inputs),
        # then concatenate with a short gap so nothing is cut and words don't run together.
        chunks = _split_text(text, self._max_chars) or [text]
        parts: list = []
        gap = torch.zeros(1, int(self._sr_or_default() * _CHUNK_GAP_S))
        for i, chunk in enumerate(chunks):
            wav = model.generate(
                chunk, language_id=lang, audio_prompt_path=ref,
                exaggeration=self._exaggeration,
                cfg_weight=self._cfg_weight,
                temperature=self._temperature,
            ).detach().cpu()
            if i > 0:
                parts.append(gap)
            parts.append(wav)
        wav = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # wav is a (1, N) float tensor on CPU; torchaudio writes it at the model's rate.
        torchaudio.save(str(out_path), wav, self._sr)
        return out_path

    def _sr_or_default(self) -> int:
        return self._sr or 24000


@register_tts_provider("chatterbox")
def _factory(**kwargs) -> ChatterboxTTS:
    return ChatterboxTTS(device=kwargs.get("device"), model=kwargs.get("model"))
