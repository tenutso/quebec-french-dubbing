"""CosyVoice (Fun-CosyVoice 3.0) TTS provider — local, free, cross-lingual cloning.

CosyVoice's ``inference_cross_lingual`` is a better fit than Chatterbox for our case: the
reference clip supplies the speaker's *timbre* while the target-language text drives the
*prosody/pronunciation*, so cloning an English speaker into French bleeds far less English
prosody. Apache-2.0, runs on the local GPU.

Install is separate (see ``scripts/install_cosyvoice.sh`` / the README): CosyVoice is a
from-source package, and the model (``Fun-CosyVoice3-0.5B``) is downloaded once. Point the
provider at them with ``COSYVOICE_MODEL_DIR`` and ``COSYVOICE_ROOT``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dubbing.providers.registry import register_tts_provider
from dubbing.providers.tts import VoiceRef

logger = logging.getLogger(__name__)

# CosyVoice speaks generic French from fr-CA text; the QC register is carried by the text.
_LOCALES = {"fr-CA", "fr-FR", "fr"}
_DEFAULT_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"


class CosyVoiceTTS:
    name = "cosyvoice"
    locale_support = set(_LOCALES)
    supports_cloning = True

    def __init__(self, device: str | None = None, model=None) -> None:
        self._device = device
        self._model = model  # lazy; injectable for tests
        self._sr: int | None = None
        # Path to the downloaded model dir and the CosyVoice checkout (whose
        # third_party/Matcha-TTS must be importable). See scripts/install_cosyvoice.sh.
        self._model_dir = os.environ.get("COSYVOICE_MODEL_DIR", _DEFAULT_MODEL_DIR)
        self._root = os.environ.get("COSYVOICE_ROOT")
        # Optional neutral fr-CA reference clip: overrides the speaker clone for prosody
        # (native French delivery, but loses the original speaker's identity).
        self._fr_ref = os.environ.get("COSYVOICE_FR_REF")

    def _ensure_model(self):
        if self._model is None:
            # CosyVoice imports its bundled Matcha-TTS via a relative path; make it
            # importable from the checkout before importing the package.
            if self._root:
                matcha = str(Path(self._root) / "third_party" / "Matcha-TTS")
                if matcha not in sys.path:
                    sys.path.append(matcha)
            from cosyvoice.cli.cosyvoice import AutoModel

            logger.info("loading CosyVoice model from %s", self._model_dir)
            self._model = AutoModel(model_dir=self._model_dir)
        if self._sr is None:
            self._sr = int(getattr(self._model, "sample_rate", 24000))
        return self._model

    def register_voice(self, speaker_id: str, samples: list[Path]) -> VoiceRef:
        """No server-side clone: the reference clip *is* the voice (zero-shot at synth)."""
        if not samples:
            raise ValueError(f"no clone reference audio for {speaker_id}")
        return VoiceRef(provider=self.name, voice_id=str(samples[0]), is_clone=True)

    def preset_voice(self, speaker_id: str, index: int) -> VoiceRef:
        # Cross-lingual synthesis needs a reference clip; an empty voice_id falls back to
        # COSYVOICE_FR_REF at synth time (else synthesize raises a clear error).
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

        # Reference: the speaker clone, or a neutral fr-CA clip if configured (overrides).
        ref = voice.voice_id if voice.voice_id and Path(voice.voice_id).exists() else None
        if self._fr_ref and Path(self._fr_ref).exists():
            ref = self._fr_ref
        if not ref:
            raise RuntimeError(
                "CosyVoice cross-lingual synthesis needs a reference clip; none for this "
                "cue and COSYVOICE_FR_REF is unset."
            )

        model = self._ensure_model()
        # Cross-lingual: English-timbre prompt + French text -> French prosody, cloned voice.
        parts = [
            j["tts_speech"].detach().cpu()
            for j in model.inference_cross_lingual(text, ref, stream=False)
        ]
        wav = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), wav, self._sr)
        return out_path


@register_tts_provider("cosyvoice")
def _factory(**kwargs) -> CosyVoiceTTS:
    return CosyVoiceTTS(device=kwargs.get("device"), model=kwargs.get("model"))
