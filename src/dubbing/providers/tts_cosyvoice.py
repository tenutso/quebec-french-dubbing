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
import tempfile
from pathlib import Path

from dubbing.providers.registry import register_tts_installer, register_tts_provider
from dubbing.providers.tts import VoiceRef

logger = logging.getLogger(__name__)

# CosyVoice speaks generic French from fr-CA text; the QC register is carried by the text.
_LOCALES = {"fr-CA", "fr-FR", "fr"}

# Where scripts/install_cosyvoice.sh puts the checkout and model by default. The provider
# auto-detects these when COSYVOICE_ROOT / COSYVOICE_MODEL_DIR are unset, so a standard
# install works with no environment setup (the env vars remain overrides for custom paths).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHECKOUT = _REPO_ROOT / "third_party" / "CosyVoice"
_DEFAULT_MODEL_SUBDIR = "pretrained_models/Fun-CosyVoice3-0.5B"


def _resolve_root() -> Path | None:
    """The CosyVoice source checkout: ``COSYVOICE_ROOT``, else the repo's default clone."""
    env = os.environ.get("COSYVOICE_ROOT")
    if env:
        return Path(env).resolve()
    return _DEFAULT_CHECKOUT if _DEFAULT_CHECKOUT.exists() else None


def _resolve_model_dir() -> str:
    """The downloaded model dir: ``COSYVOICE_MODEL_DIR``, else ``<checkout>/<subdir>``."""
    env = os.environ.get("COSYVOICE_MODEL_DIR")
    if env:
        return env
    root = _resolve_root()
    if root is not None:
        return str(root / _DEFAULT_MODEL_SUBDIR)
    return _DEFAULT_MODEL_SUBDIR  # relative last resort (back-compat; needs the right cwd)


class CosyVoiceTTS:
    name = "cosyvoice"
    locale_support = set(_LOCALES)
    supports_cloning = True

    def __init__(self, device: str | None = None, model=None) -> None:
        self._device = device
        self._model = model  # lazy; injectable for tests
        self._sr: int | None = None
        # The model dir and checkout are resolved lazily (see _resolve_*), honoring
        # COSYVOICE_MODEL_DIR / COSYVOICE_ROOT and otherwise the default install location.
        # Optional neutral fr-CA reference clip: overrides the speaker clone for prosody
        # (native French delivery, but loses the original speaker's identity).
        self._fr_ref = os.environ.get("COSYVOICE_FR_REF")
        # CosyVoice can't extract a speaker token from a prompt longer than 30s (hard assert
        # in its frontend), and long prompts don't improve cloning. Trim over-long references
        # to this leading window; override with COSYVOICE_MAX_PROMPT_S.
        self._max_prompt_s = float(os.environ.get("COSYVOICE_MAX_PROMPT_S", 20.0))
        self._prompt_cache: dict[str, str] = {}
        # CosyVoice3's LLM requires the synthesis text to carry an instruct prefix
        # terminated by <|endofprompt|> (it asserts on that token); the marker also makes
        # the frontend treat the line as a single unit (no re-splitting). Override the
        # instruct text with COSYVOICE_INSTRUCT.
        self._instruct = os.environ.get("COSYVOICE_INSTRUCT", "You are a helpful assistant.")

    def _ensure_model(self):
        if self._model is None:
            # CosyVoice is a source checkout, not a pip package: put the checkout root (which
            # holds the `cosyvoice` package) and its bundled Matcha-TTS on sys.path before
            # importing. Falls back to the default clone under third_party/ when unset.
            root = _resolve_root()
            if root is not None:
                for p in (str(root), str(root / "third_party" / "Matcha-TTS")):
                    if p not in sys.path:
                        sys.path.insert(0, p)
            from cosyvoice.cli.cosyvoice import AutoModel

            model_dir = _resolve_model_dir()
            logger.info("loading CosyVoice model from %s", model_dir)
            self._model = AutoModel(model_dir=model_dir)
            _coerce_llm_fp32(self._model)
        if self._sr is None:
            self._sr = int(getattr(self._model, "sample_rate", 24000))
        return self._model

    def _prompt_clip(self, ref: str) -> str:
        """Return a path to a prompt clip no longer than ``_max_prompt_s`` seconds.

        CosyVoice asserts the speaker prompt is <=30s; our clone references (whole speaker
        segments) routinely exceed that. Trim over-long clips to a leading window and cache
        the result per source clip. Best-effort: if the file can't be probed we pass it
        through and let CosyVoice surface any error.
        """
        cached = self._prompt_cache.get(ref)
        if cached is not None and Path(cached).exists():
            return cached

        import torchaudio

        try:
            info = torchaudio.info(ref)
            dur = info.num_frames / float(info.sample_rate)
        except Exception:  # unprobeable (e.g. a test stub); leave it to CosyVoice
            return ref
        if dur <= self._max_prompt_s:
            self._prompt_cache[ref] = ref
            return ref

        wav, sr = torchaudio.load(ref)
        wav = wav[:, : int(self._max_prompt_s * sr)]
        tmpdir = Path(tempfile.gettempdir()) / "cosyvoice_prompts"
        tmpdir.mkdir(parents=True, exist_ok=True)
        trimmed = tmpdir / f"{abs(hash(ref)) & 0xFFFFFFFF:08x}.wav"
        torchaudio.save(str(trimmed), wav, sr)
        logger.info(
            "trimmed CosyVoice prompt %s from %.1fs to %.1fs", ref, dur, self._max_prompt_s
        )
        self._prompt_cache[ref] = str(trimmed)
        return str(trimmed)

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

        # CosyVoice rejects prompt audio >30s; trim over-long references first.
        ref = self._prompt_clip(ref)

        model = self._ensure_model()
        wav = self._synthesize(model, text, ref, target_duration)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), wav, self._sr)
        return out_path

    def _synthesize(self, model, text: str, ref: str, target_duration: float | None):
        """Cross-lingual synth for CosyVoice3.

        The text must carry an instruct prefix ending in ``<|endofprompt|>`` — CosyVoice3's
        LLM asserts on that token and, without it, the generation thread dies and emits no
        speech (a silent clip). If the vocoder still gets too few frames to run its conv
        ("Kernel size can't be greater than actual input size" — very rare once the prefix
        is present), fall back to a short silence so one cue can't abort the whole dub.
        """
        import torch

        model_text = f"{self._instruct}<|endofprompt|>{text.strip()}"
        try:
            parts = [
                j["tts_speech"].detach().cpu()
                for j in model.inference_cross_lingual(model_text, ref, stream=False)
            ]
            return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        except RuntimeError as e:
            if "Kernel size can't be greater" not in str(e):
                raise
            secs = target_duration if target_duration and target_duration > 0 else 0.3
            logger.warning(
                "CosyVoice produced too little audio for %r (%s); emitting %.2fs silence",
                text, e, secs,
            )
            return torch.zeros(1, int(self._sr * secs))


@register_tts_provider("cosyvoice")
def _factory(**kwargs) -> CosyVoiceTTS:
    return CosyVoiceTTS(device=kwargs.get("device"), model=kwargs.get("model"))


def _coerce_llm_fp32(cosyvoice) -> None:
    """Force the CosyVoice LLM to float32 after loading.

    CosyVoice loads its Qwen2 LLM with ``Qwen2ForCausalLM.from_pretrained`` and no
    ``torch_dtype``; the checkpoint's config declares ``bfloat16``. transformers<5 ignored
    that and loaded fp32, but transformers>=5 honors it and loads bf16 — while CosyVoice
    runs the LLM with no autocast (fp16=False), so bf16 weights meet fp32 activations and
    raise "mat1 and mat2 must have the same dtype". Casting the LLM to fp32 restores
    CosyVoice's intended fp32 runtime. Best-effort: layout differences are ignored.
    """
    import torch

    inner = getattr(cosyvoice, "model", None)
    llm = getattr(inner, "llm", None)
    if llm is not None and hasattr(llm, "to"):
        llm.to(torch.float32)
        logger.info("coerced CosyVoice LLM to float32 (transformers>=5 bf16 load)")


def _is_installed() -> bool:
    """True when both the `cosyvoice` source checkout and the model weights are present."""
    root = _resolve_root()
    if root is None or not (root / "cosyvoice" / "__init__.py").exists():
        return False
    model_dir = Path(_resolve_model_dir())
    return model_dir.exists() and any(model_dir.glob("*.pt"))


@register_tts_installer("cosyvoice")
def ensure_installed(report=None) -> None:
    """Make the CosyVoice runtime available, installing it on first use.

    CosyVoice is a from-source package with a ~8 GB model, so it isn't part of
    ``make install``. If the checkout or model is missing, run
    ``scripts/install_cosyvoice.sh`` (clone + additive deps + model download); otherwise
    return immediately. Idempotent, so the web UI can call it before every run. ``report``
    receives short progress lines for a UI to display.
    """
    if _is_installed():
        return

    import subprocess

    script = _REPO_ROOT / "scripts" / "install_cosyvoice.sh"
    if not script.exists():
        raise RuntimeError(f"CosyVoice installer not found at {script}")
    if report is not None:
        report("Installing CosyVoice (first run: clone + ~8 GB model download)…")
    logger.info("installing CosyVoice via %s", script)
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"CosyVoice install failed (exit {proc.returncode}). Last output:\n{tail}"
        )
    if not _is_installed():
        raise RuntimeError(
            "CosyVoice install script finished but the checkout/model are still missing — "
            f"check the output of {script}."
        )
