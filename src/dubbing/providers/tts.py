"""TTS provider interface.

Any vendor that supports Quebec French prosody can be plugged in by implementing
``TTSProvider`` and registering it. The pipeline selects one per job by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from dubbing.models import TARGET_LOCALE


class VoiceRef(BaseModel):
    """Opaque handle to a synthesizable voice.

    ``voice_id`` is the vendor-side id (a preset catalogue voice or a created clone).
    ``is_clone`` records how it was obtained so callers/tests can reason about cost.
    """

    provider: str
    voice_id: str
    is_clone: bool = False


@runtime_checkable
class TTSProvider(Protocol):
    """Strategy interface for text-to-speech vendors.

    Contract:
      * ``locale_support`` MUST include ``"fr-CA"`` (validated at job start).
      * If ``supports_cloning`` is False, ``register_voice`` should raise
        ``NotImplementedError`` and callers must supply a preset voice instead.
      * ``synthesize`` returns a path to a mono WAV clip. ``prev_text``/``next_text``
        give cross-cue context for prosody continuity where the vendor supports it;
        ``target_duration`` is an advisory hint (e.g. Azure SSML rate) — the pipeline's
        time-fit stage is the authoritative fit.
    """

    name: str
    locale_support: set[str]
    supports_cloning: bool

    def register_voice(self, speaker_id: str, samples: list[Path]) -> VoiceRef: ...

    def preset_voice(self, speaker_id: str, index: int) -> VoiceRef: ...

    def synthesize(
        self,
        text: str,
        voice: VoiceRef,
        out_path: Path,
        *,
        prev_text: str | None = None,
        next_text: str | None = None,
        target_duration: float | None = None,
    ) -> Path: ...


def assert_supports_target(provider: TTSProvider, locale: str = TARGET_LOCALE) -> None:
    """Fail fast if a selected provider can't produce the target locale."""
    if locale not in provider.locale_support:
        raise ValueError(
            f"TTS provider {provider.name!r} does not support {locale!r}; "
            f"supported: {sorted(provider.locale_support)}"
        )
