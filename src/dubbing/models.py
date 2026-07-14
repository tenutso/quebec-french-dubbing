"""Core data models shared across all pipeline stages.

These are the typed artifacts that flow between stages. Every stage takes and returns
one of these (or a list of them), which keeps the pipeline runner simple and makes each
stage independently testable.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

TARGET_LOCALE = "fr-CA"


class DubStyle(str, Enum):
    """How the French audio replaces / overlays the original."""

    FULL_REPLACEMENT = "full_replacement"  # remove original speech, keep music/SFX
    VOICE_OVER = "voice_over"  # duck original under French track
    SUBTITLES_ONLY = "subtitles_only"  # no dub


class VoiceStrategy(str, Enum):
    CLONE = "clone"  # clone each original speaker's voice
    PRESET = "preset"  # map each speaker to a preset fr-CA neural voice


class LoudnessTarget(str, Enum):
    """Integrated loudness target for the mastered dub track."""

    WEB = "web"  # -16 LUFS (course/web default)
    BROADCAST = "broadcast"  # -23 LUFS (EBU R128 broadcast)

    @property
    def lufs(self) -> float:
        return -16.0 if self is LoudnessTarget.WEB else -23.0


class ProviderSelection(BaseModel):
    """Which vendors run the two premium stages for a given job."""

    translation: str = "ollama"  # local OSS LLM by default; "claude" for premium
    tts: str = "chatterbox"  # local OSS cloning TTS; "elevenlabs"/"azure" pluggable


class Job(BaseModel):
    """Top-level job spec, typically loaded from config/job.yaml."""

    input_path: Path
    work_dir: Path
    target_locale: str = TARGET_LOCALE
    dub_style: DubStyle = DubStyle.FULL_REPLACEMENT
    voice_strategy: VoiceStrategy = VoiceStrategy.CLONE
    loudness: LoudnessTarget = LoudnessTarget.WEB
    providers: ProviderSelection = Field(default_factory=ProviderSelection)
    burn_in_subtitles: bool = False
    glossary_path: Path | None = None

    @field_validator("target_locale")
    @classmethod
    def _only_fr_ca(cls, v: str) -> str:
        # This pipeline is purpose-built for Quebec French; guard against silent misuse.
        if v != TARGET_LOCALE:
            raise ValueError(f"target_locale must be {TARGET_LOCALE!r}, got {v!r}")
        return v


class MediaAsset(BaseModel):
    """A source or intermediate media file plus probed metadata."""

    path: Path
    duration: float  # seconds
    fps: float | None = None  # None for audio-only assets
    sample_rate: int | None = None
    channels: int | None = None


class Word(BaseModel):
    """A single transcribed word with timing (from WhisperX)."""

    text: str
    start: float
    end: float
    speaker_id: str | None = None


class SpeakerSegment(BaseModel):
    """A contiguous span attributed to one speaker (from diarization)."""

    speaker_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class Cue(BaseModel):
    """A subtitle/dub unit: a timed span of one speaker's speech.

    ``source_text`` is the original transcription. Translation fills the two target
    fields: ``target_text_sub`` (concise, reading-speed compliant) and
    ``target_text_dub`` (spoken register, length-controlled for TTS time-fit).
    """

    index: int
    start: float
    end: float
    speaker_id: str
    source_text: str
    target_text_sub: str | None = None
    target_text_dub: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class VoiceProfile(BaseModel):
    """Binds a diarized speaker to a synthesized voice.

    For cloning providers, ``voice_ref`` is the vendor's clone id created from
    ``sample_paths``. For preset providers it is a catalogue voice id and
    ``sample_paths`` is empty.
    """

    speaker_id: str
    provider: str
    voice_ref: str
    sample_paths: list[Path] = Field(default_factory=list)


class SynthClip(BaseModel):
    """A synthesized French audio clip for one cue."""

    cue_index: int
    wav_path: Path
    native_duration: float  # as returned by TTS
    fitted_duration: float | None = None  # after time-fit to the cue slot
