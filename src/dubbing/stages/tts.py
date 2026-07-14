"""Stage 8: TTS synthesis via the selected TTSProvider.

Builds one VoiceProfile per diarized speaker (a clone from isolated vocal samples for
cloning providers, or a preset fr-CA neural voice otherwise), then synthesizes a French
audio clip per cue using the dub-text variant. Clones are cached by speaker so re-runs
don't re-clone (cost).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dubbing import providers
from dubbing.models import SynthClip, VoiceProfile, VoiceStrategy
from dubbing.providers.tts import assert_supports_target

if TYPE_CHECKING:
    from dubbing.pipeline import Context

logger = logging.getLogger(__name__)

# Voice-clone sampling: skip tiny segments, then concatenate a speaker's longest clean
# segments into one reference clip up to CLONE_TARGET_SECONDS (more material = better clone).
CLONE_MIN_SEGMENT = 0.8  # seconds — drop sub-second fragments
CLONE_TARGET_SECONDS = 30.0


def _clone_cache_path(work_dir: Path) -> Path:
    return work_dir / "voice_profiles.json"


def _load_cache(work_dir: Path) -> dict[str, VoiceProfile]:
    p = _clone_cache_path(work_dir)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: VoiceProfile.model_validate(v) for k, v in raw.items()}


def _save_cache(work_dir: Path, voices: dict[str, VoiceProfile]) -> None:
    _clone_cache_path(work_dir).write_text(
        json.dumps({k: json.loads(v.model_dump_json()) for k, v in voices.items()})
    )


def _extract_samples(ctx: "Context", speaker_id: str) -> list[Path]:
    """Build one clone reference clip by concatenating the speaker's cleanest segments.

    Real diarized turns are often short (1-3s), so requiring a single long segment yields
    nothing. Instead we take the longest segments (skipping sub-second fragments) up to
    ~30s total and concatenate them into a single WAV — enough material for a good clone.
    Returns ``[]`` if the speaker has no usable audio (caller falls back to a preset voice).
    """
    import subprocess

    stem = ctx.vocals_stem or ctx.asr_audio
    assert stem is not None, "no vocals stem available for cloning"

    segs = sorted(
        (s for s in ctx.segments
         if s.speaker_id == speaker_id and s.duration >= CLONE_MIN_SEGMENT),
        key=lambda s: s.duration, reverse=True,
    )
    chosen, total = [], 0.0
    for s in segs:
        chosen.append(s)
        total += s.duration
        if total >= CLONE_TARGET_SECONDS:
            break
    if not chosen:
        return []

    chosen.sort(key=lambda s: s.start)  # chronological for natural-sounding reference
    parts: list[Path] = []
    for i, s in enumerate(chosen):
        p = ctx.job.work_dir / f"clone_{speaker_id}_{i}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(stem.path), "-ss", str(s.start),
             "-to", str(s.end), "-ac", "1", "-ar", "24000", str(p)],
            capture_output=True, check=True,
        )
        parts.append(p)

    if len(parts) == 1:
        return parts
    # Concatenate the slices into a single reference clip (absolute paths + -safe 0).
    listing = ctx.job.work_dir / f"clone_{speaker_id}_list.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    ref = ctx.job.work_dir / f"clone_{speaker_id}.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing.resolve()),
         "-c", "copy", str(ref.resolve())],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clone-sample concat failed:\n{proc.stderr}")
    return [ref]


def _build_voices(ctx: "Context", provider) -> dict[str, VoiceProfile]:
    speakers = sorted({c.speaker_id for c in ctx.cues})
    cache = _load_cache(ctx.job.work_dir)
    voices: dict[str, VoiceProfile] = {}
    cloning = (
        ctx.job.voice_strategy is VoiceStrategy.CLONE and provider.supports_cloning
    )
    for idx, spk in enumerate(speakers):
        if spk in cache and cache[spk].provider == provider.name:
            voices[spk] = cache[spk]
            continue
        profile: VoiceProfile | None = None
        if cloning:
            samples = _extract_samples(ctx, spk)
            if samples:
                try:
                    ref = provider.register_voice(spk, samples)
                    profile = VoiceProfile(
                        speaker_id=spk, provider=provider.name,
                        voice_ref=ref.voice_id, sample_paths=samples,
                    )
                except Exception as e:  # e.g. plan lacks cloning permission
                    logger.warning(
                        "cloning failed for %s (%s); falling back to a preset voice",
                        spk, e,
                    )
        if profile is None:  # preset path (non-cloning provider or clone fallback)
            ref = provider.preset_voice(spk, idx)
            profile = VoiceProfile(
                speaker_id=spk, provider=provider.name, voice_ref=ref.voice_id,
            )
        voices[spk] = profile
    _save_cache(ctx.job.work_dir, voices)
    return voices


def run(ctx: "Context") -> None:
    provider = providers.get_tts_provider(ctx.job.providers.tts)
    assert_supports_target(provider, ctx.job.target_locale)

    ctx.voices = _build_voices(ctx, provider)

    from dubbing.providers.tts import VoiceRef

    clips: list[SynthClip] = []
    for i, cue in enumerate(ctx.cues):
        profile = ctx.voices[cue.speaker_id]
        voice = VoiceRef(
            provider=profile.provider,
            voice_id=profile.voice_ref,
            is_clone=bool(profile.sample_paths),
        )
        text = cue.target_text_dub or cue.target_text_sub or cue.source_text
        prev_text = ctx.cues[i - 1].target_text_dub if i > 0 else None
        next_text = ctx.cues[i + 1].target_text_dub if i + 1 < len(ctx.cues) else None
        wav = ctx.job.work_dir / f"tts_{cue.index:05d}.wav"
        provider.synthesize(
            text, voice, wav,
            prev_text=prev_text, next_text=next_text,
            target_duration=cue.duration,
        )
        clips.append(
            SynthClip(cue_index=cue.index, wav_path=wav, native_duration=_wav_dur(wav))
        )
    ctx.clips = clips


def _wav_dur(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
