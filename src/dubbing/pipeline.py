"""Stage runner.

The pipeline is a linear sequence of typed stages. Each stage is a small function that
reads/writes artifacts under ``job.work_dir`` and returns its primary output. The runner
keeps a ``Context`` of accumulated artifacts so later stages can pull what they need.

GPU stages (separate/diarize/asr) are defined here as thin wrappers; on Modal they are
dispatched to a GPU function (see ``modal_app.py``), and locally they call the same stage
implementations. Keeping the orchestration framework-light (plain Python) is deliberate —
we can graduate to Prefect/Modal DAGs later without rewriting stage logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dubbing.models import (
    Cue,
    DubStyle,
    Job,
    MediaAsset,
    SpeakerSegment,
    SynthClip,
    VoiceProfile,
    Word,
)


@dataclass
class Context:
    """Mutable bag of artifacts produced as the pipeline runs."""

    job: Job
    source: MediaAsset | None = None
    asr_audio: MediaAsset | None = None
    full_audio: MediaAsset | None = None
    vocals_stem: MediaAsset | None = None
    background_stem: MediaAsset | None = None
    segments: list[SpeakerSegment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    voices: dict[str, VoiceProfile] = field(default_factory=dict)
    clips: list[SynthClip] = field(default_factory=list)
    subtitle_paths: dict[str, Path] = field(default_factory=dict)
    dub_track: MediaAsset | None = None
    output_path: Path | None = None


def run(job: Job) -> Context:
    """Execute the full pipeline for ``job`` and return the final context.

    Imports of stage modules are local so that a partial install (e.g. no GPU deps on a
    laptop authoring subtitles) doesn't break import of the package.
    """
    from dubbing.stages import (
        asr,
        cues,
        diarize,
        ingest,
        mix,
        mux,
        separate,
        subtitles,
        timefit,
        translate,
        tts,
    )

    job.work_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(job=job)

    # --- OSS media core + GPU analysis --------------------------------------
    ingest.run(ctx)
    if job.dub_style is DubStyle.FULL_REPLACEMENT:
        separate.run(ctx)  # split vocals vs background so music/SFX survive the dub
    diarize.run(ctx)
    asr.run(ctx)

    # --- Cues + premium stages ----------------------------------------------
    cues.run(ctx)
    translate.run(ctx)
    subtitles.run(ctx)  # SRT/VTT are always produced, even subtitles-only jobs

    if job.dub_style is not DubStyle.SUBTITLES_ONLY:
        tts.run(ctx)
        timefit.run(ctx)
        mix.run(ctx)

    mux.run(ctx)
    return ctx
