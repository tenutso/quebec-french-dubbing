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
from typing import Callable

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


# A progress callback: (stage_label, completed_stages, total_stages) -> None.
ProgressFn = Callable[[str, int, int], None]


def run(job: Job, progress: ProgressFn | None = None) -> Context:
    """Execute the full pipeline for ``job`` and return the final context.

    ``progress`` is an optional callback invoked as each stage starts, with the
    stage's human label plus (completed, total) counts — used by the web UI to
    render a progress bar. The CLI passes ``None``.

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

    # Build the stage plan for this job's style, so progress totals are accurate.
    # SRT/VTT are always produced (even subtitles-only jobs); separation and the
    # TTS/time-fit/mix trio are conditional on the dub style.
    plan: list[tuple[str, Callable[[Context], object]]] = [("Extracting audio", ingest.run)]
    if job.dub_style is DubStyle.FULL_REPLACEMENT:
        plan.append(("Separating vocals / music", separate.run))
    plan += [
        ("Diarizing speakers", diarize.run),
        ("Transcribing (ASR)", asr.run),
        ("Building cues", cues.run),
        ("Translating to fr-CA", translate.run),
        ("Writing subtitles", subtitles.run),
    ]
    if job.dub_style is not DubStyle.SUBTITLES_ONLY:
        plan += [
            ("Synthesizing voices", tts.run),
            ("Time-fitting audio", timefit.run),
            ("Mixing (EBU R128)", mix.run),
        ]
    plan.append(("Muxing output", mux.run))

    total = len(plan)
    for done, (label, stage) in enumerate(plan):
        if progress is not None:
            progress(label, done, total)
        stage(ctx)
    if progress is not None:
        progress("Done", total, total)

    return ctx
