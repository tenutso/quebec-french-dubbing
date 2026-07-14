"""Stage 7: subtitle authoring. fr-CA cues -> SRT + WebVTT (+ optional iTT/TTML).

Uses the subtitle-concise translation variant (``target_text_sub``), applies fr-CA
typography and two-line balancing, and writes standards-compliant sidecar files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pysubs2

from dubbing import subtitle_rules as rules
from dubbing.models import Cue

if TYPE_CHECKING:
    from dubbing.pipeline import Context


def _render_line(cue: Cue) -> str:
    """Typography + two-line wrap for a single cue's on-screen text."""
    text = cue.target_text_sub or cue.source_text
    text = rules.apply_french_typography(text)
    return "\n".join(rules.wrap_two_lines(text))


def build_subtitle_file(cues: list[Cue]) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    for cue in cues:
        subs.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=cue.start),
                end=pysubs2.make_time(s=cue.end),
                text=_render_line(cue).replace("\n", r"\N"),
            )
        )
    return subs


def write_subtitles(cues: list[Cue], out_dir: Path, stem: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subs = build_subtitle_file(cues)
    paths: dict[str, Path] = {}
    for fmt, ext in (("srt", "srt"), ("vtt", "vtt")):
        p = out_dir / f"{stem}.fr-CA.{ext}"
        subs.save(str(p), format_=fmt)
        paths[fmt] = p
    return paths


def run(ctx: "Context") -> None:
    stem = ctx.job.input_path.stem
    ctx.subtitle_paths = write_subtitles(ctx.cues, ctx.job.work_dir, stem)
