"""Stage 11: mux & package.

Combines the source video with the mastered French audio track and subtitle sidecars
into a deliverable MP4. For subtitles-only jobs it just carries the original audio.
Optionally burns subtitles into the picture.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from dubbing.models import MediaAsset

if TYPE_CHECKING:
    from dubbing.pipeline import Context


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{' '.join(cmd)}\n{proc.stderr}")


def run(ctx: "Context") -> None:
    job = ctx.job
    src = job.input_path
    out = job.work_dir / f"{src.stem}.fr-CA.mp4"

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    dub: MediaAsset | None = ctx.dub_track

    if dub is not None:
        cmd += ["-i", str(dub.path)]

    if job.burn_in_subtitles and ctx.subtitle_paths.get("srt"):
        # Burn the SRT into the video stream.
        srt = str(ctx.subtitle_paths["srt"]).replace(":", r"\:")
        cmd += ["-vf", f"subtitles='{srt}'"]
        vcodec = ["-c:v", "libx264", "-crf", "18"]
    else:
        vcodec = ["-c:v", "copy"]

    if dub is not None:
        # French dub becomes the default/first audio track; keep original as secondary.
        cmd += [
            "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
            *vcodec, "-c:a", "aac", "-b:a", "192k",
            "-metadata:s:a:0", "language=fra", "-disposition:s:a:0", "default",
            "-metadata:s:a:1", "language=eng",
        ]
    else:
        cmd += ["-map", "0", *vcodec, "-c:a", "copy"]

    cmd.append(str(out))
    _run(cmd)
    ctx.output_path = out
