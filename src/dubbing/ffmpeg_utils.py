"""Thin ffmpeg/ffprobe helpers used by the CPU media stages."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )
    return proc.stdout


def probe(path: Path) -> dict:
    """Return the ffprobe JSON for ``path``."""
    out = _run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
    return json.loads(out)


def duration_seconds(info: dict) -> float:
    if "format" in info and info["format"].get("duration"):
        return float(info["format"]["duration"])
    for s in info.get("streams", []):
        if s.get("duration"):
            return float(s["duration"])
    raise ValueError("could not determine duration from ffprobe output")


def video_fps(info: dict) -> float | None:
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            num, _, den = s.get("r_frame_rate", "0/0").partition("/")
            try:
                n, d = float(num), float(den)
                return n / d if d else None
            except ValueError:
                return None
    return None


def extract_audio(
    src: Path, dst: Path, *, sample_rate: int, channels: int
) -> Path:
    """Extract a PCM WAV from ``src`` at the given rate/channels."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", str(channels),
            str(dst),
        ]
    )
    return dst


def loudnorm_measure(path: Path) -> dict:
    """Measure integrated loudness with ffmpeg's loudnorm (pass 1). Returns JSON dict."""
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(path), "-af",
            "loudnorm=print_format=json", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # loudnorm prints the JSON block at the end of stderr.
    stderr = proc.stderr
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"could not parse loudnorm output:\n{stderr}")
    return json.loads(stderr[start : end + 1])
