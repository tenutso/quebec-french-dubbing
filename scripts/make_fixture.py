"""Generate a tiny synthetic 2-speaker clip for local pipeline smoke tests.

This is a placeholder (tones over a color card), enough to exercise ingest/mux/mix. For a
real end-to-end run (Demucs/pyannote/WhisperX + premium TTS) supply an actual webinar clip
and point config/job.yaml at it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_2spk.mp4"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Two tone bursts (proxy for two speakers) over a color card.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=640x360:d=4:r=25",
            "-f", "lavfi", "-i",
            "sine=frequency=240:duration=1.5,adelay=200|200,apad=pad_dur=2.5",
            "-shortest", "-pix_fmt", "yuv420p", str(OUT),
        ],
        check=True,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
