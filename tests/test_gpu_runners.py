"""Unit tests for the pure logic inside the GPU runners (no models required)."""

from __future__ import annotations

from dubbing.gpu_runners import _speaker_at


def test_speaker_at_exact_overlap():
    segs = [
        {"speaker": "S1", "start": 0.0, "end": 2.0},
        {"speaker": "S2", "start": 2.5, "end": 4.0},
    ]
    assert _speaker_at(segs, 1.0) == "S1"
    assert _speaker_at(segs, 3.0) == "S2"


def test_speaker_at_nearest_when_in_gap():
    segs = [
        {"speaker": "S1", "start": 0.0, "end": 2.0},
        {"speaker": "S2", "start": 2.5, "end": 4.0},
    ]
    # 2.2 is in the gap; closest boundary is S1's end (2.0) at 0.2 vs S2 start at 0.3
    assert _speaker_at(segs, 2.2) == "S1"


def test_speaker_at_empty():
    assert _speaker_at([], 1.0) is None
