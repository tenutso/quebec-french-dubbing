"""Tests for cue construction (stage 5)."""

from __future__ import annotations

from dubbing import subtitle_rules as rules
from dubbing.models import Word
from dubbing.stages.cues import (
    MAX_SOURCE_CHARS,
    build_cues,
    dub_char_budget,
)


def W(text, start, end, spk="S1"):
    return Word(text=text, start=start, end=end, speaker_id=spk)


def test_speaker_change_forces_break():
    words = [
        W("Hello", 0.0, 0.4, "S1"),
        W("everyone", 0.4, 0.9, "S1"),
        W("Welcome", 1.0, 1.5, "S2"),
    ]
    cues = build_cues(words)
    assert len(cues) == 2
    assert cues[0].speaker_id == "S1"
    assert cues[1].speaker_id == "S2"


def test_long_pause_forces_break():
    words = [
        W("First", 0.0, 0.4),
        W("phrase", 0.4, 0.9),
        W("second", 1.8, 2.2),  # 0.9s gap > PAUSE_BREAK
    ]
    cues = build_cues(words)
    assert len(cues) == 2


def test_char_budget_forces_break():
    # ~10 words of 9 chars each -> exceeds MAX_SOURCE_CHARS within one speaker/no pause.
    words = [W("wordwords", i * 0.3, i * 0.3 + 0.25) for i in range(12)]
    cues = build_cues(words)
    assert len(cues) >= 2
    for c in cues:
        assert len(c.source_text) <= MAX_SOURCE_CHARS + 9  # last word may push slightly


def test_max_duration_forces_break():
    # Continuous speech spanning ~9s must split before MAX_DURATION (7s).
    words = [W("word", i * 0.9, i * 0.9 + 0.5) for i in range(10)]
    cues = build_cues(words)
    assert all(c.duration <= rules.MAX_DURATION + 1e-6 for c in cues)
    assert len(cues) >= 2


def test_short_cue_extended_to_cps_floor():
    words = [W("Oui", 0.0, 0.2, "S1")]
    cues = build_cues(words)
    assert cues[0].duration >= rules.MIN_DURATION - 1e-6


def test_short_cue_extension_capped_by_next_cue():
    words = [
        W("Oui", 0.0, 0.2, "S1"),
        W("Bonjour", 0.5, 0.9, "S2"),  # forces a second cue starting at 0.5
    ]
    cues = build_cues(words)
    # First cue must not run into the second (min gap preserved).
    assert cues[0].end <= cues[1].start


def test_dub_char_budget_scales_with_duration():
    words = [W("word", 0.0, 4.0)]
    cues = build_cues(words)
    assert dub_char_budget(cues[0]) > dub_char_budget(
        type(cues[0])(
            index=0, start=0.0, end=1.0, speaker_id="S1", source_text="x"
        )
    )
