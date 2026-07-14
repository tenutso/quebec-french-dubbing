"""Tests for the subtitle standards engine (Netflix fr-CA baseline)."""

from __future__ import annotations

import pytest

from dubbing import subtitle_rules as rules
from dubbing.subtitle_rules import CueTiming


def test_cps_and_min_duration():
    text = "a" * 34  # 34 chars
    # At exactly 2s, 34/2 = 17 CPS -> at the limit, not over.
    assert rules.cps(text, 2.0) == pytest.approx(17.0)
    # Min duration to stay <= 17 CPS is 34/17 = 2.0s.
    assert rules.min_duration_for(text) == pytest.approx(2.0)


def test_min_duration_floor():
    # Very short text is floored at the 5/6 s minimum, not len/CPS.
    assert rules.min_duration_for("Oui.") == pytest.approx(rules.MIN_DURATION)


def test_wrap_single_line_when_fits():
    assert rules.wrap_two_lines("Bonjour tout le monde") == ["Bonjour tout le monde"]


def test_wrap_two_balanced_lines():
    text = "Bienvenue dans ce webinaire sur la conception de contenu"
    lines = rules.wrap_two_lines(text)
    assert len(lines) == 2
    assert all(len(ln) <= rules.MAX_LINE_CHARS for ln in lines)


def test_wrap_does_not_strand_function_word():
    # The most *balanced* break here ends line 1 on "la"; the function-word penalty
    # must push the break elsewhere so an article isn't stranded.
    text = "Nous allons explorer la conception pedagogique du cours"
    lines = rules.wrap_two_lines(text)
    assert len(lines) == 2
    assert all(len(ln) <= rules.MAX_LINE_CHARS for ln in lines)
    last_word = lines[0].split()[-1].strip(".,;:!?").lower()
    assert last_word not in rules._NO_BREAK_AFTER


def test_french_typography_guillemets_and_punct():
    out = rules.apply_french_typography('Il a dit "bonjour" vraiment?')
    assert "«" in out and "»" in out
    assert " ?" in out  # NBSP before question mark
    assert "..." not in rules.apply_french_typography("attends...")


def test_validate_clean_cue_has_no_problems():
    text = "\n".join(rules.wrap_two_lines("Bonjour et bienvenue dans ce cours"))
    problems = rules.validate(text, CueTiming(0.0, 3.0))
    assert problems == []


def test_validate_flags_overlong_and_fast():
    text = "x" * 60  # too long for one line and too fast at short duration
    problems = rules.validate(text, CueTiming(0.0, 1.0))
    assert any("chars >" in p for p in problems)
    assert any("CPS" in p for p in problems)


def test_validate_flags_duration_bounds():
    short = rules.validate("Oui", CueTiming(0.0, 0.2))
    assert any("< 0.833s" in p for p in short)
    long = rules.validate("Bonjour", CueTiming(0.0, 9.0))
    assert any("> 7.0s" in p for p in long)


def test_validate_gap_check():
    # Cue ends at 2.0s, next starts at 2.01s; at 25 fps min gap is 2/25 = 0.08s.
    problems = rules.validate(
        "Bonjour", CueTiming(1.0, 2.0), next_start=2.01, fps=25.0
    )
    assert any("gap" in p for p in problems)
