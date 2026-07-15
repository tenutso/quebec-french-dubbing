"""Stage 6: translation EN -> fr-CA via the selected TranslationProvider.

Builds the per-cue dub character budget (from cue duration) and the glossary, resolves
the configured provider from the registry, and fills the two target-text fields on each
cue in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dubbing.config import load_glossary
from dubbing.stages.cues import dub_char_budget

if TYPE_CHECKING:
    from dubbing.pipeline import Context

REGISTER = "registre québécois soutenu, oral naturel"

# The dub char budget is what fits the cue slot at a natural rate. Time-fit can still
# absorb a bit more (it compresses ~15% and borrows adjacent silence), so only fall
# back to the shorter variant when the dub line overshoots by more than this.
DUB_BUDGET_TOLERANCE = 1.30


def _fit_dub_to_budget(cues, budgets: dict[int, int]) -> None:
    """Deterministic length guard against clips that overflow their cue slot.

    LLMs (especially small local ones) routinely ignore the per-cue ``max_dub_chars``
    hint, so a dub line can be far longer than its slot — which then gets its tail
    chopped at time-fit. When the spoken variant overshoots its budget and the concise
    subtitle variant is shorter, use the subtitle text for the dub: it is a faithful
    condensed rendering the model already produced, so the dub fits without losing words.
    """
    for cue in cues:
        dub = cue.target_text_dub
        sub = cue.target_text_sub
        if not dub:
            continue
        ceiling = budgets.get(cue.index, 200) * DUB_BUDGET_TOLERANCE
        if len(dub) > ceiling and sub and len(sub) < len(dub):
            cue.target_text_dub = sub


def run(ctx: "Context") -> None:
    from dubbing import providers  # ensures concrete providers self-register

    if not ctx.cues:
        return

    glossary = load_glossary(ctx.job.glossary_path)
    budgets = {c.index: dub_char_budget(c) for c in ctx.cues}

    provider = providers.get_translation_provider(ctx.job.providers.translation)
    ctx.cues = provider.translate(
        ctx.cues,
        glossary=glossary,
        register=REGISTER,
        max_chars_per_cue=budgets,
    )
    _fit_dub_to_budget(ctx.cues, budgets)
    # Free any GPU the translator held (e.g. local Ollama) before the TTS stage runs.
    release = getattr(provider, "release", None)
    if callable(release):
        release()
