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
    # Free any GPU the translator held (e.g. local Ollama) before the TTS stage runs.
    release = getattr(provider, "release", None)
    if callable(release):
        release()
