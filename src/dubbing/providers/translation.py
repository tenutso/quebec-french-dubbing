"""Translation provider interface.

EN -> Quebec French is the highest-impact quality stage, so it is behind a Strategy
interface too (Claude by default, DeepL or others pluggable).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dubbing.models import Cue


@runtime_checkable
class TranslationProvider(Protocol):
    """Strategy interface for machine/LLM translation.

    ``translate`` receives all cues at once so the provider can use full-transcript
    context. It must populate ``target_text_sub`` (concise, reading-speed compliant)
    and ``target_text_dub`` (spoken register, within the per-cue char budget) on each
    returned cue, preserving ``index`` order.

    * ``glossary`` maps source term -> preferred fr-CA rendering (plus do-not-translate).
    * ``register`` is a natural-language style instruction, e.g.
      "registre quebecois soutenu, oral naturel".
    * ``max_chars_per_cue`` maps ``cue.index`` -> soft char budget for the dub variant,
      derived from cue duration so synthesized speech roughly fits its slot.
    """

    name: str

    def translate(
        self,
        cues: list[Cue],
        *,
        glossary: dict[str, str],
        register: str,
        max_chars_per_cue: dict[int, int],
    ) -> list[Cue]: ...
