"""Shared building blocks for LLM translation providers (Claude, Ollama, ...).

Keeps the Quebec-French system instructions, the structured-output schema, and the
per-batch prompt assembly in one place so every provider produces the same two variants
(subtitle-concise + dub-length-fit) from identical instructions.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from dubbing.models import Cue

# Translate in batches so each response stays reliable; the full source transcript is
# provided as context on every batch so register/pronoun choices stay consistent.
BATCH_SIZE = 40

SYSTEM_INSTRUCTIONS = """\
Tu es un traducteur professionnel spécialisé en localisation français canadien (Québec) \
pour du contenu de formation, des webinaires et des vidéos pédagogiques.

Registre: français québécois soutenu mais naturel à l'oral. Évite les tournures \
hexagonales quand un équivalent québécois courant existe. Respecte la terminologie du \
glossaire fourni EXACTEMENT (les termes mappés sur eux-mêmes ne se traduisent pas).

Pour chaque réplique, produis DEUX variantes:
- target_text_sub: sous-titre concis, fidèle, lisible (vitesse de lecture confortable). \
Peut condenser légèrement.
- target_text_dub: version parlée pour doublage, ton oral naturel, qui tient dans la \
durée de la réplique. Respecte le budget de caractères indiqué (max_dub_chars): reformule \
plus court si nécessaire sans perdre le sens.

Conserve le sens, le ton et l'intention du locuteur. Ne traduis jamais les noms propres \
ni les marques. Réponds uniquement avec le JSON demandé."""


class Unit(BaseModel):
    index: int
    target_text_sub: str
    target_text_dub: str


class Batch(BaseModel):
    units: list[Unit]


def glossary_block(glossary: dict[str, str]) -> str:
    if not glossary:
        return "(aucun terme imposé)"
    return "\n".join(f"- {src} -> {tgt}" for src, tgt in sorted(glossary.items()))


def transcript_context(cues: list[Cue]) -> str:
    return "\n".join(f"[{c.index}] {c.source_text}" for c in cues)


def batch_user_message(batch: list[Cue], max_chars_per_cue: dict[int, int]) -> str:
    payload = [
        {
            "index": c.index,
            "speaker": c.speaker_id,
            "duration_s": round(c.duration, 2),
            "max_dub_chars": max_chars_per_cue.get(c.index, 200),
            "source_text": c.source_text,
        }
        for c in batch
    ]
    return (
        "Traduis en français québécois les répliques suivantes (JSON). "
        "Retourne une entrée par index, dans l'ordre.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def apply_units(cues_by_index: dict[int, Cue], batch: Batch) -> None:
    for unit in batch.units:
        cue = cues_by_index.get(unit.index)
        if cue is not None:
            cue.target_text_sub = unit.target_text_sub
            cue.target_text_dub = unit.target_text_dub
