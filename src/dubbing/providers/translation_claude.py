"""Claude-backed EN -> Quebec French translation provider (premium option).

Kept as a pluggable premium alternative to the default local Ollama provider. Uses
adaptive thinking + structured outputs (``messages.parse``); the stable system prompt +
glossary + transcript are prompt-cached across batches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dubbing.models import Cue
from dubbing.providers import translation_common as tc
from dubbing.providers.registry import register_translation_provider

if TYPE_CHECKING:  # avoid importing the SDK at module import time
    import anthropic

MODEL = "claude-opus-4-8"


class ClaudeTranslation:
    name = "claude"

    def __init__(self, client: "anthropic.Anthropic | None" = None) -> None:
        if client is None:
            import anthropic  # imported lazily so the package loads without the SDK

            client = anthropic.Anthropic()
        self._client = client

    def translate(
        self,
        cues: list[Cue],
        *,
        glossary: dict[str, str],
        register: str,
        max_chars_per_cue: dict[int, int],
    ) -> list[Cue]:
        by_index = {c.index: c for c in cues}
        system = [
            {"type": "text", "text": tc.SYSTEM_INSTRUCTIONS},
            {"type": "text", "text": f"Registre demandé: {register}"},
            {"type": "text", "text": "Glossaire:\n" + tc.glossary_block(glossary)},
            {
                "type": "text",
                "text": "Transcription source complète (contexte):\n"
                + tc.transcript_context(cues),
                "cache_control": {"type": "ephemeral"},  # stable prefix -> cache it
            },
        ]

        for start in range(0, len(cues), tc.BATCH_SIZE):
            batch = cues[start : start + tc.BATCH_SIZE]
            user = tc.batch_user_message(batch, max_chars_per_cue)
            resp = self._client.messages.parse(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=tc.Batch,
            )
            tc.apply_units(by_index, resp.parsed_output)

        return cues


@register_translation_provider("claude")
def _factory(**kwargs: object) -> ClaudeTranslation:
    client = kwargs.get("client")
    return ClaudeTranslation(client=client)  # type: ignore[arg-type]
