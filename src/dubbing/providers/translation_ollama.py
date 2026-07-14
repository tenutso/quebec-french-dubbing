"""Ollama-backed EN -> Quebec French translation provider (default, local, free).

Runs a local open-source LLM via Ollama instead of a paid API — the cost-optimized
default. Mistral models are French-native and handle Québec register well; override with
``OLLAMA_MODEL``. Uses Ollama structured outputs (a JSON schema in ``format``) so the
model returns the same two-variant batch shape the pipeline expects.
"""

from __future__ import annotations

import json
import os
from typing import Callable

from dubbing.models import Cue
from dubbing.providers import translation_common as tc
from dubbing.providers.registry import register_translation_provider

DEFAULT_MODEL = "mistral-small"
DEFAULT_HOST = "http://localhost:11434"
# Context must hold the full source transcript + a batch; give the model room.
NUM_CTX = 16384

# A chat transport: (messages, json_schema) -> assistant message content (a JSON string).
ChatFn = Callable[[list[dict], dict], str]


class OllamaTranslation:
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        chat: ChatFn | None = None,
    ) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self._host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self._chat = chat or self._http_chat

    def _http_chat(self, messages: list[dict], schema: dict) -> str:
        import requests

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "format": schema,  # structured output constrained to our batch schema
            "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
        }
        # OLLAMA_KEEP_ALIVE (e.g. "30s", "0") lets the model unload after translation so
        # a subsequent GPU-analysis run isn't starved of VRAM by a resident LLM.
        keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE")
        if keep_alive:
            payload["keep_alive"] = keep_alive
        resp = requests.post(f"{self._host}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def release(self) -> None:
        """Evict the model from GPU memory (keep_alive=0) and wait until it is actually
        unloaded, so a downstream GPU stage (e.g. the local TTS) isn't starved of VRAM.
        Best-effort; ignores transport errors."""
        import time

        try:
            import requests

            requests.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "keep_alive": 0},
                timeout=30,
            )
            # Poll /api/ps until the model is gone (VRAM returned to the driver).
            for _ in range(30):
                ps = requests.get(f"{self._host}/api/ps", timeout=10).json()
                if not any(m.get("name", "").startswith(self._model)
                           for m in ps.get("models", [])):
                    break
                time.sleep(0.5)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def translate(
        self,
        cues: list[Cue],
        *,
        glossary: dict[str, str],
        register: str,
        max_chars_per_cue: dict[int, int],
    ) -> list[Cue]:
        by_index = {c.index: c for c in cues}
        schema = tc.Batch.model_json_schema()
        system = "\n\n".join([
            tc.SYSTEM_INSTRUCTIONS,
            f"Registre demandé: {register}",
            "Glossaire:\n" + tc.glossary_block(glossary),
            "Transcription source complète (contexte):\n" + tc.transcript_context(cues),
        ])

        for start in range(0, len(cues), tc.BATCH_SIZE):
            batch = cues[start : start + tc.BATCH_SIZE]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": tc.batch_user_message(batch, max_chars_per_cue)},
            ]
            content = self._chat(messages, schema)
            parsed = tc.Batch.model_validate(json.loads(content))
            tc.apply_units(by_index, parsed)

        return cues


@register_translation_provider("ollama")
def _factory(**kwargs) -> OllamaTranslation:
    return OllamaTranslation(
        model=kwargs.get("model"),
        host=kwargs.get("host"),
        chat=kwargs.get("chat"),
    )
