"""Name -> factory registries for pluggable providers.

Concrete provider modules register themselves via the decorators. The pipeline resolves
a provider by the name given in the job config, so it never imports a vendor SDK
directly and unused SDKs need not be installed.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from dubbing.providers.translation import TranslationProvider
from dubbing.providers.tts import TTSProvider

_TTS_FACTORIES: dict[str, Callable[..., TTSProvider]] = {}
_TRANSLATION_FACTORIES: dict[str, Callable[..., TranslationProvider]] = {}

F = TypeVar("F", bound=Callable[..., object])


def register_tts_provider(name: str) -> Callable[[F], F]:
    def deco(factory: F) -> F:
        _TTS_FACTORIES[name] = factory  # type: ignore[assignment]
        return factory

    return deco


def register_translation_provider(name: str) -> Callable[[F], F]:
    def deco(factory: F) -> F:
        _TRANSLATION_FACTORIES[name] = factory  # type: ignore[assignment]
        return factory

    return deco


def get_tts_provider(name: str, **kwargs: object) -> TTSProvider:
    if name not in _TTS_FACTORIES:
        raise KeyError(
            f"Unknown TTS provider {name!r}. Registered: {sorted(_TTS_FACTORIES)}"
        )
    return _TTS_FACTORIES[name](**kwargs)


def get_translation_provider(name: str, **kwargs: object) -> TranslationProvider:
    if name not in _TRANSLATION_FACTORIES:
        raise KeyError(
            f"Unknown translation provider {name!r}. "
            f"Registered: {sorted(_TRANSLATION_FACTORIES)}"
        )
    return _TRANSLATION_FACTORIES[name](**kwargs)
