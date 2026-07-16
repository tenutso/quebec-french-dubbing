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

# Optional per-provider "install/prepare the runtime" hooks, keyed by the same provider
# name as the factories. Kept separate from the factories so a caller can make a provider
# ready (download weights, clone a source checkout) WITHOUT constructing it — construction
# for some providers builds an API client or imports a vendor SDK. Providers register an
# installer only if they have setup that isn't covered by `make install`.
_TTS_INSTALLERS: dict[str, Callable[..., None]] = {}
_TRANSLATION_INSTALLERS: dict[str, Callable[..., None]] = {}

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


def register_tts_installer(name: str) -> Callable[[F], F]:
    """Register a ``report=None`` callable that makes the named TTS runtime ready."""

    def deco(fn: F) -> F:
        _TTS_INSTALLERS[name] = fn  # type: ignore[assignment]
        return fn

    return deco


def register_translation_installer(name: str) -> Callable[[F], F]:
    """Register a ``report=None`` callable that makes the named translation runtime ready."""

    def deco(fn: F) -> F:
        _TRANSLATION_INSTALLERS[name] = fn  # type: ignore[assignment]
        return fn

    return deco


def ensure_tts_ready(name: str, report: Callable[[str], None] | None = None) -> None:
    """Run the installer for ``name`` if one is registered; a no-op otherwise.

    ``report`` receives short human-readable progress lines (e.g. for a UI). Installers
    are idempotent — safe to call before every run.
    """
    fn = _TTS_INSTALLERS.get(name)
    if fn is not None:
        fn(report=report)


def ensure_translation_ready(name: str, report: Callable[[str], None] | None = None) -> None:
    """Run the installer for ``name`` if one is registered; a no-op otherwise."""
    fn = _TRANSLATION_INSTALLERS.get(name)
    if fn is not None:
        fn(report=report)


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
