"""Pluggable provider layer.

Two Strategy interfaces (``TTSProvider`` and ``TranslationProvider``) let vendors be
swapped per job via config. Registries map a provider name -> factory so the pipeline
never imports a concrete vendor SDK directly.
"""

from dubbing.providers.registry import (
    get_translation_provider,
    get_tts_provider,
    register_translation_provider,
    register_tts_provider,
)
from dubbing.providers.translation import TranslationProvider
from dubbing.providers.tts import TTSProvider, VoiceRef

# Import concrete providers so their registry decorators run. Each concrete module
# imports its vendor SDK lazily (inside methods), so this stays cheap and does not
# require every vendor SDK to be installed.
from dubbing.providers import translation_ollama as _translation_ollama  # noqa: E402,F401
from dubbing.providers import translation_claude as _translation_claude  # noqa: E402,F401
from dubbing.providers import tts_chatterbox as _tts_chatterbox  # noqa: E402,F401
from dubbing.providers import tts_cosyvoice as _tts_cosyvoice  # noqa: E402,F401
from dubbing.providers import tts_elevenlabs as _tts_elevenlabs  # noqa: E402,F401
from dubbing.providers import tts_azure as _tts_azure  # noqa: E402,F401

__all__ = [
    "TTSProvider",
    "VoiceRef",
    "TranslationProvider",
    "get_tts_provider",
    "get_translation_provider",
    "register_tts_provider",
    "register_translation_provider",
]
