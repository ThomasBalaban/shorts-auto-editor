# utils/config.py
"""API-key loader. Reads from the centralized youtube_hub/config/secrets.json."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HUB_CONFIG = os.path.abspath(
    os.path.join(_HERE, "..", "..", "youtube_hub", "config"))
if _HUB_CONFIG not in sys.path:
    sys.path.insert(0, _HUB_CONFIG)

from shared_secrets import (  # noqa: E402
    get_gemini_api_key as _shared_gemini,
    get_openai_api_key as _shared_openai,
)


def get_gemini_api_key() -> str:
    return _shared_gemini()


def get_openai_api_key() -> str:
    """Return the OpenAI API key used for Whisper transcription."""
    return _shared_openai()
