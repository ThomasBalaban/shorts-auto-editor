"""
Centralized model configuration and Gemini client factory.

All Gemini model identifiers and default thinking levels live here so future
model upgrades are a single-file change.

Models are Preview as of this writing; pin explicitly rather than using
`-latest` aliases so behavior doesn't shift under us.
"""

from google import genai
from google.genai import types

from utils.config import get_gemini_api_key


# ──────────────────────────────────────────────────────────────────────────
# Model identifiers
# ──────────────────────────────────────────────────────────────────────────

# Flagship reasoning model — used only for trim analysis, where cut quality
# is the most important factor in the whole pipeline.
# (Newest pro as of 2026-10: `gemini-pro-latest` also resolves to this.)
MODEL_PRO = "gemini-3.1-pro-preview"

# High-speed, high-volume workhorse — used for per-event vision captions and
# the AI Director's conflict-resolution calls.
MODEL_FLASH = "gemini-3.8-flash"


# ──────────────────────────────────────────────────────────────────────────
# Thinking levels  (see Gemini 3 docs)
# ──────────────────────────────────────────────────────────────────────────
#   minimal  — near-zero thinking, max throughput (NOT supported by
#              gemini-3.8-flash — its floor is `low`)
#   low      — simple instructions, chat
#   medium   — balanced
#   high     — deep reasoning, slower, more expensive
#
# The SDK accepts either the string or the enum; we use the enum.

THINKING_TRIM = types.ThinkingLevel.HIGH        # Pro — cut quality is paramount
THINKING_DIRECTOR = types.ThinkingLevel.MEDIUM  # Flash — editorial priority
THINKING_VISION = types.ThinkingLevel.LOW       # Flash — simple captioning


# ──────────────────────────────────────────────────────────────────────────
# Media resolution (vision token budget per image / video frame)
# ──────────────────────────────────────────────────────────────────────────
# For video: low and medium are both 70 tok/frame, high is 280 tok/frame.
# For images: low=280, medium=560, high=1120 tokens.

MEDIA_RES_VISION = types.MediaResolution.MEDIA_RESOLUTION_LOW   # onomatopoeia captions
MEDIA_RES_TRIM = types.MediaResolution.MEDIA_RESOLUTION_HIGH    # trim analysis — details matter


# ──────────────────────────────────────────────────────────────────────────
# Safety — permissive across the board. This is a gaming-content tool and
# we frequently handle profanity, shouting, and in-game violence that would
# otherwise trigger false positives.
# ──────────────────────────────────────────────────────────────────────────

def get_safety_settings():
    """Build the permissive safety settings list used across all callers."""
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]


# ──────────────────────────────────────────────────────────────────────────
# Client factories
# ──────────────────────────────────────────────────────────────────────────
# Two clients, because v1alpha and v1beta have different strengths:
#
#   v1beta  — the stable endpoint. File API URIs resolve correctly here
#             and this is what every Gemini video-understanding doc example
#             targets. Use this for anything that uploads via files.upload()
#             and references the URI in generate_content.
#
#   v1alpha — preview endpoint. Required if you want per-part
#             media_resolution. Use for inline (base64) images where you
#             want fine-grained control over vision token budget.
#
# Do NOT send File API URIs to v1alpha — the file store lives under
# /v1beta/files/..., and cross-endpoint resolution returns the misleading
# "Cannot fetch content from the provided URL" 400 error.

def get_gemini_client() -> genai.Client:
    """
    Standard Gemini client on the v1beta endpoint.

    Use for File API workflows (upload → ACTIVE → reference by URI) and
    anywhere you don't need per-part media_resolution.
    """
    api_key = get_gemini_api_key()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Gemini API key not found or not set in config.json")
    return genai.Client(api_key=api_key)


def get_gemini_client_alpha() -> genai.Client:
    """
    Gemini client pinned to the v1alpha preview endpoint.

    Use ONLY when you need per-part media_resolution on inline images.
    Do not pass File API URIs to this client.
    """
    api_key = get_gemini_api_key()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Gemini API key not found or not set in config.json")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )