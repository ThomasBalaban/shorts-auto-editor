# core/transcriber.py
"""
Transcription using OpenAI Whisper API for accurate word-level timestamps.
Falls back to faster-whisper locally if the API is unavailable.
"""

import os
import time
import subprocess
import tempfile
import numpy as np
from typing import List, Tuple

from utils.timestamp_processor import apply_duration_adjustments, fix_overlapping_timestamps
from utils.hallucination_filter import filter_hallucinations

# ── OpenAI client (primary) ──────────────────────────────────────────────────
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False
    print("⚠️  [transcriber] 'openai' package not installed — "
          "run: pip install openai   (will use local Whisper instead)")

# ── Local Whisper fallback ────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    _local_whisper_available = True
except ImportError:
    _local_whisper_available = False

# ── WhisperX alignment (only used in fallback path) ──────────────────────────
try:
    from whisperx import load_align_model, align
    _whisperx_available = True
except ImportError:
    _whisperx_available = False


def _get_openai_client(log_func=None):
    """Initialise OpenAI client from config. Logs the failure reason if it can't."""
    _log = log_func or print
    try:
        from utils.config import get_openai_api_key
        key = get_openai_api_key()
        if not key:
            _log("⚠️  OPENAI_API_KEY is empty in config.json — falling back to local Whisper")
            return None
        return OpenAI(api_key=key)
    except FileNotFoundError:
        _log("⚠️  config.json not found — falling back to local Whisper")
        return None
    except Exception as e:
        _log(f"⚠️  Could not load OpenAI client: {e} — falling back to local Whisper")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_audio(input_file, output_file, track_index, log_func):
    """Extract a single audio track to a 16 kHz mono WAV."""
    try:
        ffmpeg_path = "ffmpeg"
        try:
            subprocess.run(["which", "ffmpeg"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            for path in ["/usr/local/bin/ffmpeg",
                         "/opt/homebrew/bin/ffmpeg",
                         "/opt/local/bin/ffmpeg"]:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            else:
                raise FileNotFoundError("ffmpeg not found.")

        log_func(f"Extracting audio track {track_index}...")
        cmd = [
            ffmpeg_path, "-y",
            "-i", input_file,
            "-map", f"0:{track_index}",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8")
        if result.returncode != 0:
            log_func(f"Extraction failed for track {track_index}.")
            return False
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            return False
        log_func(f"Audio extracted ({os.path.getsize(output_file)} bytes)")
        return True
    except Exception as e:
        log_func(f"Error extracting audio: {e}")
        return False


def _to_mp3_for_api(wav_path: str, log_func) -> str:
    """
    Convert WAV → MP3 so the file stays well under the 25 MB API limit.
    Returns the new path; caller is responsible for cleanup.
    """
    mp3_path = wav_path.replace(".wav", "_api.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ac", "1", "-ar", "16000",
        "-b:a", "64k",          # 64 kbps mono ≈ 0.5 MB/min – plenty for speech
        mp3_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(mp3_path):
        log_func("⚠️  MP3 conversion failed, will upload WAV directly")
        return wav_path
    size_mb = os.path.getsize(mp3_path) / 1_048_576
    log_func(f"   Audio compressed to MP3: {size_mb:.1f} MB")
    return mp3_path


def _openai_transcribe(audio_path: str, language: str,
                       is_mic_track: bool, track_name: str,
                       log_func) -> List[str]:
    """
    Call the OpenAI Whisper API and return word-level timestamp strings.
    Returns [] on any failure so the caller can fall back.
    """
    client = _get_openai_client(log_func)
    if client is None:
        return []  # reason already logged inside _get_openai_client

    mp3_path = _to_mp3_for_api(audio_path, log_func)
    temp_mp3_created = mp3_path != audio_path

    try:
        file_size_mb = os.path.getsize(mp3_path) / 1_048_576
        log_func(f"📤 Uploading to OpenAI Whisper ({file_size_mb:.1f} MB) …")

        lang_code = "en" if language.lower() in ("english", "en") else language.lower()

        with open(mp3_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=lang_code,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        words = getattr(response, "words", None) or []
        if not words:
            log_func("⚠️  OpenAI returned no word-level timestamps")
            return []

        log_func(f"✅ OpenAI transcribed {len(words)} words for {track_name}")

        transcriptions = []
        for w in words:
            text = w.word.strip()
            if not text:
                continue
            if is_mic_track:
                text = text.upper()
            transcriptions.append(f"{w.start:.2f}-{w.end:.2f}: {text}")

        return transcriptions

    except Exception as e:
        # Surface as much detail as possible — the OpenAI SDK hides useful info
        # in attributes like .status_code, .response, .body, .code, .message.
        log_func(f"⚠️  OpenAI Whisper error: {type(e).__name__}: {e}")
        for attr in ("status_code", "code", "message", "type", "param"):
            val = getattr(e, attr, None)
            if val is not None:
                log_func(f"     .{attr} = {val!r}")
        body = getattr(e, "body", None)
        if body is not None:
            log_func(f"     .body = {body!r}")
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                log_func(f"     .response.status_code = {resp.status_code}")
                log_func(f"     .response.text = {resp.text[:1000]!r}")
            except Exception:
                pass
        import traceback
        log_func(traceback.format_exc())
        return []
    finally:
        if temp_mp3_created and os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
            except Exception:
                pass


def _local_transcribe(model_path: str, device: str,
                      audio_path: str, language: str,
                      is_mic_track: bool, track_name: str,
                      log_func) -> List[str]:
    """Local faster-whisper + optional WhisperX alignment fallback."""
    if not _local_whisper_available:
        log_func("❌ faster-whisper not installed and OpenAI path failed.")
        return []

    log_func(f"🔄 Using local Whisper model ({model_path}) for {track_name} …")
    compute_type = "float16" if device == "cuda" else "float32"
    model = WhisperModel(model_path, device=device, compute_type=compute_type)

    lang = "en" if language.lower() in ("english", "en") else language.lower()
    segments, info = model.transcribe(
        audio_path, language=lang,
        word_timestamps=True, beam_size=5,
    )
    segments_list = list(segments)

    # Try WhisperX alignment
    if _whisperx_available and segments_list:
        try:
            from whisperx import load_align_model, align as wx_align
            model_a, metadata = load_align_model(
                language_code=info.language, device=device)
            wx_segs = [{"start": s.start, "end": s.end, "text": s.text}
                       for s in segments_list]
            aligned = wx_align(wx_segs, model_a, metadata,
                               audio_path, device,
                               return_char_alignments=False)
            aligned_segs = aligned["segments"]
            log_func(f"   WhisperX alignment done for {track_name}")

            transcriptions = []
            for seg in aligned_segs:
                for w in seg.get("words", []):
                    text = w["word"].strip()
                    if not text:
                        continue
                    if is_mic_track:
                        text = text.upper()
                    transcriptions.append(
                        f"{w['start']:.2f}-{w['end']:.2f}: {text}")
            return transcriptions
        except Exception as e:
            log_func(f"   WhisperX alignment failed ({e}), using raw timestamps")

    # Raw faster-whisper output
    transcriptions = []
    for seg in segments_list:
        for w in (seg.words or []):
            text = w.word.strip()
            if not text:
                continue
            if is_mic_track:
                text = text.upper()
            transcriptions.append(f"{w.start:.2f}-{w.end:.2f}: {text}")
    return transcriptions


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (matches original signature)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    model_path: str,
    device: str,
    audio_path: str,
    include_timecodes: bool,
    log_func,
    language: str,
    track_name: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Transcribe audio to word-level timestamp strings.

    Priority:  OpenAI Whisper API  →  local faster-whisper + WhisperX

    Returns:
        (raw_for_trimming, adjusted_for_subtitles)
    """
    try:
        log_func(f"\nStarting transcription: {track_name}")

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            log_func(f"ERROR: Invalid audio file: {audio_path}")
            return [], []

        is_mic_track = "mic" in track_name.lower() or "track 2" in track_name.lower()

        # ── 1. Try OpenAI API ─────────────────────────────────────────────
        transcriptions = []
        if not _openai_available:
            log_func("⚠️  OpenAI package not installed — using local Whisper "
                     "(run: pip install openai  to enable the API path)")
        else:
            log_func(f"🌐 Attempting OpenAI Whisper API for {track_name} …")
            transcriptions = _openai_transcribe(
                audio_path, language, is_mic_track, track_name, log_func)

        # ── 2. Fall back to local model ───────────────────────────────────
        if not transcriptions:
            log_func(f"🔄 Falling back to local Whisper for {track_name} …")
            transcriptions = _local_transcribe(
                model_path, device, audio_path,
                language, is_mic_track, track_name, log_func)

        if not transcriptions:
            log_func(f"⚠️  No transcription produced for {track_name}")
            return [], []

        log_func(f"   Raw words: {len(transcriptions)}")

        if not include_timecodes:
            return list(transcriptions), list(transcriptions)

        # ── 3. Hallucination filter ───────────────────────────────────────
        transcriptions = filter_hallucinations(
            transcriptions, audio_path, track_name, log_func)

        raw_for_trimming = fix_overlapping_timestamps(list(transcriptions))

        adjusted = apply_duration_adjustments(
            list(transcriptions), track_name, log_func)
        adjusted_for_subs = fix_overlapping_timestamps(adjusted)

        return raw_for_trimming, adjusted_for_subs

    except Exception as e:
        log_func(f"Error in transcription: {e}")
        import traceback
        log_func(traceback.format_exc())
        err = [f"0.0-5.0: Transcription error: {e}"]
        return err, err