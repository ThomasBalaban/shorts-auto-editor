# utils/timestamp_processor.py

"""
Timestamp processing utilities for subtitle generation.
Handles duration adjustments, overlap fixing, and timestamp formatting.
"""

from typing import List, Tuple
import numpy as np # type: ignore

try:
    import librosa # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def detect_continuous_vocalization(
    audio_path: str,
    check_time: float,
    lookforward_duration: float = 1.5,
    energy_threshold: float = 0.025,
    log_func = None
) -> Tuple[bool, float]:
    """
    Detects if there's sustained vocalization using RMS energy.
    Kept as a backup for non-speech sounds (screams not caught by Whisper).
    """
    if not LIBROSA_AVAILABLE or not audio_path:
        return False, check_time
    
    try:
        audio, sr = librosa.load(audio_path, sr=16000, offset=check_time, duration=lookforward_duration)
        if len(audio) == 0: return False, check_time
        
        window_size = int(0.05 * sr)
        hop_size = int(0.01 * sr)
        sustained_until = check_time
        is_vocalizing = False
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms_energy = np.sqrt(np.mean(window**2))
            current_time = check_time + (i / sr)
            
            if rms_energy > energy_threshold:
                sustained_until = current_time + (window_size / sr)
                is_vocalizing = True
            elif is_vocalizing:
                break
        
        return is_vocalizing, sustained_until
        
    except Exception:
        return False, check_time


# Subtitle word duration floor.
#
# Whisper word durations for short tokens like "the" / "a" / "of" are
# commonly 80–150 ms; the prior 0.30 s floor stretched these to linger
# ~150–220 ms past the actual speech, which read as "the subtitle hangs
# after I'm done talking." 0.08 s ≈ 5 frames at 60 fps and is short
# enough to feel snappy while still being rendered.
MIN_SUBTITLE_WORD_DURATION_S = 0.08
MAX_SUBTITLE_WORD_DURATION_S = 1.5


def adjust_word_duration(
    start_time,
    end_time,
    min_duration: float = MIN_SUBTITLE_WORD_DURATION_S,
    max_duration: float = MAX_SUBTITLE_WORD_DURATION_S,
):
    """Adjust word timestamps based on duration constraints.

    For OVER-long words (held vocalizations like "OKAYYYY"), the cap
    preserves END and pushes START forward. This matters because for
    these tokens Whisper's annotated start is often the moment audio
    energy first appears, which can be hundreds of ms before the user
    perceives the word beginning. Preserving END keeps the rendered
    subtitle near the perceived peak of the vocalization.

    For UNDER-short words, the floor preserves START and stretches END.
    """
    current_duration = end_time - start_time
    if current_duration > max_duration:
        return end_time - max_duration, end_time
    if current_duration < min_duration:
        return start_time, start_time + min_duration
    return start_time, end_time


def apply_duration_adjustments(transcriptions, track_name="", log_func=None):
    """Apply duration adjustments to a list of transcriptions."""
    if not transcriptions or not log_func: return transcriptions

    log_func(f"Applying duration adjustments for {track_name}...")
    adjusted_transcriptions = []

    for line in transcriptions:
        try:
            time_part, text = line.split(':', 1)
            start_str, end_str = time_part.split('-')
            new_start, new_end = adjust_word_duration(float(start_str), float(end_str))
            adjusted_transcriptions.append(f"{new_start:.2f}-{new_end:.2f}:{text}")
        except (ValueError, IndexError):
            adjusted_transcriptions.append(line)

    return adjusted_transcriptions


def fix_overlapping_timestamps(transcriptions, min_duration=0.1):
    """Fix overlapping timestamps in transcriptions."""
    if not transcriptions: return []
    
    parsed = []
    for line in transcriptions:
        try:
            time_part, text = line.split(':', 1)
            start, end = map(float, time_part.split('-'))
            parsed.append((start, end, text.strip()))
        except ValueError: continue
    
    parsed.sort(key=lambda x: x[0])
    fixed = []
    if parsed:
        fixed.append(parsed[0])
        for i in range(1, len(parsed)):
            prev_start, prev_end, prev_text = fixed[-1]
            curr_start, curr_end, curr_text = parsed[i]
            
            if curr_end - curr_start < min_duration:
                curr_end = curr_start + min_duration
            
            if curr_start < prev_end:
                curr_start = prev_end + 0.01
                if curr_end < curr_start + min_duration:
                    curr_end = curr_start + min_duration
            
            fixed.append((curr_start, curr_end, curr_text))
    
    return [f"{s:.2f}-{e:.2f}: {t}" for s, e, t in fixed]


def parse_timestamp_line(line):
    try:
        time_part, text = line.split(':', 1)
        start, end = map(float, time_part.split('-'))
        return start, end, text.strip()
    except (ValueError, IndexError):
        return None

def format_timestamp_line(start_time, end_time, text):
    return f"{start_time:.2f}-{end_time:.2f}: {text}"


def shift_transcriptions_to_output_time(
    transcriptions: List[str],
    kept_segments: List[Tuple[float, float]],
) -> List[str]:
    """Map source-time transcription lines into trimmed-output time.

    Used after a trim is applied: each kept segment [s_i, e_i] in source-time
    becomes a contiguous range starting at sum_{j<i}(e_j-s_j) in output-time.
    A word at source time T inside segment i lands at:
        T_out = (T - s_i) + cumulative_offset(i)

    Lines whose start or end falls outside all kept segments are dropped.
    Lines spanning a boundary are clipped to the segment they start in
    (rare for word-level timestamps; safer than dropping mid-utterance).
    """
    if not transcriptions or not kept_segments:
        return []

    # Precompute cumulative offsets per segment.
    seg_with_offset: List[Tuple[float, float, float]] = []
    cum = 0.0
    for s, e in kept_segments:
        seg_with_offset.append((float(s), float(e), cum))
        cum += float(e) - float(s)

    out: List[str] = []
    for line in transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed is None:
            continue
        start, end, text = parsed
        # Find segment containing the start time.
        for s, e, off in seg_with_offset:
            if s <= start < e:
                clipped_end = min(end, e)
                if clipped_end <= start:
                    break
                new_start = (start - s) + off
                new_end = (clipped_end - s) + off
                out.append(format_timestamp_line(new_start, new_end, text))
                break
        # Lines starting outside any kept segment are dropped.
    return out


def extend_segments_for_dialogue(
    segments_to_keep: List[Tuple[float, float]],
    raw_mic_transcriptions: List[str],
    raw_desktop_transcriptions: List[str],
    log_func,
    max_extension_seconds: float = 3.0,
    buffer_seconds: float = 0.5,
    mic_audio_path: str = None,
    desktop_audio_path: str = None
) -> List[Tuple[float, float]]:
    """
    Extends trim segments to protect dialogue using BOTH Whisper timestamps (Primary) 
    and RMS energy (Secondary/Fallback).
    """
    if not segments_to_keep:
        log_func("   No segments to protect")
        return []
    
    # 1. Parse all words into a searchable list
    all_words = [] 
    for line in raw_mic_transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed: all_words.append(parsed + ("mic",))
    for line in raw_desktop_transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed: all_words.append(parsed + ("desktop",))
    
    all_words.sort(key=lambda x: x[0])
    segments_to_keep.sort(key=lambda x: x[0])
    
    extended_segments = []
    last_end_time = 0.0

    for segment_idx, (ai_start, ai_end) in enumerate(segments_to_keep):
        log_func(f"\n   📍 Processing segment {segment_idx + 1}: {ai_start:.2f}s - {ai_end:.2f}s")
        
        new_start = max(0.0, ai_start - buffer_seconds)
        new_end = ai_end
        
        if new_start < last_end_time:
            new_start = last_end_time

        # --- STEP 1: WHISPER-BASED PROTECTION (The Fix) ---
        # Check if the cut point (ai_end) slices through a word or is uncomfortably close
        extension_found = False

        # Look for words that start before the cut but end after it
        # OR words that start within a small window after the cut (don't cut mid-sentence flow)
        sentence_flow_window = 0.5

        # A real spoken word is rarely over 1.5s. Whisper sometimes assigns
        # huge durations when there's silence around a token, or stretches a
        # held scream/shout ("MEEEEE!") into a single word. Protecting those
        # against the trimmer's cut decision yanks dead air back into the
        # final cut. When a candidate word's annotated span is longer than
        # this threshold we trust the trimmer's intent over the protector.
        MAX_PROTECTABLE_WORD_DURATION = 1.5

        for (w_start, w_end, text, track) in all_words:
            word_duration = w_end - w_start

            # Case A: Cut is inside a word
            if w_start <= ai_end < w_end:
                if word_duration > MAX_PROTECTABLE_WORD_DURATION:
                    log_func(
                        f"      🚫 Skipped abnormally-long protected word "
                        f"({track}): '{text}' "
                        f"({w_start:.2f}-{w_end:.2f}, {word_duration:.1f}s) "
                        "— likely scream / held vocal / silence artifact, "
                        "honoring trimmer's cut"
                    )
                    continue
                log_func(f"      🛡️ Protected cut word ({track}): '{text}' ({w_start:.2f}-{w_end:.2f})")
                new_end = max(new_end, w_end + buffer_seconds)
                extension_found = True

            # Case B: Cut is right before a word (likely mid-sentence)
            elif ai_end <= w_start < (ai_end + sentence_flow_window):
                if word_duration > MAX_PROTECTABLE_WORD_DURATION:
                    log_func(
                        f"      🚫 Skipped abnormally-long flow word "
                        f"({track}): '{text}' "
                        f"({w_start:.2f}-{w_end:.2f}, {word_duration:.1f}s)"
                    )
                    continue
                # Only extend if it doesn't push us too far
                if (w_end - ai_end) < max_extension_seconds:
                    log_func(f"      🛡️ Extended for flow ({track}): '{text}' starts at {w_start:.2f}")
                    new_end = max(new_end, w_end + buffer_seconds)
                    extension_found = True

        if extension_found:
            # Cap the extension
            if (new_end - ai_end) > max_extension_seconds:
                new_end = ai_end + max_extension_seconds
                log_func(f"      ⚠️ Extension capped at {new_end:.2f}s")

        # --- STEP 2: ENERGY-BASED PROTECTION (The Backup) ---
        # Only run if Whisper didn't already extend it significantly.
        # This catches screams/laughs that Whisper ignored.
        if (new_end - ai_end) < 0.5 and LIBROSA_AVAILABLE:
            log_func(f"      🎤 Checking for untranscribed vocalization (screams/laughs)...")
            
            mic_active, mic_end = detect_continuous_vocalization(mic_audio_path, new_end, 1.5, 0.025, log_func)
            desk_active, desk_end = detect_continuous_vocalization(desktop_audio_path, new_end, 1.5, 0.025, log_func)
            
            actual_vocal_end = max(mic_end, desk_end)
            if actual_vocal_end > new_end:
                extension = actual_vocal_end - new_end
                if extension <= max_extension_seconds:
                    new_end = actual_vocal_end + buffer_seconds
                    log_func(f"      🔊 Extended for vocal energy to {new_end:.2f}s")
                else:
                    new_end = new_end + max_extension_seconds
                    log_func(f"      🔊 Energy extends too far, capped.")

        # Default buffer if nothing else happened
        if new_end == ai_end:
            new_end += buffer_seconds

        if new_start < new_end:
            extended_segments.append((new_start, new_end))
            last_end_time = new_end
            log_func(f"      ✅ Final: {new_start:.2f}s - {new_end:.2f}s")

    return extended_segments