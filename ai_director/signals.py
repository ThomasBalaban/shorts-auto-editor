# ai_director/signals.py
"""
Signal extraction for the content-driven AI Director.

The director decides zooms from independent, real-timestamped signals rather
than a fixed time grid. This module produces those signals — all aligned to the
*working* (trimmed) clip's timeline, the same frame of reference the director's
TimelineEvents render in:

  • mic energy envelope     — vocal loudness over time (screams/laughs)
  • game audio envelope     — game loudness over time (impacts vs quiet)
  • mic utterances          — real per-utterance (start,end,text) from transcript
  • vision beats            — hybrid-sampled scene captions + epic/suspense class

Everything degrades gracefully: a missing track or a failed pass returns an
empty/low signal instead of raising into the pipeline.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─── Utterances (from the transcript strings, already in output time) ─────────

@dataclass
class Utterance:
    start: float
    end: float
    text: str


_WORD_RE = re.compile(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*:\s?(.*)$")


def parse_utterances(
    mic_transcription: List[str],
    max_gap: float = 0.7,
) -> List[Utterance]:
    """Parse the director's `mic_transcription` (a list of ``"start-end: word"``
    strings) into utterances by grouping consecutive words separated by less
    than ``max_gap`` seconds. Returns utterances sorted by start time."""
    words: List[Tuple[float, float, str]] = []
    for line in mic_transcription or []:
        m = _WORD_RE.match(line)
        if not m:
            continue
        try:
            start, end = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        text = m.group(3).strip()
        if text:
            words.append((start, end, text))

    words.sort(key=lambda w: w[0])
    utterances: List[Utterance] = []
    cur: Optional[List] = None  # [start, end, [words]]
    for start, end, text in words:
        if cur is not None and start - cur[1] <= max_gap:
            cur[1] = max(cur[1], end)
            cur[2].append(text)
        else:
            if cur is not None:
                utterances.append(Utterance(cur[0], cur[1], " ".join(cur[2])))
            cur = [start, end, [text]]
    if cur is not None:
        utterances.append(Utterance(cur[0], cur[1], " ".join(cur[2])))
    return utterances


# ─── Energy envelopes ─────────────────────────────────────────────────────────

@dataclass
class Envelope:
    """A coarse RMS-loudness-over-time curve with relative-loudness helpers.

    `norm_at`/`peak_norm` map absolute RMS onto 0–1 using the clip's own
    distribution (p10→p90), so "loud" and "quiet" are relative to this clip
    rather than an absolute scale that varies by game/mic gain.
    """
    times: np.ndarray
    rms: np.ndarray
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 1.0
    available: bool = True

    @staticmethod
    def empty() -> "Envelope":
        return Envelope(np.array([]), np.array([]), 0.0, 0.0, 1.0, False)

    def _idx(self, t: float) -> int:
        if self.times.size == 0:
            return -1
        return int(np.clip(np.searchsorted(self.times, t), 0, self.times.size - 1))

    def value_at(self, t: float) -> float:
        i = self._idx(t)
        return float(self.rms[i]) if i >= 0 else 0.0

    def norm_at(self, t: float) -> float:
        return self._normalize(self.value_at(t))

    def _normalize(self, v: float) -> float:
        span = self.p90 - self.p10
        if span <= 1e-9:
            return 0.0
        return float(np.clip((v - self.p10) / span, 0.0, 1.0))

    def peak_norm(self, t0: float, t1: float) -> Tuple[float, float]:
        """Return (normalized_peak, peak_time) of RMS within [t0, t1]."""
        if self.times.size == 0 or t1 <= t0:
            return 0.0, t0
        lo, hi = self._idx(t0), self._idx(t1)
        lo, hi = min(lo, hi), max(lo, hi)
        seg = self.rms[lo:hi + 1]
        if seg.size == 0:
            return 0.0, t0
        k = int(np.argmax(seg))
        return self._normalize(float(seg[k])), float(self.times[lo + k])

    def low_ratio(self, t0: float, t1: float) -> float:
        """Fraction of the window that sits below the clip's median loudness —
        a proxy for sustained quiet (suspense corroboration)."""
        if self.times.size == 0 or t1 <= t0:
            return 0.0
        lo, hi = self._idx(t0), self._idx(t1)
        lo, hi = min(lo, hi), max(lo, hi)
        seg = self.rms[lo:hi + 1]
        if seg.size == 0:
            return 0.0
        return float(np.mean(seg < self.p50))


def _extract_track_wav(
    video_path: str,
    track_index: str,
    log_func: Callable[[str], None],
) -> Optional[str]:
    """Extract one audio track to a temp wav via the shared FileProcessor.
    Returns the path, or None when the track is absent/extraction fails."""
    try:
        from utils.file_processor import FileProcessor
        path = FileProcessor(log_func=log_func).extract_audio_from_video(
            video_path, track_index=track_index)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    except Exception as e:
        log_func(f"   ⚠ could not extract {track_index} audio: {e}")
    return None


def energy_envelope(
    video_path: str,
    track_index: str,
    log_func: Callable[[str], None] = print,
    hop_s: float = 0.1,
    win_s: float = 0.2,
) -> Envelope:
    """Coarse RMS envelope (default 100ms hops) for one audio track of the
    working clip. Returns Envelope.empty() if the track can't be read."""
    wav = _extract_track_wav(video_path, track_index, log_func)
    if not wav:
        return Envelope.empty()
    try:
        import librosa  # local import — heavy
        sr = 22050
        audio, sr = librosa.load(wav, sr=sr)
        if audio.size == 0:
            return Envelope.empty()
        hop = max(1, int(hop_s * sr))
        frame = max(hop, int(win_s * sr))
        rms = librosa.feature.rms(
            y=audio, frame_length=frame, hop_length=hop)[0]
        times = librosa.frames_to_time(
            np.arange(rms.size), sr=sr, hop_length=hop)
        p10, p50, p90 = (float(np.percentile(rms, q)) for q in (10, 50, 90))
        return Envelope(times, rms, p10, p50, p90, True)
    except Exception as e:
        log_func(f"   ⚠ energy envelope failed for {track_index}: {e}")
        return Envelope.empty()
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


# ─── Vision beats (hybrid sampling: scene changes + heartbeat) ────────────────

@dataclass
class VisionBeat:
    time: float
    caption: str
    intensity: str  # "epic" | "suspenseful" | "normal"
    scene_context: set = field(default_factory=set)


def scene_change_times(
    video_path: str,
    threshold: float = 0.4,
    log_func: Callable[[str], None] = print,
) -> List[float]:
    """ffmpeg scene-cut timestamps (local, no Gemini cost). Empty on failure."""
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-an", "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        times = [float(m) for m in re.findall(
            r"pts_time:([0-9]+\.?[0-9]*)", proc.stderr)]
        return sorted(set(round(t, 2) for t in times))
    except Exception as e:
        log_func(f"   ⚠ scene detection failed: {e}")
        return []


def sample_times(
    video_path: str,
    duration: float,
    heartbeat_s: float = 5.0,
    scene_threshold: float = 0.4,
    min_gap_s: float = 1.5,
    max_beats: int = 48,
    log_func: Callable[[str], None] = print,
) -> List[float]:
    """Hybrid sample schedule: scene-change cuts + a sparse heartbeat, deduped
    so no two samples are within ``min_gap_s``. Capped at ``max_beats`` (logs if
    it truncates)."""
    scenes = scene_change_times(video_path, scene_threshold, log_func)
    heartbeat = list(np.arange(1.5, max(2.0, duration - 1.0), heartbeat_s))
    merged = sorted(set([round(t, 2) for t in scenes + heartbeat]))

    deduped: List[float] = []
    for t in merged:
        if 0.0 <= t <= duration and (not deduped or t - deduped[-1] >= min_gap_s):
            deduped.append(t)

    if len(deduped) > max_beats:
        log_func(
            f"   ⚠ vision sampling capped at {max_beats} beats "
            f"(had {len(deduped)}); spreading evenly.")
        idx = np.linspace(0, len(deduped) - 1, max_beats).round().astype(int)
        deduped = [deduped[i] for i in sorted(set(idx))]
    return deduped


def vision_beats(
    video_path: str,
    duration: float,
    analyzer=None,
    heartbeat_s: float = 5.0,
    log_func: Callable[[str], None] = print,
) -> List[VisionBeat]:
    """Run the hybrid-sampled vision pass and return classified beats.

    Reuses GeminiVisionAnalyzer.analyze_video_at_timestamps (frame extraction +
    captioning + the new epic/suspense `intensity`). Independent of audio onsets
    — this is what lets the director see quiet/suspenseful gameplay.
    """
    times = sample_times(video_path, duration, heartbeat_s=heartbeat_s,
                         log_func=log_func)
    if not times:
        return []
    if analyzer is None:
        try:
            from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
            analyzer = GeminiVisionAnalyzer(log_func=log_func)
        except Exception as e:
            log_func(f"   ⚠ vision analyzer unavailable: {e}")
            return []

    fake_events = [{"time": float(t)} for t in times]
    try:
        amap = analyzer.analyze_video_at_timestamps(video_path, fake_events)
    except Exception as e:
        log_func(f"   ⚠ vision beat pass failed: {e}")
        return []

    beats: List[VisionBeat] = []
    for t, a in amap.items():
        beats.append(VisionBeat(
            time=float(t),
            caption=a.get("video_caption", ""),
            intensity=a.get("intensity", "normal"),
            scene_context=a.get("scene_context", set()) or set(),
        ))
    beats.sort(key=lambda b: b.time)
    return beats
