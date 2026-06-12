# ai_director/master_director.py
"""
MasterDirector — content-driven zoom selection.

Replaces the old fixed 6s grid + per-window specialists + editorial conflict
resolver. The new flow:

    signals  →  candidates  →  selection  →  TimelineEvents

  1. Extract independent, real-timestamped signals (mic utterances + energy,
     game-audio energy, hybrid-sampled vision beats).
  2. Each signal proposes scored ZoomCandidates on one 0–1 scale.
  3. Select a non-overlapping subset above a quality threshold (greedy by
     score). Cadence emerges from the content — eventful clips get more zooms,
     calm clips fewer. Nothing is placed on a grid.

The emitted TimelineEvent action vocabulary is unchanged, so VideoEditor is
untouched.
"""

from typing import Callable, Dict, List, Optional

from ai_director.data_models import TimelineEvent, ZoomCandidate
from ai_director.specialists import ReactionClassifier
from ai_director import signals as sig
from ai_director.candidates import (
    generate_cam_candidates,
    generate_game_candidates,
)

# Only candidates scoring at/above this survive — the quality gate that keeps
# cadence content-driven rather than constant.
SCORE_THRESHOLD = 0.55
# Minimum spacing between two kept zooms' peaks, on top of no-window-overlap.
REFRACTORY_S = 1.5


class MasterDirector:
    """Orchestrates content-driven zoom decisions."""

    def __init__(self, log_func=None, detailed_logs=False):
        self.log_func = log_func or print
        self.detailed_logs = detailed_logs
        self.classifier = ReactionClassifier(log_func=self.log_func)
        self.log_func("👑 AI Master Director initialized (content-driven).")

    def analyze_video_and_create_timeline(
        self,
        video_path: str,
        video_duration: float,
        mic_transcription: List[str],
        audio_events: List[Dict],
        video_analysis_map: Dict[float, Dict],  # legacy arg, no longer required
        camera_region: Optional[Dict] = None,   # normalized {x,y,w,h} for gaze
    ) -> List[TimelineEvent]:
        self.log_func("--- AI Director: Starting Content Analysis ---")

        # One vision analyzer shared by the beat pass and the gaze gate.
        analyzer = self._make_analyzer()

        # ── 1. Signals ────────────────────────────────────────────────────────
        utterances = sig.parse_utterances(mic_transcription)
        mic_env = sig.energy_envelope(
            video_path, "a:1", self.log_func)        # mic / player voice
        game_env = sig.energy_envelope(
            video_path, "a:2", self.log_func)        # desktop / game audio
        beats = sig.vision_beats(
            video_path, video_duration, analyzer=analyzer, log_func=self.log_func)
        self.log_func(
            f"   signals: {len(utterances)} utterances, "
            f"mic_env={'ok' if mic_env.available else 'none'}, "
            f"game_env={'ok' if game_env.available else 'none'}, "
            f"{len(beats)} vision beats")

        # ── 2. Candidates ─────────────────────────────────────────────────────
        candidates: List[ZoomCandidate] = []
        cam = generate_cam_candidates(
            utterances, mic_env, self.classifier, self.log_func)
        self._apply_gaze_gate(cam, video_path, camera_region, analyzer)
        candidates += cam
        candidates += generate_game_candidates(
            beats, game_env, audio_events, self.log_func)

        if not candidates:
            self.log_func("   No candidates generated — no zooms.")
            return []

        # ── 3. Selection ──────────────────────────────────────────────────────
        timeline = self._select(candidates)
        self.log_func(
            f"✅ AI Director complete. {len(timeline)} zooms from "
            f"{len(candidates)} candidates.")
        return timeline

    # ── Vision analyzer + camera-gaze gate ────────────────────────────────────
    def _make_analyzer(self):
        """Create the shared Gemini vision analyzer, or None if unavailable."""
        try:
            from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
            return GeminiVisionAnalyzer(log_func=self.log_func)
        except Exception as e:
            self.log_func(f"   ⚠ vision analyzer unavailable: {e}")
            return None

    def _apply_gaze_gate(
        self,
        cam_candidates: List[ZoomCandidate],
        video_path: str,
        camera_region: Optional[Dict],
        analyzer,
    ) -> None:
        """Penalize cam reactions whose avatar looks bugged (looking down/away
        or face not visible) in the camera region, so a game zoom or no-zoom
        wins the slot. Only checks candidates that could actually be selected
        (≥ threshold) to keep the extra vision calls minimal. Unknown/failed
        checks apply no penalty."""
        if not camera_region or analyzer is None:
            return
        for c in cam_candidates:
            if c.kind != "cam_reaction" or c.score < SCORE_THRESHOLD:
                continue
            q = analyzer.analyze_camera_gaze(video_path, c.t_peak, camera_region)
            if q is None or q >= 1.0:
                continue
            old = c.score
            c.score = round(c.score * (0.4 + 0.6 * q), 3)
            c.reason = f"{c.reason}|gaze={q:.2f}"
            self.log_func(
                f"   👁 gaze penalty: cam @ {c.t_peak:.2f}s "
                f"{old:.2f}→{c.score:.2f} (eyes q={q:.2f})")

    # ── Selection: greedy non-max suppression by score ────────────────────────
    def _select(self, candidates: List[ZoomCandidate]) -> List[TimelineEvent]:
        if self.detailed_logs:
            self.log_func("\n--- AI Director: Candidate Selection ---")

        # Highest score first; ties broken by earlier time for determinism.
        ranked = sorted(candidates, key=lambda c: (-c.score, c.t_peak))
        kept: List[ZoomCandidate] = []

        for c in ranked:
            if c.score < SCORE_THRESHOLD:
                if self.detailed_logs:
                    self.log_func(
                        f"  - drop  {c.kind:13s} @ {c.t_peak:6.2f}s "
                        f"score={c.score:.2f} ({c.reason}) — below threshold")
                continue
            clash = self._overlaps(c, kept)
            if clash is not None:
                if self.detailed_logs:
                    self.log_func(
                        f"  - drop  {c.kind:13s} @ {c.t_peak:6.2f}s "
                        f"score={c.score:.2f} ({c.reason}) — clashes with "
                        f"{clash.kind} @ {clash.t_peak:.2f}s")
                continue
            kept.append(c)
            if self.detailed_logs:
                cap = f" “{c.caption[:48]}”" if c.caption else ""
                self.log_func(
                    f"  - KEEP  {c.kind:13s} @ {c.t_peak:6.2f}s "
                    f"score={c.score:.2f} ({c.reason}){cap}")

        kept.sort(key=lambda c: c.t_peak)
        return [c.to_timeline_event() for c in kept]

    @staticmethod
    def _overlaps(
        c: ZoomCandidate, kept: List[ZoomCandidate]
    ) -> Optional[ZoomCandidate]:
        """Return a kept candidate that conflicts with `c`, else None. Two zooms
        conflict if their on-screen windows overlap or their peaks fall within
        the refractory gap."""
        c0, c1 = c.t_peak, c.t_peak + c.duration
        for k in kept:
            k0, k1 = k.t_peak, k.t_peak + k.duration
            if c0 < k1 and k0 < c1:
                return k
            if abs(c.t_peak - k.t_peak) < REFRACTORY_S:
                return k
        return None
