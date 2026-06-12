# ai_director/data_models.py
"""
Data models for the AI Director system.

Defines the structure for tasks, responses, and the final decision timeline,
facilitating communication between the Master Director and specialist AIs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Literal, Optional
import uuid

# --- Task & Response Models ---

@dataclass
class DirectorTask:
    """A task dispatched by the Master Director to a specialist AI."""
    task_type: Literal[
        "analyze_text_content",
        "analyze_audio_event",
        "analyze_visual_context"
    ]
    time_range: Tuple[float, float]
    priority: Literal["low", "medium", "high"]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, any] = field(default_factory=dict)
    instructions: str = ""

@dataclass
class SpecialistResponse:
    """A response from a specialist AI back to the Master Director."""
    task_id: str
    result: Literal[
        "wild_content_detected", "awkward_content_detected", "fear_surprise_detected",
        "dramatic_moment_detected", "attention_direction_detected", "no_significant_event",
        "dramatic_moment_visual"
    ]
    confidence: float
    recommended_action: Optional[Literal[
        "zoom_to_cam", "zoom_to_game", "zoom_out",
        "zoom_to_cam_reaction"
    ]] = None
    details: Dict[str, any] = field(default_factory=dict)

# --- Decision Timeline Model ---

@dataclass
class TimelineEvent:
    """Represents a single editing decision in the final timeline."""
    timestamp: float
    action: Literal[
        "zoom_to_cam", "zoom_to_game", "zoom_out", "no_action",
        "zoom_to_cam_reaction"
    ]
    duration: float
    reason: str
    confidence: float
    source_task_id: Optional[str] = None


# --- Candidate Model (content-driven director) ---

# Map a candidate's kind → the TimelineEvent action that renders it.
KIND_TO_ACTION = {
    "cam_reaction":  "zoom_to_cam_reaction",
    "game_epic":     "zoom_to_game",
    "game_suspense": "zoom_to_game",
    "awkward":       "zoom_out",
}

# Default on-screen duration (seconds) per kind.
KIND_TO_DURATION = {
    "cam_reaction":  2.0,
    "game_epic":     3.0,
    "game_suspense": 3.0,
    "awkward":       3.0,
}


@dataclass
class ZoomCandidate:
    """A single proposed zoom moment before selection.

    The director generates many of these from independent signals (mic
    reactions, game audio, vision beats), scores each on a common 0–1 scale,
    then selects a non-overlapping subset above a quality threshold. This is
    what replaces the old fixed-grid + editorial-conflict logic.
    """
    t_peak: float                       # when the zoom should hit
    kind: Literal[
        "cam_reaction", "game_epic", "game_suspense", "awkward"
    ]
    score: float                        # 0–1, comparable across kinds
    reason: str = ""                    # short human-readable label for logs
    caption: str = ""                   # vision caption / utterance text
    t_start: Optional[float] = None     # signal window (optional, for logs)
    t_end: Optional[float] = None

    @property
    def action(self) -> str:
        return KIND_TO_ACTION.get(self.kind, "no_action")

    @property
    def duration(self) -> float:
        return KIND_TO_DURATION.get(self.kind, 2.0)

    def to_timeline_event(self) -> "TimelineEvent":
        return TimelineEvent(
            timestamp=self.t_peak,
            action=self.action,          # type: ignore[arg-type]
            duration=self.duration,
            reason=self.reason or self.kind,
            confidence=self.score,
        )