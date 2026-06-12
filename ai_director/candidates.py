# ai_director/candidates.py
"""
Candidate generation for the content-driven AI Director.

Turns the raw signals (mic utterances + energy, game-audio energy, vision beats)
into scored ZoomCandidates on a common 0–1 scale. The director then selects a
non-overlapping subset above a quality threshold — so cadence emerges from the
content instead of a fixed grid.

Scoring intent (operator's goals):
  • cam_reaction  ← standout spoken reactions OR loud wordless vocal bursts
  • game_epic     ← exciting gameplay beats (vision 'epic' and/or loud impacts)
  • game_suspense ← tense/eerie beats (vision 'suspenseful'), a bonus vision adds
Epic outweighs suspense; vision carries the game side when audio is unsure.
"""

from typing import Callable, List

from ai_director.data_models import ZoomCandidate
from ai_director.signals import Envelope, Utterance, VisionBeat

# Loud wordless burst (relative loudness) needed for a 'normal' line to still
# earn a cam reaction (a scream/laugh over unremarkable words).
VOCAL_BURST_THRESHOLD = 0.8


def generate_cam_candidates(
    utterances: List[Utterance],
    mic_env: Envelope,
    classifier,
    log_func: Callable[[str], None] = print,
) -> List[ZoomCandidate]:
    """Reaction candidates from the mic: standout speech (LLM) blended with
    vocal loudness; plus loud wordless bursts even when the words are plain."""
    out: List[ZoomCandidate] = []
    for u in utterances:
        vocal_norm, peak_t = mic_env.peak_norm(u.start, u.end)
        label, base = classifier.classify(u.text)

        if label == "wild":
            score = min(1.0, max(base, 0.6 + 0.35 * vocal_norm))
            out.append(ZoomCandidate(
                t_peak=peak_t, kind="cam_reaction", score=score,
                reason="reaction:wild", caption=u.text,
                t_start=u.start, t_end=u.end))
        elif label == "awkward":
            out.append(ZoomCandidate(
                t_peak=peak_t, kind="awkward", score=0.45 + 0.15 * vocal_norm,
                reason="reaction:awkward", caption=u.text,
                t_start=u.start, t_end=u.end))
        elif vocal_norm >= VOCAL_BURST_THRESHOLD:
            # Plain words but a big vocal burst — scream/laugh.
            out.append(ZoomCandidate(
                t_peak=peak_t, kind="cam_reaction", score=0.55 + 0.30 * vocal_norm,
                reason="reaction:vocal_burst", caption=u.text,
                t_start=u.start, t_end=u.end))
    log_func(f"   • cam candidates: {len(out)} from {len(utterances)} utterances")
    return out


def generate_game_candidates(
    beats: List[VisionBeat],
    game_env: Envelope,
    audio_events: List[dict],
    log_func: Callable[[str], None] = print,
) -> List[ZoomCandidate]:
    """Game candidates from vision beats (epic/suspense) corroborated by game
    audio, plus standalone loud impacts the vision pass may have missed."""
    out: List[ZoomCandidate] = []

    for b in beats:
        if b.intensity == "epic":
            audio_norm, _ = game_env.peak_norm(b.time - 1.0, b.time + 1.0)
            score = min(1.0, 0.78 + 0.20 * audio_norm)
            out.append(ZoomCandidate(
                t_peak=b.time, kind="game_epic", score=score,
                reason="vision:epic", caption=b.caption))
        elif b.intensity == "suspenseful":
            quiet = game_env.low_ratio(b.time - 1.5, b.time + 1.5)
            score = min(0.85, 0.52 + 0.18 * quiet)
            out.append(ZoomCandidate(
                t_peak=b.time, kind="game_suspense", score=score,
                reason="vision:suspense", caption=b.caption))

    # Standalone loud impacts (major onsets) not already covered by a vision
    # epic nearby — audio carries the game side when vision said 'normal'.
    # NB: these are the fused onomatopoeia effects, whose timing field is
    # `precise_peak_time`/`start_time` (not `time`).
    for e in audio_events or []:
        if e.get("tier") != "major":
            continue
        t = float(e.get("precise_peak_time", e.get("start_time", 0.0)))
        if any(c.kind == "game_epic" and abs(t - c.t_peak) < 2.0 for c in out):
            continue
        if game_env.available:
            norm = game_env.norm_at(t)
        else:
            norm = min(1.0, float(e.get("energy", 0.0)) / 0.1)
        out.append(ZoomCandidate(
            t_peak=t, kind="game_epic", score=0.50 + 0.25 * norm,
            reason="audio:impact", caption=""))

    log_func(
        f"   • game candidates: {len(out)} "
        f"(vision beats={len(beats)}, audio events={len(audio_events or [])})")
    return out
