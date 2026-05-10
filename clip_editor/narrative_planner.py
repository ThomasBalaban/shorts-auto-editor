"""
Pre-trim narrative planner.

Runs before IntelligentTrimmer. Watches the source video AND reads the
transcripts, then identifies the moments that absolutely must survive the
cut: a hook (when one exists), and any number of "best moments" of any
kind — laughs, jumpscares, kills, tension peaks, satisfying reveals, etc.

The trimmer receives these as hard constraints in its prompt; an
auto-repair pass after parsing guarantees they're covered even if the
trimmer's response missed one.

Why video access matters
------------------------
Earlier iterations of this module read transcripts only. That worked for
dialogue-driven punchlines but went blind on horror clips, visual gags,
and tension built from silence. Sending the video lets the planner see
silent jumpscares and physical comedy the transcript can never reveal.

Failure mode
------------
Best-effort. Any LLM/parse failure or ``SHORTS_NARRATIVE_PLANNER=0``
returns ``None`` to the caller; the trimmer then falls back to its prior
blind-pick mode. Never raises.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional, Tuple

from google.genai import types

from utils.models import (
    MODEL_PRO,
    THINKING_TRIM,
    get_gemini_client,
    get_safety_settings,
)


def is_planner_enabled() -> bool:
    """Read the env-var gate. Default ON.

    Set ``SHORTS_NARRATIVE_PLANNER=0`` (or false/no/off) to disable; the
    trimmer then runs in its prior blind-pick mode.
    """
    raw = os.environ.get("SHORTS_NARRATIVE_PLANNER", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


# Inline upload threshold mirrors IntelligentTrimmer.INLINE_MAX_BYTES — the
# hard API request cap is 20 MB; 18 MB leaves room for the prompt + JSON.
INLINE_MAX_BYTES = 18 * 1024 * 1024


_PROMPT = """=== SOURCE ===
Total duration: {video_duration:.1f}s

DIALOGUE TRANSCRIPTS (source-time, in seconds):

PLAYER VOICE (microphone — the streamer's commentary):
{mic_dialogue}

GAME AUDIO (NPCs, events; may contain noise):
{game_dialogue}

=== TASK ===
You are the narrative planner for a short-form video editor. The downstream
trimmer is good at cutting filler but it picks segments without grasping
what makes the clip worth watching. Your job is to identify the moments
that must survive — the trimmer will receive your list as hard constraints
and is required to keep each moment fully intact.

Watch the video and read the transcripts together. Many of the best
moments are visual and have no dialogue (silent tension, jumpscares,
physical reactions). Don't let the absence of speech make you skip a
moment that obviously matters on screen.

Identify TWO things:

  A) HOOK — the opening beat that gives a first-time viewer a reason to
     keep watching past the first second or two. A hook can be:
       - a spoken line that promises payoff (a brag, a setup question)
       - a visible setup that creates curiosity (something approaching,
         a door, an unexplored area, a change in lighting)
       - silent tension where the absence of sound is itself the hook
         (common in horror — the breath, the pause, the wait)
     Try hard to find one. Returning hook=null should be rare; only do
     it when the clip genuinely starts in the middle of action with
     nothing before the action that would help a viewer orient.

  B) MUST_KEEP_MOMENTS — every distinct beat the cut would be diminished
     by losing. There is NO required count and NO required structure.
     A clip might have one moment (a single laugh) or six (a chain of
     jumpscares). Each moment can be any kind:
       - a laugh, a joke landing, a one-liner
       - a jumpscare, a death, a hit, a reveal
       - a tension peak (silent or spoken)
       - a satisfying win or save
       - a reaction worth seeing
       - anything else that would make a viewer feel something specific
         (surprise, fear, satisfaction, amusement, recognition)
     Be inclusive. If you're unsure whether a moment matters, keep it —
     the trimmer is allowed to decide which keep-segment includes it,
     but it cannot decide to drop it.

For HOOK and each MOMENT, give a tight [start, end] range that:
  - STARTS at a sentence/utterance boundary OR a visible action boundary
    (not mid-word, not mid-action)
  - ENDS after the moment fully resolves (~0.5–2s of breathing room)
  - Is as TIGHT as possible while still covering the full beat

Recommend a target_duration_range [min, max] in seconds. Default
[12, 40]. Widen up to 50 if there are 3+ distinct must-keep moments
that genuinely earn it.

=== HOOK STATUS ===
Set ``hook_status`` to one of:
  - "found"  — confident hook identified
  - "weak"   — best available hook, but it's a stretch
  - "absent" — clip has no usable hook (rare; explain in hook_status_reason)

If status is "found" or "weak", the hook range goes in BOTH the hook
field AND as the first entry in must_keep_moments (with kind="hook").
If status is "absent", omit the hook range entirely.

=== OUTPUT ===
Return ONLY this JSON, no other text, no code fences:

{{
  "story_summary": "One sentence: what the clip is about and what makes it work.",
  "hook": {{
    "start": 5.4,
    "end": 7.2,
    "kind": "hook",
    "rationale": "what the hook is and why it pulls a viewer in"
  }} | null,
  "hook_status": "found" | "weak" | "absent",
  "hook_status_reason": "one sentence explaining the status (especially if weak/absent)",
  "must_keep_moments": [
    {{
      "kind": "hook" | "laugh" | "jumpscare" | "tension_peak" | "reveal" | "kill" | "reaction" | "win" | "other",
      "start": 5.4,
      "end": 7.2,
      "rationale": "what happens here and why a viewer would feel something"
    }}
  ],
  "target_duration_range": [12, 35]
}}

Moments must be listed in chronological order and must not overlap. Do
NOT output any kind of "deletion hint" — your job is solely to identify
what must be kept; the trimmer decides what to drop based on the video
itself.

Based on the video and transcripts above, produce that JSON now.
"""


class NarrativePlanner:
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.client = get_gemini_client()
        self.safety_settings = get_safety_settings()
        self.log_func(
            f"🧭 Narrative planner initialized: {MODEL_PRO} "
            f"(thinking={THINKING_TRIM})"
        )

    def plan(
        self,
        video_path: str,
        video_duration: float,
        mic_transcriptions: Optional[List[str]] = None,
        desktop_transcriptions: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """Produce a narrative plan, or None on any failure.

        Callers MUST treat None as "no plan available" and proceed with the
        trimmer's blind-pick path. Never raises.
        """
        try:
            mic = self._format_transcription(mic_transcriptions or [])
            game = self._format_transcription(desktop_transcriptions or [])

            prompt = _PROMPT.format(
                mic_dialogue=mic,
                game_dialogue=game,
                video_duration=video_duration,
            )

            mime_type = self._mime_type_for(video_path)
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)

            if file_size <= INLINE_MAX_BYTES:
                self.log_func(
                    f"🧭 Planning narrative beats with video inline "
                    f"({file_size_mb:.1f} MB, {mime_type})"
                )
                response = self._generate_inline(
                    video_path, mime_type, prompt,
                )
            else:
                self.log_func(
                    f"🧭 Planning narrative beats via File API "
                    f"({file_size_mb:.1f} MB, {mime_type})"
                )
                response = self._generate_via_files(
                    video_path, mime_type, prompt,
                )

            text = getattr(response, "text", None) or ""
            if not text.strip():
                self.log_func("⚠️  Narrative planner returned empty response")
                return None

            plan = self._parse(text, video_duration)
            if plan is None:
                return None
            self._log_plan(plan)
            return plan
        except Exception as e:
            self.log_func(f"⚠️  Narrative planner failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return None

    # ── Gemini transports ────────────────────────────────────────────────
    def _generate_inline(
        self, video_path: str, mime_type: str, prompt: str,
    ):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_part = types.Part.from_bytes(
            data=video_bytes, mime_type=mime_type,
        )
        return self.client.models.generate_content(
            model=MODEL_PRO,
            contents=[prompt, video_part],
            config=types.GenerateContentConfig(
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_TRIM,
                ),
            ),
        )

    def _generate_via_files(
        self, video_path: str, mime_type: str, prompt: str,
    ):
        video_file = self.client.files.upload(
            file=video_path, config={"mime_type": mime_type},
        )
        try:
            poll_start = time.time()
            max_wait_seconds = 300
            while True:
                state = video_file.state.name
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise ValueError(
                        f"Gemini file processing FAILED: "
                        f"{getattr(video_file, 'error', 'no details')}"
                    )
                if time.time() - poll_start > max_wait_seconds:
                    raise TimeoutError(
                        f"Planner upload didn't become ACTIVE within "
                        f"{max_wait_seconds}s (last state: {state})"
                    )
                self.log_func(
                    f"   Waiting for ACTIVE (state: {state}) …")
                time.sleep(3)
                video_file = self.client.files.get(name=video_file.name)
            self.log_func(f"   ✅ File ACTIVE: {video_file.name}")

            file_part = types.Part.from_uri(
                file_uri=video_file.uri,
                mime_type=video_file.mime_type or mime_type,
            )
            return self.client.models.generate_content(
                model=MODEL_PRO,
                contents=[prompt, file_part],
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_TRIM,
                    ),
                ),
            )
        finally:
            try:
                self.client.files.delete(name=video_file.name)
            except Exception:
                pass

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _mime_type_for(path: str) -> str:
        """Mirror IntelligentTrimmer._mime_type_for — same set of accepted
        extensions so the planner can handle whatever the trimmer can."""
        ext = os.path.splitext(path)[1].lower()
        return {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".avi": "video/x-msvideo",
            ".webm": "video/webm",
            ".m4v": "video/x-m4v",
            ".mpeg": "video/mpeg",
            ".mpg": "video/mpeg",
            ".wmv": "video/x-ms-wmv",
            ".flv": "video/x-flv",
            ".3gp": "video/3gpp",
        }.get(ext, "video/mp4")

    @staticmethod
    def _format_transcription(transcriptions: List[str]) -> str:
        """Mirror IntelligentTrimmer._format_transcription so the planner
        sees the same dialogue view the trimmer does."""
        if not transcriptions:
            return "No dialogue detected"
        parsed: List[Tuple[float, str]] = []
        for line in transcriptions:
            try:
                time_part, text = line.split(":", 1)
                start_str, _ = time_part.split("-")
                parsed.append((float(start_str), text.strip()))
            except Exception:
                continue
        if not parsed:
            return "No valid dialogue"
        return "\n".join(f"[{ts:.1f}s] {text}" for ts, text in parsed)

    def _parse(self, text: str, video_duration: float) -> Optional[dict]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                self.log_func("⚠️  No JSON found in planner response")
                return None
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            self.log_func(f"⚠️  Planner JSON parse failed: {e}")
            return None

        # Hook handling
        hook_status = str(
            data.get("hook_status") or "absent"
        ).strip().lower()
        if hook_status not in ("found", "weak", "absent"):
            hook_status = "absent"
        hook_status_reason = str(data.get("hook_status_reason") or "")[:300]

        hook_obj = data.get("hook")
        hook = self._coerce_moment(hook_obj, video_duration, default_kind="hook") \
            if hook_status in ("found", "weak") else None

        # Must-keep moments
        moments: List[dict] = []
        for raw in data.get("must_keep_moments") or []:
            m = self._coerce_moment(raw, video_duration)
            if m is not None:
                moments.append(m)

        # Ensure the hook is also represented in must_keep_moments so the
        # trimmer's anchor-coverage check protects it. Prepend if missing.
        if hook is not None:
            already_has_hook = any(
                abs(m["start"] - hook["start"]) < 0.5
                and abs(m["end"] - hook["end"]) < 0.5
                for m in moments
            )
            if not already_has_hook:
                moments.insert(0, dict(hook))

        moments.sort(key=lambda x: x["start"])

        # Drop overlapping moments — the second one is contradictory as a
        # constraint. Keep the earlier one.
        deduped: List[dict] = []
        for m in moments:
            if deduped and m["start"] < deduped[-1]["end"]:
                self.log_func(
                    f"   ⚠ dropping overlapping moment "
                    f"{m['kind']}@{m['start']:.1f}-{m['end']:.1f}s"
                )
                continue
            deduped.append(m)

        if not deduped:
            self.log_func(
                "⚠️  Planner produced no must-keep moments — treating as no plan"
            )
            return None

        target_range = data.get("target_duration_range") or [12, 40]
        try:
            target_min = float(target_range[0])
            target_max = float(target_range[1])
        except (TypeError, ValueError, IndexError):
            target_min, target_max = 12.0, 40.0

        return {
            "story_summary": str(data.get("story_summary") or "")[:500],
            "hook": hook,
            "hook_status": hook_status,
            "hook_status_reason": hook_status_reason,
            "must_keep_moments": deduped,
            "target_duration_range": [target_min, target_max],
        }

    @staticmethod
    def _coerce_moment(
        raw, video_duration: float, default_kind: str = "other",
    ) -> Optional[dict]:
        if not isinstance(raw, dict):
            return None
        try:
            start = max(0.0, float(raw.get("start")))
            end = min(video_duration, float(raw.get("end")))
        except (TypeError, ValueError):
            return None
        if start >= end:
            return None
        return {
            "kind": str(raw.get("kind") or default_kind).strip().lower(),
            "start": start,
            "end": end,
            "rationale": str(raw.get("rationale") or "")[:300],
        }

    def _log_plan(self, plan: dict) -> None:
        summary = plan.get("story_summary") or ""
        if summary:
            self.log_func(f"   📖 {summary}")
        status = plan.get("hook_status", "absent")
        hook = plan.get("hook")
        if hook:
            self.log_func(
                f"   🪝 hook ({status}): "
                f"{hook['start']:6.1f}s – {hook['end']:6.1f}s "
                f"— {hook['rationale'][:120]}"
            )
        else:
            reason = plan.get("hook_status_reason") or "no reason given"
            self.log_func(f"   🪝 hook: ABSENT — {reason}")
        for m in plan.get("must_keep_moments", []):
            self.log_func(
                f"   ⚓ {m['kind']:<14} "
                f"{m['start']:6.1f}s – {m['end']:6.1f}s "
                f"({m['end'] - m['start']:4.1f}s) "
                f"— {m['rationale'][:100]}"
            )
        tr = plan.get("target_duration_range") or []
        if len(tr) == 2:
            self.log_func(
                f"   🎯 target duration: {tr[0]:.0f}–{tr[1]:.0f}s"
            )


# ── Validator + auto-repair ────────────────────────────────────────────────

def _moments(plan: Optional[dict]) -> List[dict]:
    """Plan-shape accessor. Returns the list of must-keep moments or []
    when the plan is missing/empty.
    """
    if not plan:
        return []
    moments = plan.get("must_keep_moments")
    if not isinstance(moments, list):
        return []
    return moments


def validate_segments_against_anchors(
    segments: List[Tuple[float, float]],
    plan: Optional[dict],
    tolerance: float = 0.25,
) -> List[dict]:
    """Return the list of must-keep moments NOT fully covered by any
    segment. ``tolerance`` allows a tiny float slack at the boundaries.

    Empty list = every required moment is satisfied. None/empty plan
    means "no constraints" → no violations.
    """
    moments = _moments(plan)
    if not moments:
        return []
    violations: List[dict] = []
    for m in moments:
        m_start = float(m["start"])
        m_end = float(m["end"])
        covered = any(
            (s - tolerance) <= m_start and m_end <= (e + tolerance)
            for s, e in segments
        )
        if not covered:
            violations.append(m)
    return violations


def repair_segments_for_anchors(
    segments: List[Tuple[float, float]],
    violations: List[dict],
    video_duration: float,
    log_func=print,
) -> List[Tuple[float, float]]:
    """Force every violated moment to be covered by extending the nearest
    overlapping segment, or appending a new one if none overlaps. Then
    merge any resulting overlaps and sort.
    """
    out: List[Tuple[float, float]] = [(float(s), float(e)) for s, e in segments]
    for m in violations:
        m_start = max(0.0, float(m["start"]))
        m_end = min(video_duration, float(m["end"]))
        if m_start >= m_end:
            continue

        merged_into_existing = False
        for i, (s, e) in enumerate(out):
            if not (e < m_start or s > m_end):  # overlaps
                new_s = min(s, m_start)
                new_e = max(e, m_end)
                out[i] = (new_s, new_e)
                merged_into_existing = True
                log_func(
                    f"   🔧 moment '{m.get('kind')}' "
                    f"{m_start:.1f}-{m_end:.1f}s: extended segment "
                    f"{i} to {new_s:.1f}-{new_e:.1f}s"
                )
                break
        if not merged_into_existing:
            out.append((m_start, m_end))
            log_func(
                f"   🔧 moment '{m.get('kind')}' "
                f"{m_start:.1f}-{m_end:.1f}s: appended as new segment"
            )

    return _merge_overlaps(out)


def _merge_overlaps(
    segments: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [segs[0]]
    for s, e in segs[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def format_anchors_for_prompt(plan: dict) -> str:
    """Render the must-keep moments as a prompt-injection block placed
    inside the trim prompt. Returns empty string if the plan has nothing
    to enforce.

    Note: deletion hints are NOT emitted. The planner only declares what
    must survive; the trimmer (which has the video) decides what to drop.
    """
    moments = _moments(plan)
    if not moments:
        return ""

    lines: List[str] = []
    lines.append("=== MUST-KEEP MOMENTS (HARD CONSTRAINTS) ===")
    summary = plan.get("story_summary") or ""
    if summary:
        lines.append(f"Story: {summary}")

    status = plan.get("hook_status")
    if status:
        reason = plan.get("hook_status_reason") or ""
        lines.append(f"Hook status: {status}{(' — ' + reason) if reason else ''}")

    lines.append(
        "A narrative planner watched the video and identified these "
        "moments as essential. You MUST keep every range below FULLY "
        "COVERED by at least one entry in segments_to_keep. You may "
        "extend segments to include surrounding context, and you may "
        "merge moments into a single segment, but you may not drop any."
    )
    for m in moments:
        lines.append(
            f"  - [{m['kind']}] {m['start']:.1f}–{m['end']:.1f}s "
            f"— {m['rationale']}"
        )

    target = plan.get("target_duration_range") or []
    if len(target) == 2:
        lines.append(
            f"Target final duration: {target[0]:.0f}–{target[1]:.0f}s "
            "(override the generic window above). The constraint is "
            "ABSOLUTE on must-keep moments — if covering them all forces "
            "the cut over the target maximum, exceed the target."
        )

    lines.append("")
    return "\n".join(lines)


def build_retry_addendum(violations: List[dict]) -> str:
    """Corrective text appended to the trim prompt on retry when the
    first response missed required moments.
    """
    bullets = "\n".join(
        f"  - [{m['kind']}] {m['start']:.1f}–{m['end']:.1f}s — {m['rationale']}"
        for m in violations
    )
    return (
        "\n\n=== CORRECTION REQUIRED ===\n"
        "Your previous segments_to_keep did NOT fully cover these "
        "must-keep moments:\n"
        f"{bullets}\n"
        "Produce a new JSON response where every moment above is fully "
        "contained within at least one entry of segments_to_keep. Extend "
        "an existing segment's bounds or add a new segment as needed. "
        "Keep all other selection logic the same.\n"
    )
