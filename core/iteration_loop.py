"""
Pre-publish iteration loop.

Closes the feedback loop with shorts_strategist: after each cut shorts-auto-editor
publishes editorial_decisions in metadata; the strategist reviews and either
writes edit directives or signals which iteration to ship; this orchestrator
applies the directives to a re-render and stops when the strategist says ship
(or after max_iterations is reached).

Wire format documented in shorts_strategist/gameplan.md ("Pre-publish
edit-review feedback loop"). All paths assume the canonical sibling layout —
shorts_strategist as a sibling directory to shorts-auto-editor — but the
directive directory is overridable via SHORTS_STRATEGIST_EDITS_DIR.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai_director.data_models import TimelineEvent

DEFAULT_MAX_ITERATIONS = 3
# 10 min — long enough that a backlogged thinker queue (e.g. 10+ stale
# tasks each taking ~20-60s for two-model reasoning) can drain past our
# specific edit-review task. The orchestrator also hits /thinker/force to
# bump our task to the front, so this rarely needs to fully exhaust.
DEFAULT_DIRECTIVE_TIMEOUT_S = 600
DEFAULT_POLL_INTERVAL_S = 3.0
DEFAULT_STRATEGIST_BASE_URL = "http://localhost:9022"


# ─── path resolution ─────────────────────────────────────────────────────────

def _default_edits_dir() -> str:
    """Resolve the strategist's edits/ directory under the sibling layout."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # shorts-auto-editor/
    parent = os.path.dirname(here)
    return os.path.join(parent, "shorts_strategist", "output", "recommendations", "edits")


def _edits_dir() -> str:
    return os.environ.get("SHORTS_STRATEGIST_EDITS_DIR") or _default_edits_dir()


def _directive_path(base: str) -> str:
    return os.path.join(_edits_dir(), f"{base}.json")


def _strategist_base_url() -> str:
    return (os.environ.get("SHORTS_STRATEGIST_URL")
            or DEFAULT_STRATEGIST_BASE_URL).rstrip("/")


def force_strategist_task(
    task_type: str,
    key: str,
    log_func: Callable[[str], None] = print,
) -> bool:
    """Tell the strategist to bump a specific task to the front of its queue.

    Best-effort: if the strategist isn't reachable we just log and continue —
    the wait_for_fresh_directive loop still works without prioritization, the
    orchestrator just has to wait longer.
    """
    url = f"{_strategist_base_url()}/thinker/force"
    body = json.dumps({"task_type": task_type, "key": key}).encode("utf-8")
    try:
        # urllib avoids a hard `requests` dependency on the shorts-auto-editor side.
        from urllib.request import Request, urlopen
        req = Request(url, data=body, method="POST",
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as resp:
            ok = 200 <= resp.status < 300
        if ok:
            log_func(f"   ↑ asked strategist to prioritize {task_type}/{key}")
        else:
            log_func(f"   ⚠ strategist /thinker/force returned non-2xx")
        return ok
    except Exception as e:
        log_func(f"   ⚠ couldn't reach strategist to prioritize task: {e}")
        return False


# ─── directive I/O ───────────────────────────────────────────────────────────

def read_directive(base: str) -> Optional[Dict[str, Any]]:
    """Read the strategist's directive file for a given metadata base name.

    Returns None if missing. Returns the artifact's payload (not the envelope)
    so callers can directly read verdict / edit_directives / ship_which_iteration.
    """
    path = _directive_path(base)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            envelope = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return envelope.get("payload") or envelope


def wait_for_fresh_directive(
    base: str,
    after_mtime: Optional[float],
    timeout_s: float = DEFAULT_DIRECTIVE_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    log_func: Callable[[str], None] = print,
) -> Optional[Dict[str, Any]]:
    """Poll for a directive whose mtime is newer than ``after_mtime``.

    Returns the directive payload, or None if the timeout elapsed without a
    fresh directive showing up. ``after_mtime=None`` accepts any directive
    that exists (used for the first iteration where no prior file existed).
    """
    path = _directive_path(base)
    deadline = time.time() + timeout_s
    last_log = 0.0
    while time.time() < deadline:
        if os.path.isfile(path):
            mtime = os.path.getmtime(path)
            if after_mtime is None or mtime > after_mtime:
                directive = read_directive(base)
                if directive is not None:
                    return directive
        if time.time() - last_log > 30:
            remaining = int(deadline - time.time())
            log_func(f"   ⏱  Waiting for strategist directive on {base}.json … "
                     f"({remaining}s remaining)")
            last_log = time.time()
        time.sleep(poll_interval_s)
    return None


# ─── pre-baked decisions container ───────────────────────────────────────────

@dataclass
class PreBakedDecisions:
    """Decisions reused on iteration > 1 to skip the expensive LLM phases.

    Built by the orchestrator from a prior iteration's editorial_decisions
    block (deserialized) plus the current iteration's edit_directives applied.
    """
    trim_segments: List[Tuple[float, float]]
    last_trim_data: Optional[Dict[str, Any]]
    onomatopoeia_events: List[Dict[str, Any]]
    decision_timeline: List[TimelineEvent]
    title_metadata: Optional[Dict[str, Any]] = None  # title + title_analysis + title_provenance
    narrative_plan: Optional[Dict[str, Any]] = None   # planner output, possibly modified by strategist's replan_anchors


# ─── deserialization helpers ─────────────────────────────────────────────────

def _trim_segments_from_decisions(decisions: Dict[str, Any]) -> List[Tuple[float, float]]:
    return [
        (float(s), float(e))
        for s, e in (decisions.get("trim_segments_kept") or [])
    ]


def _last_trim_data_from_decisions(decisions: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "punch_point_time":        decisions.get("trim_punch_point"),
        "punch_point_description": decisions.get("trim_punch_description"),
        "setup_rationale":         decisions.get("trim_setup_rationale"),
        "segments_to_keep": [
            {"start": s, "end": e, "reason": ""}
            for s, e in (decisions.get("trim_segments_kept") or [])
        ],
    }


def _timeline_from_decisions(decisions: Dict[str, Any]) -> List[TimelineEvent]:
    out: List[TimelineEvent] = []
    for d in decisions.get("zoom_timeline") or []:
        try:
            out.append(TimelineEvent(
                timestamp=float(d.get("time", 0.0)),
                action=d.get("action") or "no_action",
                duration=float(d.get("duration", 1.5) or 1.5),
                reason=str(d.get("reason") or ""),
                confidence=float(d.get("confidence", 0.5) or 0.5),
            ))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda e: e.timestamp)
    return out


def _narrative_plan_from_decisions(decisions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plan = decisions.get("narrative_plan")
    return plan if isinstance(plan, dict) else None


def _onomatopoeia_from_decisions(decisions: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in decisions.get("onomatopoeia_events") or []:
        peak = d.get("time")
        if peak is None:
            continue
        out.append({
            "word":              d.get("word") or "BUMP",
            "start_time":        float(peak),
            "end_time":          float(peak) + 0.7,
            "precise_peak_time": float(peak),
            "animation_type":    d.get("animation") or "drift_fade",
            "energy":            float(d.get("intensity", 0.5) or 0.5),
            "confidence":        0.9,
            "tier":              d.get("tier", "medium"),
            "context":           [],
            "onset_type":        "DIRECTIVE",
            "timing_offset":     0.0,
        })
    out.sort(key=lambda e: e["precise_peak_time"])
    return out


def _title_metadata_from_iteration(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "title" not in metadata:
        return None
    return {
        "title":            metadata.get("title"),
        "title_analysis":   metadata.get("title_analysis"),
        "title_provenance": metadata.get("title_provenance"),
    }


# ─── directive application ──────────────────────────────────────────────────

def _matches_time(a: float, b: float, tolerance: float = 0.5) -> bool:
    return abs(a - b) <= tolerance


def apply_directives(
    directives: Dict[str, Any],
    *,
    trim_segments: List[Tuple[float, float]],
    decision_timeline: List[TimelineEvent],
    onomatopoeia_events: List[Dict[str, Any]],
    narrative_plan: Optional[Dict[str, Any]] = None,
    log_func: Callable[[str], None] = print,
) -> Tuple[
    List[Tuple[float, float]],
    List[TimelineEvent],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    """Apply edit_directives to (trim, timeline, events, plan) in-memory.

    Each directive is best-effort: an unmatched timestamp on remove/replace is
    logged and skipped, not raised. The final timeline + events are sorted by
    time. ``narrative_plan`` is mutated by replan_anchors directives.
    """
    new_trim: List[Tuple[float, float]] = list(trim_segments or [])
    new_timeline: List[TimelineEvent] = list(decision_timeline or [])
    new_events: List[Dict[str, Any]] = list(onomatopoeia_events or [])
    new_plan: Optional[Dict[str, Any]] = (
        dict(narrative_plan) if isinstance(narrative_plan, dict) else None
    )

    # ── retrim ─────────────────────────────────────────────────────────────
    retrim = (directives or {}).get("retrim") or {}
    if retrim.get("should_retrim") and retrim.get("new_segments_to_keep"):
        new_trim = [
            (float(s), float(e))
            for s, e in retrim["new_segments_to_keep"]
        ]
        log_func(f"   ↻ retrim: {len(new_trim)} segments — {retrim.get('reason', '')[:120]}")

    # ── cinematography (zoom) changes ──────────────────────────────────────
    for change in (directives or {}).get("cinematography_changes") or []:
        ctype = change.get("type")
        at_time = change.get("at_time")
        reason = (change.get("reason") or "")[:160]
        if at_time is None:
            log_func(f"   ⚠ cinematography directive missing at_time: {change}")
            continue
        if ctype == "remove":
            before = len(new_timeline)
            new_timeline = [e for e in new_timeline if not _matches_time(e.timestamp, at_time)]
            if len(new_timeline) == before:
                log_func(f"   ⚠ no zoom matched at {at_time}s for remove")
            else:
                log_func(f"   ✂ removed zoom @{at_time:.1f}s — {reason}")
        elif ctype == "add" and change.get("to_action"):
            new_timeline.append(TimelineEvent(
                timestamp=float(at_time),
                action=change["to_action"],
                duration=float(change.get("duration") or 1.5),
                reason=f"strategist directive: {reason}",
                confidence=0.9,
            ))
            log_func(f"   ＋ added {change['to_action']} @{at_time:.1f}s — {reason}")
        elif ctype == "replace" and change.get("to_action"):
            replaced = False
            for i, e in enumerate(new_timeline):
                if _matches_time(e.timestamp, at_time):
                    new_timeline[i] = TimelineEvent(
                        timestamp=float(at_time),
                        action=change["to_action"],
                        duration=float(change.get("duration") or e.duration),
                        reason=f"strategist directive: {reason}",
                        confidence=0.9,
                    )
                    replaced = True
                    log_func(f"   ⇄ replaced zoom @{at_time:.1f}s → {change['to_action']} — {reason}")
                    break
            if not replaced:
                log_func(f"   ⚠ no zoom matched at {at_time}s for replace")
        else:
            log_func(f"   ⚠ unknown cinematography directive: {change}")

    new_timeline.sort(key=lambda e: e.timestamp)

    # ── onomatopoeia changes ───────────────────────────────────────────────
    for change in (directives or {}).get("onomatopoeia_changes") or []:
        ctype = change.get("type")
        at_time = change.get("at_time")
        reason = (change.get("reason") or "")[:160]
        if at_time is None:
            log_func(f"   ⚠ onomatopoeia directive missing at_time: {change}")
            continue
        if ctype == "remove":
            before = len(new_events)
            new_events = [
                ev for ev in new_events
                if not _matches_time(ev.get("precise_peak_time", ev.get("start_time", 0)), at_time)
            ]
            if len(new_events) == before:
                log_func(f"   ⚠ no onomatopoeia matched at {at_time}s for remove")
            else:
                log_func(f"   ✂ removed onomatopoeia @{at_time:.1f}s — {reason}")
        elif ctype == "add":
            word = change.get("word") or "BUMP"
            anim = change.get("animation") or "drift_fade"
            intensity = float(change.get("intensity", 0.5) or 0.5)
            new_events.append({
                "word":              word,
                "start_time":        float(at_time),
                "end_time":          float(at_time) + 0.7,
                "precise_peak_time": float(at_time),
                "animation_type":    anim,
                "energy":            intensity,
                "confidence":        0.9,
                "tier":              "directive",
                "context":           ["directive"],
                "onset_type":        "DIRECTIVE",
                "timing_offset":     0.0,
            })
            log_func(f"   ＋ added {word} ({anim}) @{at_time:.1f}s — {reason}")
        elif ctype == "move":
            # The wire format spec doesn't include a from_time on move; treat
            # as add-then-remove if the spec ever firms up. For now skip.
            log_func(f"   ⚠ onomatopoeia 'move' directive not yet supported: {change}")
        else:
            log_func(f"   ⚠ unknown onomatopoeia directive: {change}")

    new_events.sort(key=lambda e: e.get("precise_peak_time", e.get("start_time", 0.0)))

    # ── replan_anchors ─────────────────────────────────────────────────────
    # Strategist surgery on the narrative plan. Operates on
    # narrative_plan["must_keep_moments"] and (for "set_hook_status") the
    # hook fields. Each operation is best-effort.
    replan = (directives or {}).get("replan_anchors") or {}
    operations = replan.get("operations") or []
    if operations and new_plan is not None:
        moments = list(new_plan.get("must_keep_moments") or [])
        hook = new_plan.get("hook")
        for op in operations:
            otype = op.get("type")
            if otype == "add":
                m = {
                    "kind":      str(op.get("kind") or "other"),
                    "start":     float(op.get("start", 0.0) or 0.0),
                    "end":       float(op.get("end", 0.0) or 0.0),
                    "rationale": str(op.get("rationale") or "")[:300],
                }
                if m["start"] < m["end"]:
                    moments.append(m)
                    log_func(
                        f"   ＋ added must-keep '{m['kind']}' "
                        f"@{m['start']:.1f}-{m['end']:.1f}s"
                    )
                else:
                    log_func(f"   ⚠ skipped invalid add: {op}")
            elif otype == "remove":
                at = op.get("at_time")
                if at is None:
                    log_func(f"   ⚠ replan remove missing at_time: {op}")
                    continue
                before = len(moments)
                moments = [
                    m for m in moments
                    if not (float(m["start"]) <= float(at) <= float(m["end"]))
                ]
                if len(moments) == before:
                    log_func(
                        f"   ⚠ no must-keep moment overlapping "
                        f"{at}s for remove"
                    )
                else:
                    log_func(
                        f"   ✂ removed must-keep covering {at}s — "
                        f"{(op.get('reason') or '')[:120]}"
                    )
            elif otype == "replace":
                at = op.get("at_time")
                if at is None:
                    log_func(f"   ⚠ replan replace missing at_time: {op}")
                    continue
                replaced = False
                for i, m in enumerate(moments):
                    if float(m["start"]) <= float(at) <= float(m["end"]):
                        moments[i] = {
                            "kind":      str(op.get("kind", m["kind"])),
                            "start":     float(op.get("start", m["start"])),
                            "end":       float(op.get("end", m["end"])),
                            "rationale": str(op.get("rationale", m["rationale"]))[:300],
                        }
                        replaced = True
                        log_func(
                            f"   ⇄ replaced must-keep at {at}s → "
                            f"{moments[i]['kind']} "
                            f"{moments[i]['start']:.1f}-"
                            f"{moments[i]['end']:.1f}s"
                        )
                        break
                if not replaced:
                    log_func(
                        f"   ⚠ no must-keep moment overlapping "
                        f"{at}s for replace"
                    )
            elif otype == "set_hook":
                hook = {
                    "kind":      "hook",
                    "start":     float(op.get("start", 0.0) or 0.0),
                    "end":       float(op.get("end", 0.0) or 0.0),
                    "rationale": str(op.get("rationale") or "")[:300],
                }
                if hook["start"] >= hook["end"]:
                    hook = None
                    log_func(f"   ⚠ skipped invalid set_hook: {op}")
                else:
                    new_plan["hook_status"] = "found"
                    new_plan["hook_status_reason"] = str(
                        op.get("reason") or "set by strategist"
                    )[:300]
                    # Mirror into must_keep_moments
                    moments = [
                        m for m in moments
                        if not (m.get("kind") == "hook")
                    ]
                    moments.insert(0, dict(hook))
                    log_func(
                        f"   🪝 hook set by strategist: "
                        f"{hook['start']:.1f}-{hook['end']:.1f}s"
                    )
            elif otype == "clear_hook":
                hook = None
                new_plan["hook_status"] = "absent"
                new_plan["hook_status_reason"] = str(
                    op.get("reason") or "cleared by strategist"
                )[:300]
                moments = [m for m in moments if m.get("kind") != "hook"]
                log_func("   🪝 hook cleared by strategist")
            else:
                log_func(f"   ⚠ unknown replan_anchors operation: {op}")

        moments.sort(key=lambda m: float(m["start"]))
        new_plan["must_keep_moments"] = moments
        new_plan["hook"] = hook

    return new_trim, new_timeline, new_events, new_plan


# ─── pre-baked builder ──────────────────────────────────────────────────────

def build_pre_baked_for_next_iteration(
    prior_metadata: Dict[str, Any],
    directives: Dict[str, Any],
    log_func: Callable[[str], None] = print,
) -> PreBakedDecisions:
    """Produce a PreBakedDecisions for the next iteration by deserializing
    the prior iteration's editorial_decisions and applying the strategist's
    directives on top.
    """
    decisions = prior_metadata.get("editorial_decisions") or {}
    trim_segments = _trim_segments_from_decisions(decisions)
    timeline      = _timeline_from_decisions(decisions)
    events        = _onomatopoeia_from_decisions(decisions)
    plan          = _narrative_plan_from_decisions(decisions)

    trim_segments, timeline, events, plan = apply_directives(
        directives or {},
        trim_segments=trim_segments,
        decision_timeline=timeline,
        onomatopoeia_events=events,
        narrative_plan=plan,
        log_func=log_func,
    )

    return PreBakedDecisions(
        trim_segments=trim_segments,
        last_trim_data=_last_trim_data_from_decisions(decisions),
        decision_timeline=timeline,
        onomatopoeia_events=events,
        title_metadata=_title_metadata_from_iteration(prior_metadata),
        narrative_plan=plan,
    )


# ─── orchestrator ───────────────────────────────────────────────────────────

@dataclass
class _IterationState:
    iteration: int
    output_path: str
    metadata: Dict[str, Any]


class IterationOrchestrator:
    """Wraps VideoProcessor.process_single_video with the strategist feedback loop.

    Per video:
      1. iteration 1 runs the full pipeline, writes metadata.
      2. polls the strategist's edits/<base>.json for a fresh verdict.
      3. on verdict=needs_edits, builds pre-baked decisions, runs iteration N+1.
      4. on verdict=ship_*, picks the indicated iteration's output.
      5. caps at max_iterations regardless of the strategist's verdict.
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        directive_timeout_s: float = DEFAULT_DIRECTIVE_TIMEOUT_S,
        log_func: Callable[[str], None] = print,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.directive_timeout_s = float(directive_timeout_s)
        self.log_func = log_func

    # ── main loop ──────────────────────────────────────────────────────────
    def run(
        self,
        *,
        input_file: str,
        output_file_template: str,
        metadata_path: str,
        animation_type: str,
        sync_offset: float,
        detailed_logs: bool = False,
        final_output_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run the iteration loop on one video.

        ``output_file_template`` should contain ``{iter}`` to be replaced with
        the iteration number (e.g. ``finalvideo-as-2-v{iter}.mp4``).
        ``metadata_path`` is the canonical metadata file overwritten each
        iteration so the strategist sees a fresh hash.

        ``final_output_path`` (optional) is the canonical filename downstream
        consumers expect. When provided, after the ship iteration is chosen
        the orchestrator: (a) moves that iteration's file to
        final_output_path, (b) deletes the other iterations' files, and
        (c) updates metadata so file_info.output_filename matches.

        Returns (final_output_path, final_metadata_dict).
        """
        from core.video_processor import VideoProcessor

        base = self._base_for(metadata_path)
        states: Dict[int, _IterationState] = {}
        history: List[Dict[str, Any]] = []
        last_directive_mtime: Optional[float] = self._existing_directive_mtime(base)
        pre_baked: Optional[PreBakedDecisions] = None

        iteration = 1
        while iteration <= self.max_iterations:
            self.log_func("\n" + "=" * 60)
            self.log_func(f"ITERATION {iteration} of {self.max_iterations}")
            self.log_func("=" * 60)

            iter_output = output_file_template.replace("{iter}", str(iteration))
            final_path, _, metadata = VideoProcessor.process_single_video(
                input_file=input_file,
                output_file=iter_output,
                animation_type=animation_type,
                sync_offset=sync_offset,
                detailed_logs=detailed_logs,
                log_func=self.log_func,
                pre_baked=pre_baked,
            )
            if metadata is None:
                self.log_func(f"❌ Iteration {iteration} returned no metadata; aborting loop.")
                # Fall back to whatever the prior iteration produced.
                if iteration > 1:
                    prior = states[iteration - 1]
                    return self._finalize_and_return(
                        ship_state=prior, states=states,
                        metadata_path=metadata_path,
                        final_output_path=final_output_path,
                    )
                # No prior either — bail out without cleanup, nothing to ship.
                return final_path, {}

            # Patch iteration tracking + history into the metadata before write.
            metadata.setdefault("file_info", {})
            metadata["file_info"]["iteration"]      = iteration
            metadata["file_info"]["max_iterations"] = self.max_iterations
            metadata["file_info"]["iteration_history"] = list(history)
            self._write_metadata(metadata_path, metadata)

            states[iteration] = _IterationState(iteration=iteration,
                                                output_path=final_path,
                                                metadata=metadata)
            history.append({
                "iteration": iteration,
                "output_filename": os.path.basename(final_path),
                "trigger": ("initial_cut" if iteration == 1
                            else f"strategist_directive_v{iteration - 1}"),
                "editorial_decisions": metadata.get("editorial_decisions") or {},
            })

            # Final iteration → don't bother polling; pick best.
            if iteration >= self.max_iterations:
                self.log_func(f"⚠ Iteration cap reached at v{iteration}.")
                final = self._pick_final(base, states, default_iteration=iteration)
                return self._finalize_and_return(
                    ship_state=final, states=states,
                    metadata_path=metadata_path,
                    final_output_path=final_output_path,
                )

            # Ask the strategist to prioritize the edit-review task for this
            # specific metadata file. Without this, a backlogged queue can
            # delay the directive past the orchestrator's poll window.
            force_strategist_task(
                task_type="pre_publish_edit_review",
                key=base,
                log_func=self.log_func,
            )

            # Wait for the strategist to produce a verdict for this iteration.
            self.log_func(f"\n📡 Waiting for strategist verdict on iteration {iteration}…")
            directive = wait_for_fresh_directive(
                base=base,
                after_mtime=last_directive_mtime,
                timeout_s=self.directive_timeout_s,
                log_func=self.log_func,
            )
            if directive is None:
                self.log_func(f"⏱ No verdict after {int(self.directive_timeout_s)}s. "
                              f"Shipping v{iteration}.")
                return self._finalize_and_return(
                    ship_state=states[iteration], states=states,
                    metadata_path=metadata_path,
                    final_output_path=final_output_path,
                )
            last_directive_mtime = os.path.getmtime(_directive_path(base))

            verdict = directive.get("verdict")
            self.log_func(f"   verdict: {verdict}")

            if verdict in ("ship_current", "ship_prior"):
                ship_iter = directive.get("ship_which_iteration") or iteration
                state = states.get(ship_iter)
                if state is None:
                    self.log_func(f"⚠ ship_which_iteration={ship_iter} not in states; "
                                  f"shipping v{iteration} instead.")
                    state = states[iteration]
                self.log_func(f"✅ Shipping v{state.iteration}: "
                              f"{os.path.basename(state.output_path)}")
                return self._finalize_and_return(
                    ship_state=state, states=states,
                    metadata_path=metadata_path,
                    final_output_path=final_output_path,
                )

            if verdict == "needs_edits":
                edit_directives = directive.get("edit_directives") or {}
                self.log_func("\n📝 Applying strategist directives for next iteration:")
                pre_baked = build_pre_baked_for_next_iteration(
                    prior_metadata=metadata,
                    directives=edit_directives,
                    log_func=self.log_func,
                )
                iteration += 1
                continue

            if verdict == "awaiting_editorial_data":
                # The strategist's review couldn't run — likely because it
                # didn't see editorial_decisions on its tick. Shipping the
                # current cut is the right call.
                self.log_func("⚠ Strategist returned advisory mode; "
                              "shipping current.")
                return self._finalize_and_return(
                    ship_state=states[iteration], states=states,
                    metadata_path=metadata_path,
                    final_output_path=final_output_path,
                )

            self.log_func(f"⚠ Unknown verdict: {verdict!r}. Shipping current.")
            return self._finalize_and_return(
                ship_state=states[iteration], states=states,
                metadata_path=metadata_path,
                final_output_path=final_output_path,
            )

        # Shouldn't reach here, but be defensive.
        final = self._pick_final(base, states, default_iteration=iteration - 1)
        return self._finalize_and_return(
            ship_state=final, states=states,
            metadata_path=metadata_path,
            final_output_path=final_output_path,
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _base_for(self, metadata_path: str) -> str:
        name = os.path.basename(metadata_path)
        return name[:-5] if name.endswith(".json") else name

    def _existing_directive_mtime(self, base: str) -> Optional[float]:
        path = _directive_path(base)
        try:
            return os.path.getmtime(path) if os.path.isfile(path) else None
        except OSError:
            return None

    def _write_metadata(self, metadata_path: str, metadata: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        # Match the existing batch convention: a single-element list.
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([metadata], f, indent=2, default=str)

    def _finalize_and_return(
        self,
        *,
        ship_state: _IterationState,
        states: Dict[int, _IterationState],
        metadata_path: str,
        final_output_path: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Single exit point for the iteration loop.

        Renames the chosen iteration's file to the canonical
        ``final_output_path`` (when provided), deletes the other iterations'
        files, updates the metadata's ``file_info.output_filename`` +
        ``shipped_iteration`` + ``shipped_at``, rewrites the metadata file,
        and returns (final_path, final_metadata).

        Downstream consumers should read the canonical metadata file and
        treat ``file_info.output_filename`` as the only on-disk artifact.
        Files referenced in ``iteration_history`` are deleted by this method.
        """
        chosen_path = ship_state.output_path
        final_meta = dict(ship_state.metadata)

        # 1. Move chosen iteration to its canonical filename if we got one.
        if final_output_path and chosen_path != final_output_path:
            try:
                if os.path.exists(final_output_path):
                    os.remove(final_output_path)
                shutil.move(chosen_path, final_output_path)
                self.log_func(
                    f"   📦 Renamed v{ship_state.iteration} → "
                    f"{os.path.basename(final_output_path)}"
                )
                chosen_path = final_output_path
            except OSError as e:
                self.log_func(
                    f"   ⚠ Could not rename ship file: {e} "
                    f"(keeping {os.path.basename(chosen_path)})"
                )

        # 2. Delete every other iteration's file.
        # Two passes: (a) the in-memory states we know about, (b) a glob of
        # the output dir for any sibling -v<N> files that match the iteration
        # template. The glob pass catches the case where state.output_path
        # drifted from what's actually on disk (e.g. an earlier failed run
        # left v1 around but the current run's `states` only knows about
        # v1/v2 of the new attempt). Without it, those orphans persist.
        import glob as _glob
        deleted: List[str] = []
        already_handled: set = set()

        # Pass (a): tracked iterations.
        for it_num, state in states.items():
            if state.output_path == chosen_path:
                already_handled.add(os.path.abspath(state.output_path))
                continue
            already_handled.add(os.path.abspath(state.output_path))
            if not os.path.exists(state.output_path):
                continue
            try:
                os.remove(state.output_path)
                deleted.append(f"v{it_num} ({os.path.basename(state.output_path)})")
            except OSError as e:
                self.log_func(f"   ⚠ Could not delete v{it_num}: {e}")

        # Pass (b): glob the iteration sibling pattern. Reconstruct the
        # template from any tracked state (or from chosen_path if it still
        # ends in -v<N>). Pattern: <stem>-v[0-9]*<ext>.
        sample_path: Optional[str] = None
        for state in states.values():
            if "-v" in os.path.basename(state.output_path):
                sample_path = state.output_path
                break
        if sample_path is None and "-v" in os.path.basename(chosen_path):
            sample_path = chosen_path
        if sample_path:
            sample_dir = os.path.dirname(sample_path)
            sample_name = os.path.basename(sample_path)
            sample_stem, sample_ext = os.path.splitext(sample_name)
            # Strip the -v<digits> suffix to recover the canonical stem.
            m = re.match(r"^(.*)-v\d+$", sample_stem)
            if m:
                base_stem = m.group(1)
                pattern = os.path.join(
                    sample_dir, f"{base_stem}-v[0-9]*{sample_ext}"
                )
                chosen_abs = os.path.abspath(chosen_path)
                for orphan in _glob.glob(pattern):
                    orphan_abs = os.path.abspath(orphan)
                    if orphan_abs == chosen_abs:
                        continue
                    if orphan_abs in already_handled:
                        continue
                    try:
                        os.remove(orphan)
                        deleted.append(f"orphan ({os.path.basename(orphan)})")
                    except OSError as e:
                        self.log_func(
                            f"   ⚠ Could not delete orphan "
                            f"{os.path.basename(orphan)}: {e}"
                        )

        if deleted:
            self.log_func(f"   🗑  Cleaned up: {', '.join(deleted)}")
        else:
            self.log_func(
                f"   🗑  No iteration leftovers to clean "
                f"(shipped: {os.path.basename(chosen_path)})"
            )

        # 3. Patch metadata to point at the canonical file + record ship info.
        file_info = dict(final_meta.get("file_info") or {})
        file_info["output_filename"]   = os.path.basename(chosen_path)
        file_info["shipped_iteration"] = ship_state.iteration
        file_info["shipped_at"]        = datetime.datetime.now().isoformat()
        final_meta["file_info"] = file_info
        self._write_metadata(metadata_path, final_meta)

        return chosen_path, final_meta

    def _pick_final(
        self,
        base: str,
        states: Dict[int, _IterationState],
        default_iteration: int,
    ) -> _IterationState:
        """If a directive exists asking us to ship a specific iteration, honor it.
        Otherwise fall back to the requested default."""
        directive = read_directive(base)
        if directive:
            target = directive.get("ship_which_iteration")
            if isinstance(target, int) and target in states:
                self.log_func(f"   strategist requested ship of v{target}")
                return states[target]
        return states.get(default_iteration) or next(iter(states.values()))
