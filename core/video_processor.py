# core/video_processor.py
import os
import shutil
import tempfile
import gc
import datetime
from typing import Optional

from onomatopoeia_detector import OnomatopoeiaDetector
import core.transcriber
from core.subtitle_converter import convert_to_srt
from core.subtitle_embedder import embed_subtitles
from ai_director.master_director import MasterDirector
from ai_director.video_editor import VideoEditor
from video_utils import get_video_duration
from clip_editor.intelligent_trimmer import IntelligentTrimmer
from clip_editor.narrative_planner import NarrativePlanner, is_planner_enabled
from utils.timestamp_processor import (
    extend_segments_for_dialogue,
    shift_transcriptions_to_output_time,
)
from title_generator import TitleGenerator


def _mic_has_no_real_speech(mic_audio_path, log_func):
    """Double-check an empty mic transcript against the raw mic audio.

    Called only when the mic transcription came back with zero real
    (non-hallucination) words. An empty transcript can mean two very
    different things — the operator genuinely said nothing, OR a flaky
    transcription / over-aggressive hallucination filter ate real speech —
    so we cross-check against the actual audio before deciding to skip.

    Returns:
      True  → the mic is silent or contains no speech  → SAFE to skip.
      False → the mic has clear speech-like energy      → do NOT skip
              (don't drop a clip the operator may have talked over).
      None  → couldn't analyze (no librosa / no file)   → do NOT skip
              (never skip on an unverifiable empty transcript).
    """
    try:
        import numpy as np
        from utils.hallucination_filter import (
            LIBROSA_AVAILABLE,
            _load_audio_segment,
            _speech_likelihood,
        )
    except Exception:
        return None

    if not LIBROSA_AVAILABLE or not mic_audio_path or not os.path.exists(mic_audio_path):
        return None

    # Load the whole mic track (16 kHz mono WAV — small).
    audio = _load_audio_segment(mic_audio_path, 0.0, 36000.0)
    if audio is None or len(audio) == 0:
        return True  # no audio at all → no speech

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.01:
        log_func(f"   [no-dialogue check] mic RMS {rms:.4f} < 0.01 → silent")
        return True

    # There IS energy on the mic — is it speech, or just bleed/noise?
    speech = _speech_likelihood(audio)
    log_func(
        f"   [no-dialogue check] mic RMS {rms:.4f}, "
        f"speech_likelihood {speech:.2f}"
    )
    if speech >= 0.35:
        return False  # sounds like speech → don't trust the empty transcript
    return True       # energy present but not speech-like → skip


def _no_dialogue_skip_metadata(input_file, output_file):
    """Minimal metadata record for a clip skipped for having no dialogue.

    Carries status="skipped_no_dialogue" so callers don't ship/upload it and
    the operator has a trace of why nothing was produced. No output file is
    written for a skipped clip.
    """
    return {
        "file_info": {
            "original_filename": os.path.basename(input_file),
            "output_filename": os.path.basename(output_file),
            "processed_at": datetime.datetime.now().isoformat(),
            "iteration": 1,
            "max_iterations": 3,
            "iteration_history": [],
        },
        "status": "skipped_no_dialogue",
        "skip_reason": (
            "No mic dialogue detected (verified against the mic audio); "
            "clip not edited."
        ),
    }


class VideoProcessor:
    """
    Full per-video processing pipeline.

    Phases (trim-first order — everything after PHASE 2 operates in
    output/trimmed time, so all editorial decisions are unambiguously about
    the shipped clip):

      1. Dialogue transcription (mic + game tracks) — source time
      2. Intelligent trim analysis              (Gemini 3.1 Pro, high thinking)
      2a. Apply trim → trimmed.mp4              (ffmpeg)
      2b. Shift transcriptions to output time   (utils helper)
      3. Onomatopoeia detection on trimmed.mp4  (audio + Gemini 3 Flash vision)
      4. AI Director on trimmed.mp4             (zoom choreography)
      5. Apply zoom edits                       (ffmpeg)
      6. Subtitle embedding                     (ffmpeg)
      7. Title generation + metadata

    Title generation was moved out to a separate tool and is no longer
    part of this pipeline.
    """

    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        animation_type: str,
        sync_offset: float,
        detailed_logs: bool,
        log_func,
        enable_trimming: bool = True,
        pre_baked=None,  # Optional[core.iteration_loop.PreBakedDecisions]
        camera_region=None,  # Optional[dict] normalized {x,y,w,h} for zoom focus
        gameplay_region=None,  # Optional[dict] normalized {x,y,w,h} for zoom focus
        camera_mode: str = "vtuber",          # "vtuber" | "facecam" | "none"
        game_subtitles_enabled: bool = True,
        onomatopoeia_enabled: bool = True,
        mic_track_index: str = "a:1",
        game_track_index: str = "a:2",
        mic_margin_v=None,                    # ASS MarginV override for mic subs
        game_margin_v=None,                   # ASS MarginV override for game subs
    ):
        temp_dir = tempfile.gettempdir()
        trim_segments = None
        extended_trim_segments: list = []
        last_trim_data: Optional[dict] = None
        narrative_plan: Optional[dict] = None
        decision_timeline = None
        events = []
        video_metadata = None
        DIALOGUE_TRIM_BUFFER = 0.8

        # Holders for cleanup
        mic_audio_path_for_analysis = None
        desktop_audio_path_for_analysis = None
        mic_subtitle_path = None
        desktop_subtitle_path = None
        onomatopoeia_subtitle_path = None
        edited_video_path = None
        trimmed_video_path = None

        try:
            log_func("=" * 60)
            log_func(
                f"STARTING FULL VIDEO PROCESSING: "
                f"{os.path.basename(input_file)}"
            )
            log_func("=" * 60)

            # ── PHASE 1: Dialogue Transcription (source time) ──────────
            # Word timestamps come back referenced to the full source. We'll
            # shift them to output time after PHASE 2a applies the trim.
            log_func("\n--- PHASE 1: Dialogue Transcription ---")

            mic_transcriptions_raw = []
            mic_transcriptions_adjusted = []
            desktop_transcriptions_raw = []
            desktop_transcriptions_adjusted = []

            mic_audio_path = os.path.join(
                temp_dir, f"{os.path.basename(input_file)}_mic.wav")
            if core.transcriber.convert_to_audio(
                input_file, mic_audio_path, mic_track_index, log_func
            ):
                (
                    mic_transcriptions_raw,
                    mic_transcriptions_adjusted,
                ) = core.transcriber.transcribe_audio(
                    "large", "cpu", mic_audio_path, True, log_func,
                    "English", "Track 2 (Mic)",
                )
                log_func(
                    f"✅ Mic transcription complete: "
                    f"{len(mic_transcriptions_raw)} words"
                )

                # ── No-dialogue gate (double-checked) ─────────────────
                # If the operator never spoke (no real mic words survived
                # the hallucination filter), this clip isn't worth editing
                # into a short. But an empty transcript can also be a flaky
                # pass, so we verify against the raw mic audio first and only
                # skip when the mic is genuinely silent/non-speech. The mic
                # WAV must still exist for that check, so deletion happens
                # after it. (Skipped on the replay path — already decided.)
                if pre_baked is None and not mic_transcriptions_raw:
                    verdict = _mic_has_no_real_speech(mic_audio_path, log_func)
                    try:
                        os.remove(mic_audio_path)
                    except Exception:
                        pass
                    if verdict is True:
                        log_func(
                            "⏭️  No mic dialogue detected (double-checked "
                            "against audio) — skipping edit for this clip."
                        )
                        return (
                            output_file, None,
                            _no_dialogue_skip_metadata(input_file, output_file),
                        )
                    if verdict is False:
                        log_func(
                            "   ⚠ Mic transcript empty but the audio sounds "
                            "like speech — editing anyway (won't drop a clip "
                            "you may have talked over)."
                        )
                    else:
                        log_func(
                            "   ⚠ Couldn't verify mic audio — editing anyway "
                            "(never skip on an unverifiable empty transcript)."
                        )
                else:
                    try:
                        os.remove(mic_audio_path)
                    except Exception:
                        pass
            else:
                log_func("⚠️ Mic audio extraction failed")

            desktop_audio_path = os.path.join(
                temp_dir, f"{os.path.basename(input_file)}_desktop.wav")
            if core.transcriber.convert_to_audio(
                input_file, desktop_audio_path, game_track_index, log_func
            ):
                (
                    desktop_transcriptions_raw,
                    desktop_transcriptions_adjusted,
                ) = core.transcriber.transcribe_audio(
                    "large", "cpu", desktop_audio_path, True, log_func,
                    "English", "Track 3 (Desktop)",
                )
                try:
                    os.remove(desktop_audio_path)
                except Exception:
                    pass
                log_func(
                    f"✅ Desktop transcription complete: "
                    f"{len(desktop_transcriptions_raw)} words"
                )
            else:
                log_func("⚠️ Desktop audio extraction failed")

            source_video_duration = get_video_duration(input_file, log_func)

            # ── PHASE 2: Trim plan ─────────────────────────────────────
            # Either compute it fresh (iteration 1) or replay the prior
            # iteration's plan as-amended-by-strategist (iteration N>1).
            if pre_baked is not None:
                log_func("\n--- PRE-BAKED REPLAY (iteration > 1) ---")
                trim_segments     = list(pre_baked.trim_segments or [])
                last_trim_data    = pre_baked.last_trim_data
                events            = list(pre_baked.onomatopoeia_events or [])
                decision_timeline = list(pre_baked.decision_timeline or [])
                narrative_plan    = pre_baked.narrative_plan
                # The prior iteration already ran extend_segments_for_dialogue
                # and any directive-driven retrim — replay verbatim.
                extended_trim_segments = list(trim_segments)
                log_func(
                    f"   Replaying {len(trim_segments)} trim segments, "
                    f"{len(decision_timeline)} zoom events, "
                    f"{len(events)} onomatopoeia events"
                )
                if narrative_plan:
                    log_func(
                        f"   Carrying forward narrative plan with "
                        f"{len(narrative_plan.get('must_keep_moments', []))} "
                        "must-keep moments"
                    )
            else:
                # ── PHASE 1b: Narrative planning (pre-trim) ──────────────
                # Best-effort: identifies hook/setup/payoff anchors that the
                # trimmer must keep fully covered. Disabled via env var
                # SHORTS_NARRATIVE_PLANNER=0; falls back to None on any
                # failure. When narrative_plan is None the trimmer behaves
                # exactly as before this feature was added.
                narrative_plan = None
                if enable_trimming and is_planner_enabled():
                    log_func("\n--- PHASE 1b: Narrative Planning ---")
                    try:
                        narrative_plan = NarrativePlanner(
                            log_func=log_func
                        ).plan(
                            video_path=input_file,
                            video_duration=source_video_duration,
                            mic_transcriptions=mic_transcriptions_raw,
                            desktop_transcriptions=desktop_transcriptions_raw,
                        )
                    except Exception as e:
                        log_func(f"⚠️ Narrative planner crashed: {e}")
                        narrative_plan = None
                    if narrative_plan is None:
                        log_func(
                            "   (proceeding without moments — trimmer "
                            "will pick blind)"
                        )

                log_func("\n--- PHASE 2: Intelligent Trimming Analysis ---")
                if enable_trimming:
                    trimmer = IntelligentTrimmer(log_func=log_func)
                    trim_segments = trimmer.analyze_for_trim(
                        video_path=input_file,
                        mic_transcriptions=mic_transcriptions_raw,
                        desktop_transcriptions=desktop_transcriptions_raw,
                        narrative_plan=narrative_plan,
                    )
                    if trim_segments:
                        total_kept = sum(e - s for s, e in trim_segments)
                        log_func(
                            f"✅ Trim plan ready: {len(trim_segments)} segments, "
                            f"{total_kept:.1f}s total"
                        )
                        last_trim_data = getattr(trimmer, "last_parsed_data", None)

                        # Extend the trim window to protect dialogue runs.
                        # Done here (instead of post-edit) because the trim
                        # is now applied BEFORE the director sees the video.
                        mic_audio_path_for_analysis = os.path.join(
                            temp_dir,
                            f"{os.path.basename(input_file)}_mic_analysis.wav",
                        )
                        core.transcriber.convert_to_audio(
                            input_file, mic_audio_path_for_analysis,
                            mic_track_index, log_func,
                        )
                        if desktop_transcriptions_raw:
                            desktop_audio_path_for_analysis = os.path.join(
                                temp_dir,
                                f"{os.path.basename(input_file)}_desktop_analysis.wav",
                            )
                            core.transcriber.convert_to_audio(
                                input_file, desktop_audio_path_for_analysis,
                                game_track_index, log_func,
                            )

                        extended_trim_segments = extend_segments_for_dialogue(
                            segments_to_keep=trim_segments,
                            raw_mic_transcriptions=mic_transcriptions_raw,
                            raw_desktop_transcriptions=desktop_transcriptions_raw,
                            log_func=log_func,
                            max_extension_seconds=3.0,
                            buffer_seconds=DIALOGUE_TRIM_BUFFER,
                            mic_audio_path=mic_audio_path_for_analysis,
                            desktop_audio_path=desktop_audio_path_for_analysis,
                        )
                    else:
                        log_func(
                            "⚠️ No trim decisions made, will keep original video")
                else:
                    log_func("   Trimming disabled - skipping analysis")

            # ── PHASE 2a: Apply trim → working video ───────────────────
            # Everything from here on operates on the trimmed video, in
            # output time. This is the single change that fixes the
            # source-time/output-time coordinate mismatch — directives,
            # zoom timestamps, onomatopoeia events all live in one frame
            # of reference now.
            video_to_process = input_file
            if extended_trim_segments:
                log_func("\n--- PHASE 2a: Applying trim to working video ---")
                trimmed_video_path = os.path.join(
                    temp_dir,
                    f"{os.path.basename(input_file)}_trimmed.mp4",
                )
                applier = IntelligentTrimmer(log_func=log_func)
                trim_success = applier.apply_trim(
                    input_video=input_file,
                    output_video=trimmed_video_path,
                    segments_to_keep=extended_trim_segments,
                )
                if trim_success and os.path.exists(trimmed_video_path):
                    video_to_process = trimmed_video_path
                    trimmed_dur = get_video_duration(trimmed_video_path, log_func)
                    log_func(
                        f"✅ Trim applied — working video {trimmed_dur:.1f}s "
                        f"(from {source_video_duration:.1f}s source)"
                    )
                else:
                    log_func("⚠️ Trim execution failed — falling back to source")
                    trimmed_video_path = None

            video_duration = get_video_duration(video_to_process, log_func)

            # ── PHASE 2b: Shift transcriptions to output time ──────────
            # Word timestamps were captured in source time. If we trimmed,
            # shift them so subtitles line up with the trimmed video and so
            # the director sees mic dialogue in the same frame of reference
            # as the audio events it generates.
            if extended_trim_segments and video_to_process != input_file:
                log_func("   ↳ Shifting transcriptions to output time")
                mic_transcriptions_raw = shift_transcriptions_to_output_time(
                    mic_transcriptions_raw, extended_trim_segments)
                mic_transcriptions_adjusted = shift_transcriptions_to_output_time(
                    mic_transcriptions_adjusted, extended_trim_segments)
                desktop_transcriptions_raw = shift_transcriptions_to_output_time(
                    desktop_transcriptions_raw, extended_trim_segments)
                desktop_transcriptions_adjusted = shift_transcriptions_to_output_time(
                    desktop_transcriptions_adjusted, extended_trim_segments)
                log_func(
                    f"   Mic words after shift: "
                    f"{len(mic_transcriptions_raw)} | "
                    f"Desktop words after shift: "
                    f"{len(desktop_transcriptions_raw)}"
                )

            # Build subtitle SRT/ASS files now (in output time).
            if mic_transcriptions_adjusted:
                mic_subtitle_path_srt = os.path.join(
                    temp_dir, f"{os.path.basename(input_file)}_mic.srt")
                convert_to_srt(
                    "\n".join(mic_transcriptions_adjusted),
                    mic_subtitle_path_srt,
                    video_to_process,
                    log_func,
                    is_mic_track=True,
                    mic_margin_v=mic_margin_v,
                )
                mic_subtitle_path = mic_subtitle_path_srt.replace(
                    ".srt", ".ass")

            # Game/desktop subtitles — operator-toggleable.
            if game_subtitles_enabled and desktop_transcriptions_adjusted:
                desktop_subtitle_path = os.path.join(
                    temp_dir, f"{os.path.basename(input_file)}_desktop.srt")
                convert_to_srt(
                    "\n".join(desktop_transcriptions_adjusted),
                    desktop_subtitle_path,
                    video_to_process,
                    log_func,
                )
            else:
                if not game_subtitles_enabled:
                    log_func("   Game subtitles disabled — skipping desktop track")
                desktop_subtitle_path = None

            # ── PHASE 3: Onomatopoeia Detection (on trimmed clip) ──────
            subtitle_ext = ".ass" if animation_type != "Static" else ".srt"
            onomatopoeia_subtitle_path = os.path.join(
                temp_dir,
                f"{os.path.basename(video_to_process)}_ono{subtitle_ext}",
            )
            if pre_baked is not None:
                # Replay path: events already in output time. Render the words
                # only when onomatopoeia is enabled (events still feed zooms).
                if events and onomatopoeia_enabled:
                    from subtitle_generator import SubtitleGenerator
                    SubtitleGenerator(log_func=log_func).create_subtitle_file(
                        events, onomatopoeia_subtitle_path, animation_type)
                else:
                    onomatopoeia_subtitle_path = None
            else:
                log_func("\n--- PHASE 3: Onomatopoeia Detection ---")
                detector = OnomatopoeiaDetector(
                    log_func=log_func, game_track_index=game_track_index)
                detector.fusion_engine.sync_offset = sync_offset
                # Always detect — the AI Director uses `events` as a game-audio
                # signal even when the onomatopoeia overlay is turned off.
                events, video_map = detector.analyze_file(
                    video_to_process, animation_type, sync_offset=sync_offset)
                if onomatopoeia_enabled:
                    detector.subtitle_generator.create_subtitle_file(
                        events, onomatopoeia_subtitle_path, animation_type)
                else:
                    log_func("   Onomatopoeia overlay disabled — words hidden, "
                             "events still feed zoom decisions")
                    onomatopoeia_subtitle_path = None
                del detector
                gc.collect()

                # ── PHASE 4: AI Director (on trimmed clip) ─────────────
                log_func("\n--- PHASE 4: AI Director Editing ---")
                director = MasterDirector(
                    log_func=log_func, detailed_logs=detailed_logs)
                decision_timeline = director.analyze_video_and_create_timeline(
                    video_path=video_to_process,
                    video_duration=video_duration,
                    mic_transcription=mic_transcriptions_raw,
                    audio_events=events,
                    video_analysis_map=video_map,
                    camera_region=camera_region,
                    camera_mode=camera_mode,
                    mic_track_index=mic_track_index,
                    game_track_index=game_track_index,
                )

            # ── PHASE 5: Apply zoom edits ──────────────────────────────
            video_to_subtitle = video_to_process
            if decision_timeline:
                # Enforce the camera-gaze constraint on any cam zooms the
                # strategist added/replaced (iteration > 1). The director's own
                # cam reactions were already gaze-gated; this catches the
                # thinker's edits so a bugged avatar never gets punched into.
                from ai_director.master_director import (
                    enforce_gaze_on_strategist_cam,
                )
                decision_timeline = enforce_gaze_on_strategist_cam(
                    decision_timeline, video_to_process, camera_region, log_func,
                    camera_mode=camera_mode)

            if decision_timeline:
                editor = VideoEditor(
                    log_func=log_func,
                    camera_region=camera_region,
                    gameplay_region=gameplay_region,
                )
                edited_video_path = os.path.join(
                    temp_dir,
                    f"{os.path.basename(video_to_process)}_edited.mp4",
                )
                editor.apply_edits(
                    input_video=video_to_process,
                    output_video=edited_video_path,
                    timeline=decision_timeline,
                )
                video_to_subtitle = edited_video_path

            # ── PHASE 6: Embed Subtitles → final output ────────────────
            log_func("\n--- PHASE 6: Embedding All Subtitles ---")
            embed_subtitles(
                input_video=video_to_subtitle,
                output_video=output_file,
                track2_srt=mic_subtitle_path,
                track3_srt=desktop_subtitle_path,
                onomatopoeia_srt=onomatopoeia_subtitle_path,
                onomatopoeia_events=events,
                log=log_func,
                mic_margin_v=mic_margin_v,
                game_margin_v=game_margin_v,
            )
            log_func("✅ Subtitles embedded — final output written")

            # ── PHASE 7: Title generation + metadata ──────────────────
            log_func("\n--- PHASE 7: Title Generation ---")
            final_duration_value = (
                get_video_duration(output_file, log_func)
                if os.path.exists(output_file) else None
            )
            # Assemble editorial_decisions for shorts_strategist's
            # pre_publish_edit_review task. Wire format documented in
            # shorts_strategist/gameplan.md ("Pre-publish edit-review feedback
            # loop"). The strategist scores this block, may emit edit
            # directives we re-apply on a subsequent iteration.
            final_trim_segments = (
                extended_trim_segments if extended_trim_segments
                else (trim_segments or [])
            )
            editorial_decisions = {
                "trim_segments_kept": [
                    [float(s), float(e)] for s, e in (final_trim_segments or [])
                ],
                "trim_punch_point": (last_trim_data or {}).get("punch_point_time"),
                "trim_punch_description": (last_trim_data or {}).get("punch_point_description"),
                "trim_setup_rationale": (last_trim_data or {}).get("setup_rationale"),
                # The narrative plan that drove this iteration's trim. Stays
                # the same across iterations unless the strategist issues a
                # replan_anchors directive. Strategist consumes this to
                # score moment_coverage and decide whether anchors need
                # adjustment for a future iteration.
                "narrative_plan": narrative_plan,
                "zoom_timeline": [
                    {
                        "time":       float(getattr(ev, "timestamp", 0.0)),
                        "action":     getattr(ev, "action", None),
                        "duration":   float(getattr(ev, "duration", 0.0)),
                        "reason":     getattr(ev, "reason", None),
                        "confidence": float(getattr(ev, "confidence", 0.0)),
                    }
                    for ev in (decision_timeline or [])
                ],
                "onomatopoeia_events": [
                    {
                        "time":      float(ev.get("precise_peak_time", ev.get("start_time", 0.0)) or 0.0),
                        "word":      ev.get("word"),
                        "animation": ev.get("animation_type"),
                        "intensity": float(ev.get("energy", 0.0) or 0.0),
                        "tier":      ev.get("tier"),
                    }
                    for ev in (events or [])
                ],
            }

            video_metadata = {
                "file_info": {
                    "original_filename": os.path.basename(input_file),
                    "output_filename": os.path.basename(output_file),
                    "processed_at": datetime.datetime.now().isoformat(),
                    "original_duration": source_video_duration,
                    "final_duration": final_duration_value,
                    # Iteration tracking for the pre-publish edit-review loop.
                    # shorts-auto-editor owns these counters and enforces the cap;
                    # shorts_strategist scores each iteration and recommends
                    # re-edits up to max_iterations times.
                    "iteration": 1,
                    "max_iterations": 3,
                    "iteration_history": [],
                },
                "editorial_decisions": editorial_decisions,
            }

            if pre_baked is not None and pre_baked.title_metadata:
                # Reuse the prior iteration's title — title doesn't depend on
                # editorial decisions, so re-running TitleGenerator would
                # waste a Gemini call.
                tm = pre_baked.title_metadata
                if tm.get("title"):
                    video_metadata["title"] = tm["title"]
                if tm.get("title_analysis"):
                    video_metadata["title_analysis"] = tm["title_analysis"]
                if tm.get("title_provenance"):
                    video_metadata["title_provenance"] = tm["title_provenance"]
                log_func("   ↪ Reusing prior iteration's title (replay path)")
            else:
                try:
                    title_result = TitleGenerator(log_func=log_func).generate(
                        mic_transcriptions_raw=mic_transcriptions_raw,
                        desktop_transcriptions_raw=desktop_transcriptions_raw,
                        original_duration=source_video_duration,
                        final_duration=final_duration_value,
                        trim_segments=trim_segments,
                    )
                except Exception as title_err:
                    # Strict no-fallback: any unexpected crash in title gen
                    # is logged, but the cut still ships without a title.
                    log_func(f"[title] unexpected error: {title_err}")
                    title_result = None

                if title_result:
                    video_metadata["title"] = title_result["text"]
                    video_metadata["title_analysis"] = title_result["analysis"]
                    video_metadata["title_provenance"] = title_result["provenance"]
            log_func("✅ Metadata generated (queued for batch file)")

        except Exception as e:
            log_func(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
            if not os.path.exists(output_file) and os.path.exists(input_file):
                shutil.copy2(input_file, output_file)
        finally:
            log_func("\n--- Cleaning up temporary files ---")
            for p in (
                onomatopoeia_subtitle_path,
                mic_subtitle_path,
                desktop_subtitle_path,
                edited_video_path,
                trimmed_video_path,
                mic_audio_path_for_analysis,
                desktop_audio_path_for_analysis,
            ):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        # Preserve the existing 3-tuple return signature (callers in main.py
        # and api_server.py already ignore the middle "title" slot when we
        # pass None).
        return output_file, None, video_metadata