# clip_editor/intelligent_trimmer.py
"""
Intelligent Clip Trimming.

This module is the editorial brain of the pipeline. Since title generation
was moved to a separate tool, the trim prompt now carries full
responsibility for identifying what makes the clip work — so we spend the
compute on Gemini 3.1 Pro with `thinking_level=high` and lean hard on the
mic + game dialogue transcriptions.

Strategy:
  1. Find the punch point (the single moment of maximum impact).
  2. Build the minimum viable setup that makes the punch land.
  3. Cut aggressively everywhere else.

Gemini 3 responds best to direct, concise prompts — we don't over-engineer
chain-of-thought scaffolding; we let `thinking_level=high` do that work.
"""

import os
import re
import json
import time
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional

from google.genai import types

from utils.models import (
    MODEL_PRO,
    THINKING_TRIM,
    get_gemini_client,
    get_safety_settings,
)
from video_utils import get_video_duration


class IntelligentTrimmer:
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.client = get_gemini_client()
        self.safety_settings = get_safety_settings()
        self.max_clip_duration = 55.0
        self.min_clip_duration = 12.0
        # Stash of the last successful parse so callers (the metadata
        # writer in particular) can read the punch_point_time and
        # description without us having to widen the analyze_for_trim
        # return signature.
        self.last_parsed_data: Optional[dict] = None
        self.log_func(
            f"✂️  Intelligent trimmer initialized: {MODEL_PRO} "
            f"(thinking={THINKING_TRIM})"
        )

    # ── Public API ────────────────────────────────────────────────────────
    def analyze_for_trim(
        self,
        video_path: str,
        mic_transcriptions: Optional[List[str]] = None,
        desktop_transcriptions: Optional[List[str]] = None,
    ) -> List[Tuple[float, float]]:
        try:
            self.log_func("\n" + "=" * 60)
            self.log_func("🎬 INTELLIGENT CLIP TRIMMING — ANALYSIS PHASE")
            self.log_func("=" * 60)
            video_duration = get_video_duration(video_path, self.log_func)
            return self._call_gemini_for_trim(
                video_path,
                video_duration,
                mic_transcriptions,
                desktop_transcriptions,
            )
        except Exception as e:
            self.log_func(f"❌ Trim analysis failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []

    def apply_trim(
        self,
        input_video: str,
        output_video: str,
        segments_to_keep: List[Tuple[float, float]],
    ) -> bool:
        return self._apply_trim(input_video, output_video, segments_to_keep)

    # ── Prompt helpers ────────────────────────────────────────────────────
    def _format_transcription(
        self,
        transcriptions: List[str],
        max_lines: int = 200,
    ) -> str:
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
        if len(parsed) > max_lines:
            half = max_lines // 2
            parsed = (
                parsed[:half]
                + [(-1, "... [middle omitted] ...")]
                + parsed[-half:]
            )
        out = []
        for ts, text in parsed:
            out.append(text if ts == -1 else f"[{ts:.1f}s] {text}")
        return "\n".join(out)

    # ── Gemini call ───────────────────────────────────────────────────────
    # Strategy:
    #  • Videos ≤ 18 MB  → inline bytes (no File API round-trip, no
    #    "cannot fetch content from URL" failure mode).
    #  • Videos > 18 MB  → File API upload + ACTIVE poll + URI reference.
    #
    # The 20 MB limit is the hard API request cap; 18 MB leaves room for
    # the prompt + JSON overhead.
    INLINE_MAX_BYTES = 18 * 1024 * 1024

    def _call_gemini_for_trim(
        self,
        video_path: str,
        video_duration: float,
        mic_transcriptions,
        desktop_transcriptions,
    ) -> List[Tuple[float, float]]:
        try:
            mime_type = self._mime_type_for(video_path)
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)

            prompt = self._build_trim_prompt(
                video_duration,
                mic_transcriptions,
                desktop_transcriptions,
            )

            if file_size <= self.INLINE_MAX_BYTES:
                self.log_func(
                    f"\n📤 Sending video inline to Gemini "
                    f"({file_size_mb:.1f} MB, {mime_type}) — "
                    f"inline path (no File API)"
                )
                return self._trim_via_inline(
                    video_path, mime_type, prompt, video_duration,
                )
            else:
                self.log_func(
                    f"\n📤 Uploading video to Gemini File API "
                    f"({file_size_mb:.1f} MB, {mime_type}) — "
                    f"too large for inline"
                )
                return self._trim_via_files_api(
                    video_path, mime_type, prompt, video_duration,
                )
        except Exception as e:
            self.log_func(f"❌ Gemini trim analysis failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []

    # ── Inline path — small files ──────────────────────────────────────────
    def _trim_via_inline(
        self,
        video_path: str,
        mime_type: str,
        prompt: str,
        video_duration: float,
    ) -> List[Tuple[float, float]]:
        """Send the video as raw bytes in the request body."""
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_part = types.Part.from_bytes(
            data=video_bytes,
            mime_type=mime_type,
        )

        self.log_func(
            f"   Analysing for trim decisions "
            f"(thinking={THINKING_TRIM}) …"
        )
        response = self.client.models.generate_content(
            model=MODEL_PRO,
            contents=[prompt, video_part],
            config=types.GenerateContentConfig(
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_TRIM,
                ),
            ),
        )
        if not response or not getattr(response, "text", None):
            self.log_func("❌ Gemini returned empty response")
            return []
        return self._parse_trim_response(response.text, video_duration)

    # ── File API path — large files ────────────────────────────────────────
    def _trim_via_files_api(
        self,
        video_path: str,
        mime_type: str,
        prompt: str,
        video_duration: float,
    ) -> List[Tuple[float, float]]:
        """Upload via Files API, poll to ACTIVE, reference by URI."""
        video_file = None
        try:
            video_file = self._upload_with_retry(video_path, mime_type)

            # Wait for ACTIVE — "Cannot fetch content" error == not ready yet.
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
                        f"File did not become ACTIVE within "
                        f"{max_wait_seconds}s (last state: {state})"
                    )
                self.log_func(
                    f"   Waiting for ACTIVE (state: {state}) …")
                time.sleep(3)
                video_file = self.client.files.get(name=video_file.name)
            self.log_func(f"   ✅ File is ACTIVE: {video_file.name}")

            file_part = types.Part.from_uri(
                file_uri=video_file.uri,
                mime_type=video_file.mime_type or mime_type,
            )

            self.log_func(
                f"   Analysing for trim decisions "
                f"(thinking={THINKING_TRIM}) …"
            )
            response = self.client.models.generate_content(
                model=MODEL_PRO,
                contents=[prompt, file_part],
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_TRIM,
                    ),
                ),
            )
            if not response or not getattr(response, "text", None):
                self.log_func("❌ Gemini returned empty response")
                return []
            return self._parse_trim_response(response.text, video_duration)
        finally:
            if video_file is not None:
                try:
                    self.client.files.delete(name=video_file.name)
                except Exception:
                    pass

    def _upload_with_retry(self, video_path: str, mime_type: str):
        """
        files.upload retried on transient network failures.

        The genai SDK wraps requests in tenacity but its default retry policy
        does not include httpx.ConnectTimeout / ReadTimeout / ConnectError —
        which are exactly the failure modes seen on flaky uplinks for large
        videos. We retry with exponential backoff up to 4 attempts.
        """
        import httpx
        transient = (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            ConnectionError,
            TimeoutError,
        )
        attempts = 4
        delay = 5
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log_func(
                        f"   ↻ Upload retry {attempt}/{attempts} …"
                    )
                return self.client.files.upload(
                    file=video_path,
                    config={"mime_type": mime_type},
                )
            except transient as e:
                last_exc = e
                self.log_func(
                    f"   ⚠️  Upload attempt {attempt}/{attempts} failed: "
                    f"{type(e).__name__}: {e}"
                )
                if attempt == attempts:
                    break
                self.log_func(f"   sleeping {delay}s before retry …")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise last_exc if last_exc else RuntimeError("upload failed")

    @staticmethod
    def _mime_type_for(path: str) -> str:
        """Return the Gemini-friendly MIME type for common video extensions."""
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

    def _build_trim_prompt(
        self,
        video_duration: float,
        mic_transcriptions,
        desktop_transcriptions,
    ) -> str:
        mic_dialogue = self._format_transcription(mic_transcriptions or [])
        game_dialogue = self._format_transcription(desktop_transcriptions or [])

        if mic_transcriptions:
            self.log_func(
                f"   Added {len(mic_transcriptions)} mic lines to prompt")
        if desktop_transcriptions:
            self.log_func(
                f"   Added {len(desktop_transcriptions)} game lines to prompt")

        # Gemini 3 prompt-engineering best practice:
        #  - Put the data context first
        #  - Put the instructions at the end ("Based on the information above…")
        #  - Be concise; trust the model's reasoning
        return f"""=== VIDEO METADATA ===
Total duration: {video_duration:.1f}s
Target output: 12–45s

=== DIALOGUE TRANSCRIPTS ===
PLAYER VOICE (microphone — the streamer's commentary):
{mic_dialogue}

GAME AUDIO (NPCs, events, music — may contain noise):
{game_dialogue}

=== TASK ===
You are a viral YouTube Shorts editor. Based on the video and dialogue above,
produce a trim plan that makes this clip hit HARDER and feel FASTER.

Work through these steps in order. Do not output the steps; only the final
JSON.

STEP 1 — FIND THE PUNCH POINT
The punch point is the single moment of maximum impact: the jumpscare, the
kill, the laugh, the fail, the reveal, the one-liner that lands. It is almost
always in the last third of the video. Cite its exact timestamp. The punch
point must be fully preserved in the output.

STEP 2 — BUILD THE MINIMUM VIABLE SETUP
Ask: "What is the LEAST amount of footage a first-time viewer needs so that
the punch point lands?" Dialogue-driven setups (the ironic statement right
before the fail, the confident brag right before the death) are pure gold
and must be kept. Pure visual setup should be 5–15 seconds; longer only if
there is extraordinary justification.

STEP 3 — CUT AGGRESSIVELY
Hard jump cuts between kept segments are fine and preferred for pacing.
Delete:
  - Any pause > 1.5s that isn't tension-building
  - Trailing commentary after the punchline lands (end within ~3s of peak)
  - Repetitive phrases and filler
  - Multiple setup attempts — keep only the best one
  - Dead-air exploration / walking / menus unless they serve the setup
  - Pre-roll and post-roll chatter

STEP 4 — VERIFY BEFORE OUTPUTTING
  - Does every kept second directly serve the punch or the minimum setup?
  - Is the punch point fully intact (not truncated)?
  - Would a stranger watching this clip with no prior context understand it?
  - If pacing still feels slow, cut more.

=== OUTPUT ===
Return ONLY this JSON, no other text, no code fences:

{{
  "punch_point_time": 28.5,
  "punch_point_description": "Short description of the peak moment",
  "setup_rationale": "Why the setup segments were chosen",
  "analysis": "One-line summary of what was removed and why",
  "segments_to_keep": [
    {{"start": 22.0, "end": 29.5, "reason": "Setup + punch + immediate reaction"}},
    {{"start": 31.0, "end": 37.0, "reason": "Follow-up line that lands"}}
  ],
  "estimated_duration": 13.5,
  "cuts_made": [
    "0–22s: irrelevant exploration",
    "29.5–31s: dead air pause",
    "37s–end: rambling wind-down"
  ]
}}

Based on the video and dialogue above, produce that JSON now.
"""

    # ── Response parsing ──────────────────────────────────────────────────
    def _parse_trim_response(
        self,
        response_text: str,
        video_duration: float,
    ) -> List[Tuple[float, float]]:
        try:
            self.log_func("\n📋 Parsing trim decisions …")
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                self.log_func("⚠️  No JSON found in response")
                return []
            data = json.loads(json_match.group(0))
            # Stash for the metadata writer — gives the strategist access to
            # punch_point_time + setup_rationale without changing the return type.
            self.last_parsed_data = data

            punch = data.get("punch_point_time")
            punch_desc = data.get("punch_point_description", "")
            if punch is not None:
                self.log_func(
                    f"\n   🥊 Punch point: {punch:.1f}s — {punch_desc}")

            setup = data.get("setup_rationale", "")
            if setup:
                self.log_func(f"   📐 Setup rationale: {setup}")

            analysis = data.get("analysis", "")
            if analysis:
                self.log_func(f"   ✂️  Analysis: {analysis}")

            segments_data = data.get("segments_to_keep", [])
            if not segments_data:
                self.log_func("⚠️  No segments returned")
                return []

            segments: List[Tuple[float, float]] = []
            for i, seg in enumerate(segments_data):
                start = max(0.0, float(seg["start"]))
                end = min(video_duration, float(seg["end"]))
                if start >= end:
                    continue
                reason = seg.get("reason", "")

                # Add end-buffer to last segment only
                if i == len(segments_data) - 1:
                    buffered_end = min(end + 1.0, video_duration)
                    if buffered_end > end:
                        self.log_func(
                            f"   📏 End-buffer: {end:.1f}s → "
                            f"{buffered_end:.1f}s"
                        )
                        end = buffered_end

                segments.append((start, end))
                self.log_func(
                    f"   ✓ Keep {start:.1f}s–{end:.1f}s "
                    f"({end - start:.1f}s) — {reason}"
                )

            cuts = data.get("cuts_made", [])
            if cuts:
                self.log_func("\n   Cuts made:")
                for c in cuts:
                    self.log_func(f"   ✂️  {c}")

            est = data.get(
                "estimated_duration",
                sum(e - s for s, e in segments),
            )
            self.log_func(
                f"\n   ⏱  Estimated final duration: {est:.1f}s")
            return segments

        except Exception as e:
            self.log_func(f"❌ Error parsing response: {e}")
            self.log_func(f"   Raw: {response_text[:500]}")
            return []

    # ── FFmpeg execution (unchanged) ──────────────────────────────────────
    def _apply_trim(
        self,
        input_video: str,
        output_video: str,
        segments_to_keep: List[Tuple[float, float]],
    ) -> bool:
        try:
            self.log_func("\n✂️  Applying trim …")
            temp_dir = tempfile.gettempdir()
            segment_files = []

            for i, (start, end) in enumerate(segments_to_keep):
                seg_file = os.path.join(
                    temp_dir, f"trim_segment_{i:03d}.mp4")
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", input_video,
                    "-to", str(end - start),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    seg_file,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if (
                    result.returncode == 0
                    and os.path.exists(seg_file)
                    and os.path.getsize(seg_file) > 1000
                ):
                    segment_files.append(seg_file)
                    self.log_func(
                        f"   ✓ Segment {i + 1}/{len(segments_to_keep)}"
                        f" ({end - start:.1f}s)"
                    )
                else:
                    self.log_func(
                        f"   ✗ Segment {i + 1} extraction failed")

            if not segment_files:
                self.log_func("❌ No segments extracted")
                return False

            if len(segment_files) == 1:
                shutil.copy2(segment_files[0], output_video)
                os.remove(segment_files[0])
                return True

            concat_list = os.path.join(
                temp_dir, "trim_concat_list.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for sf in segment_files:
                    sf_abs = os.path.abspath(sf).replace("\\", "/")
                    f.write(f"file '{sf_abs}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                "-movflags", "+faststart",
                output_video,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            for sf in segment_files:
                try:
                    os.remove(sf)
                except Exception:
                    pass
            try:
                os.remove(concat_list)
            except Exception:
                pass

            if result.returncode == 0 and os.path.exists(output_video):
                final_dur = get_video_duration(output_video, self.log_func)
                expected = sum(e - s for s, e in segments_to_keep)
                if final_dur < 1.0 or final_dur < expected * 0.4:
                    self.log_func(
                        f"❌ Output suspiciously short: {final_dur:.1f}s")
                    return False
                self.log_func("   ✅ Trim applied successfully")
                return True

            self.log_func(
                f"❌ Concat failed: "
                f"{result.stderr[-500:] if result.stderr else ''}"
            )
            return False

        except Exception as e:
            self.log_func(f"❌ Trim failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return False