# ai_director/video_editor.py
"""
VideoEditor — segment-based parametric zoom pipeline.

Changes vs original:
  • zoom_to_cam:  1.50 → 1.65  (more dramatic punch-in), faster ramp
  • zoom_to_game: 1.30 → 1.55  (wider zoom-in on action), faster ramp
  • zoom_out:     blur + scale effect, faster pull-back
  • All zooms now reach their target in ~60 % of the segment duration
    then hold, which reads as snappy rather than drifting.
  • zoom_to_cam_reaction duration: 1.5 s (was 2 s) for punchier cuts.
  • zoom_to_game duration: 2.5 s (was 3 s).
"""

import subprocess
import os
import shutil
from typing import List
from ai_director.data_models import TimelineEvent


class VideoEditor:
    # Default focus regions (normalized 0–1 of the 1080×1920 frame).
    # These reproduce the legacy behavior when no layout preset is configured:
    #   • camera   → top-left corner   (old facecam position)
    #   • gameplay → centre of frame
    DEFAULT_CAMERA_REGION = {"x": 0.0, "y": 0.0, "w": 0.30, "h": 0.30}
    DEFAULT_GAMEPLAY_REGION = {"x": 0.25, "y": 0.25, "w": 0.50, "h": 0.50}

    def __init__(self, log_func=None, camera_region=None, gameplay_region=None):
        self.log_func = log_func or print
        self.camera_region = self._normalize_region(
            camera_region, self.DEFAULT_CAMERA_REGION)
        self.gameplay_region = self._normalize_region(
            gameplay_region, self.DEFAULT_GAMEPLAY_REGION)
        cam_c = self._center(self.camera_region)
        game_c = self._center(self.gameplay_region)
        self.log_func(
            "📹 AI Director Video Editor initialised (snappy zooms). "
            f"cam_center=({cam_c[0]:.2f},{cam_c[1]:.2f}) "
            f"game_center=({game_c[0]:.2f},{game_c[1]:.2f})"
        )

    # ── Region helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_region(region, fallback):
        """Validate a normalized {x,y,w,h} region, falling back to a default
        when missing or malformed. All values are clamped to [0, 1]."""
        if not isinstance(region, dict):
            return dict(fallback)
        try:
            r = {k: float(region[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            return dict(fallback)
        for k in ("x", "y", "w", "h"):
            r[k] = max(0.0, min(1.0, r[k]))
        if r["w"] <= 0 or r["h"] <= 0:
            return dict(fallback)
        return r

    @staticmethod
    def _center(region):
        """Return the (cx, cy) center of a normalized region."""
        return (region["x"] + region["w"] / 2.0,
                region["y"] + region["h"] / 2.0)

    @staticmethod
    def _focus_exprs(center):
        """Build zoompan x/y expressions that center the crop window
        (size iw/zoom × ih/zoom) on the given normalized (cx, cy), clamped
        in-bounds."""
        cx, cy = center
        x_expr = f"max(0,min(iw-iw/zoom,{cx:.4f}*iw-(iw/zoom)/2))"
        y_expr = f"max(0,min(ih-ih/zoom,{cy:.4f}*ih-(ih/zoom)/2))"
        return x_expr, y_expr

    # ── Public entry ──────────────────────────────────────────────────────────

    def apply_edits(
        self,
        input_video: str,
        output_video: str,
        timeline: List[TimelineEvent],
    ):
        applicable = [
            e for e in timeline
            if e.action in (
                "zoom_to_cam", "zoom_to_game",
                "zoom_out", "zoom_to_cam_reaction",
            )
        ]
        if not applicable:
            self.log_func("No applicable edits — copying original.")
            shutil.copy(input_video, output_video)
            return
        self.log_func(
            f"Applying {len(applicable)} edits …")
        self._apply_edits_segment_based(input_video, output_video, applicable)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _apply_edits_segment_based(
        self, input_video: str, output_video: str,
        events: List[TimelineEvent],
    ):
        import tempfile
        total_duration = self._get_video_duration(input_video)
        segments = self._create_segments(events, total_duration)
        temp_dir = tempfile.mkdtemp()
        segment_files: List[str] = []
        try:
            for i, seg in enumerate(segments):
                out = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                if seg["has_effect"]:
                    self._process_segment_with_effect(
                        input_video, out, seg)
                else:
                    self._extract_plain_segment(input_video, out, seg)
                segment_files.append(out)
            self._concatenate_segments(segment_files, output_video)
        finally:
            for f in segment_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

    def _create_segments(
        self, events: List[TimelineEvent], total_duration: float,
    ) -> List[dict]:
        segments = []
        t = 0.0
        for e in sorted(events, key=lambda x: x.timestamp):
            if t < e.timestamp:
                segments.append(
                    {"start": t, "end": e.timestamp,
                     "has_effect": False, "event": None})
            segments.append(
                {"start": e.timestamp,
                 "end": e.timestamp + e.duration,
                 "has_effect": True, "event": e})
            t = e.timestamp + e.duration
        if t < total_duration:
            segments.append(
                {"start": t, "end": total_duration,
                 "has_effect": False, "event": None})
        return segments

    def _process_segment_with_effect(
        self, input_video: str, output_file: str, segment: dict,
    ):
        event = segment["event"]
        start = segment["start"]
        dur = segment["end"] - segment["start"]

        if dur <= 0.0:
            open(output_file, "wb").close()
            return

        if event.action == "zoom_out":
            filter_str = self._zoom_out_filter(dur)
        elif event.action in ("zoom_to_cam", "zoom_to_cam_reaction"):
            filter_str = self._cam_zoom_filter(dur)
        elif event.action == "zoom_to_game":
            filter_str = self._game_zoom_filter(dur)
        else:
            self._extract_plain_segment(input_video, output_file, segment)
            return

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.6f}",
            "-t",  f"{dur:.6f}",
            "-i", input_video,
            "-filter_complex", filter_str,
            "-map", "[output]",
            "-map", "0:a?", "-c:a", "copy",
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-r", "60",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_file,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           check=True, timeout=600)
            self.log_func(
                f"✅ {event.action} @ {start:.2f}s for {dur:.2f}s")
        except (subprocess.TimeoutExpired,
                subprocess.CalledProcessError) as e:
            stderr = getattr(e, "stderr", "")
            if stderr:
                self.log_func(f"❌ FFmpeg error: {stderr[-400:]}")
            self._extract_plain_segment(input_video, output_file, segment)

    # ── Zoom filters ──────────────────────────────────────────────────────────
    # Strategy: zoom ramps up over the first 60 % of frames, then holds.
    # This gives a punchy "snap" rather than a slow drift.

    def _cam_zoom_filter(self, duration: float) -> str:
        """
        Punch-in on the configured camera region.
        1.50 → 1.65  (was 1.50 → 1.575)
        Ramp completes in 60 % of segment duration.
        Focus follows ``self.camera_region`` (legacy default: top-left).
        """
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        ramp_frames = max(1, int(duration * 60 * 0.60))
        step = 0.15 / ramp_frames  # 1.50 → 1.65 over ramp_frames

        # After ramp, hold at 1.65
        z_expr = (
            f"if(lt(on,{ramp_frames}),"
            f"min(1.65,if(eq(on,0),1.50,pzoom+{step:.6f})),"
            f"1.65)"
        )
        x_expr, y_expr = self._focus_exprs(self._center(self.camera_region))
        return (
            "[0:v]setpts=PTS-STARTPTS,"
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            ":d=1:s=1080x1920:fps=60,"
            "format=yuv420p[output]"
        )

    def _game_zoom_filter(self, duration: float) -> str:
        """
        Punch-in on the configured gameplay region.
        1.30 → 1.55  (was 1.33 → 1.46)
        Ramp completes in 60 % of segment duration.
        Focus follows ``self.gameplay_region`` (legacy default: frame centre).
        """
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        ramp_frames = max(1, int(duration * 60 * 0.60))
        step = 0.25 / ramp_frames  # 1.30 → 1.55

        z_expr = (
            f"if(lt(on,{ramp_frames}),"
            f"min(1.55,if(eq(on,0),1.30,pzoom+{step:.6f})),"
            f"1.55)"
        )
        x_expr, y_expr = self._focus_exprs(self._center(self.gameplay_region))
        return (
            "[0:v]setpts=PTS-STARTPTS,"
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            ":d=1:s=1080x1920:fps=60,"
            "format=yuv420p[output]"
        )

    def _zoom_out_filter(self, duration: float) -> str:
        """
        Blur + faster pull-back.
        Scale 0.85 → 0.75  (was 0.80 → 0.75, now starts bigger for
        more visible snap-back).
        """
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        total_frames = max(1, int(duration * 60))
        # 0.85 → 0.75 over full duration  (Δ = 0.10)
        scale_expr = f"(0.85-(0.10*n/{total_frames}))"
        w_expr = f"2*trunc(iw*{scale_expr}/2)"
        h_expr = f"2*trunc(ih*{scale_expr}/2)"

        return (
            "[0:v]setpts=PTS-STARTPTS,scale=1080:1920,split[fg_in][bg_in];"
            "[bg_in]boxblur=40:5,eq=brightness=-0.15[bg];"
            f"[fg_in]scale=w='{w_expr}':h='{h_expr}':eval=frame[fg_scaled];"
            "[bg][fg_scaled]overlay=x='(W-w)/2':y='(H-h)/2',"
            "fps=60,format=yuv420p[output]"
        )

    # ── Plain segment / concat / utils ────────────────────────────────────────

    def _extract_plain_segment(
        self, input_video: str, output_file: str, segment: dict,
    ):
        start = segment["start"]
        dur = segment["end"] - segment["start"]
        if dur <= 0:
            open(output_file, "wb").close()
            return
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.6f}",
            "-t",  f"{dur:.6f}",
            "-i", input_video,
            "-filter_complex",
            "scale=1080:1920:flags=bilinear,fps=60,format=yuv420p[vout]",
            "-map", "[vout]",
            "-map", "0:a?", "-c:a", "copy",
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-r", "60",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_file,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           check=True, timeout=300)
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            shutil.copy2(input_video, output_file)

    def _concatenate_segments(
        self, segment_files: List[str], output_video: str,
    ):
        import tempfile
        concat = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False)
        try:
            for f in segment_files:
                concat.write(f"file '{f}'\n")
            concat.close()
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat.name,
                "-fflags", "+genpts",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_video,
            ]
            subprocess.run(cmd, capture_output=True, text=True,
                           check=True, timeout=900)
            self.log_func("✅ Segments concatenated")
        except subprocess.CalledProcessError as e:
            self.log_func(f"❌ Concat failed: {e.stderr[:400] if e.stderr else ''}")
            if segment_files:
                shutil.copy2(segment_files[0], output_video)
        finally:
            try:
                os.unlink(concat.name)
            except Exception:
                pass

    def _get_video_duration(self, video_path: str) -> float:
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 check=True, timeout=10)
            return float(res.stdout.strip())
        except Exception:
            return 60.0

    def _apply_simple_fallback_edits(
        self, input_video: str, output_video: str,
    ):
        try:
            shutil.copy2(input_video, output_video)
            self.log_func("📋 Fallback: copied original.")
        except Exception as e:
            self.log_func(f"❌ Fallback copy failed: {e}")