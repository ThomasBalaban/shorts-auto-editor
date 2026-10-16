"""
Punch freeze — editing primitive (experiment arm, default OFF).

A dramatic frame-hold AT the punchline: the picture freezes for
``fx_punch_freeze_s`` while the audio keeps playing, then motion
resumes. Crucially this is DURATION-NEUTRAL — no time is inserted, so
subtitle timings, onomatopoeia events and zoom timelines downstream are
untouched (a time-inserting freeze would desync all of them).

Applied to the edited video after the zoom pass, before subtitle embed.
The punch time must already be mapped to OUTPUT time. Failure never
sinks the render: callers treat False as "no freeze".
"""

import os
import subprocess
from typing import List, Optional, Tuple

DEFAULT_FREEZE_S = 0.5


def map_source_time_to_output(
    t: float, kept_segments: List[Tuple[float, float]],
) -> Optional[float]:
    """Map a source-time instant into output time through the kept
    segments. None when t falls in a cut region."""
    elapsed = 0.0
    for s, e in kept_segments:
        if s <= t <= e:
            return elapsed + (t - s)
        elapsed += e - s
    return None


def apply_punch_freeze(video_path: str, punch_out_s: float,
                       freeze_s: float, log_func) -> bool:
    """Freeze the displayed frame at ``punch_out_s`` for ``freeze_s``
    without changing the video's duration. In-place on success."""
    tmp_out = None
    try:
        freeze_s = max(0.2, min(1.5, float(freeze_s or DEFAULT_FREEZE_S)))
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True)
        duration = float(probe.stdout.strip())
        if punch_out_s <= 0.3 or punch_out_s >= duration - 0.3:
            log_func("   🧊 Punch freeze skipped — punch too close to an edge")
            return False
        end = min(duration, punch_out_s + freeze_s)

        base, ext = os.path.splitext(video_path)
        tmp_out = f"{base}_freeze{ext}"
        t = punch_out_s
        graph = (
            f"[0:v]split[main][fr];"
            f"[fr]trim=start={t:.3f}:end={t + 0.1:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={freeze_s + 0.2:.3f},"
            f"setpts=PTS+{t:.3f}/TB[frz];"
            f"[main][frz]overlay=enable='between(t,{t:.3f},{end:.3f})'"
            f":eof_action=pass[vout]"
        )
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-filter_complex", graph,
            "-map", "[vout]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            tmp_out,
        ]
        log_func(f"   🧊 Punch freeze: holding frame at {t:.2f}s "
                 f"for {freeze_s:.2f}s (audio continues)")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(tmp_out) \
                or os.path.getsize(tmp_out) < 1000:
            log_func(f"   ⚠ Punch freeze render failed — skipping: "
                     f"{r.stderr[-300:] if r.stderr else '?'}")
            return False
        # Duration guard — the freeze must not change length.
        probe2 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", tmp_out],
            capture_output=True, text=True)
        got = float(probe2.stdout.strip())
        if abs(got - duration) > 0.5:
            log_func(f"   ⚠ Punch freeze changed duration ({duration:.2f}s "
                     f"→ {got:.2f}s) — discarded")
            return False
        os.replace(tmp_out, video_path)
        tmp_out = None
        log_func("   ✅ Punch freeze applied")
        return True
    except Exception as e:
        log_func(f"   ⚠ Punch freeze error — skipping: {e}")
        return False
    finally:
        if tmp_out and os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
