"""
Static end-card: freeze the final frame for ~2 seconds and overlay one
short punchline-reinforcing text line.

Experiment primitive (capability-gaps artifact, 2026-10: end_card_text,
3x lift on n=3 — unvalidated). Default OFF; the operator toggles it per
batch and gates wider rollout on final-second swipe-away rates.

Failure here must never sink a render: every caller treats a False
return as "no card" and the already-rendered short ships untouched.
"""

import os
import re
import subprocess
import tempfile
from typing import List, Optional

END_CARD_SECONDS = 2.0
MAX_CARD_CHARS = 48

_TIMESTAMP_PREFIX = re.compile(r"^\s*\d+\.\d+-\d+\.\d+:\s*")


def pick_end_card_text(mic_transcriptions: Optional[List[str]]) -> Optional[str]:
    """The last spoken mic line — i.e. the punchline itself.

    Returns None when there is no usable dialogue or the closing line is
    too long to read in two seconds (better no card than a wall of text).
    """
    for line in reversed(mic_transcriptions or []):
        text = _TIMESTAMP_PREFIX.sub("", line).strip()
        if not text:
            continue
        if len(text) > MAX_CARD_CHARS:
            return None
        return text
    return None


def append_end_card(video_path: str, text: str, log_func,
                    seconds: float = None) -> bool:
    """Extend ``video_path`` in place: last frame frozen END_CARD_SECONDS
    with ``text`` centered over a dimmed band, audio padded with silence.

    Returns True on success; on any failure the original file is left
    exactly as it was.
    """
    card_s = float(seconds) if seconds else END_CARD_SECONDS
    tmp_out = None
    text_file = None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip())

        # textfile= sidesteps drawtext's escaping rules for quotes/colons.
        fd, text_file = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)

        base, ext = os.path.splitext(video_path)
        tmp_out = f"{base}_endcard{ext}"

        vf = (
            f"tpad=stop_mode=clone:stop_duration={card_s},"
            f"drawbox=y=ih*0.42:h=ih*0.16:color=black@0.55:t=fill"
            f":enable='gte(t,{duration:.3f})',"
            f"drawtext=textfile='{text_file}':fontsize=h*0.045"
            f":fontcolor=white:borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
            f":enable='gte(t,{duration:.3f})'"
        )
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", vf,
            "-af", f"apad=pad_dur={card_s}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac",
            tmp_out,
        ]
        log_func(f"   🃏 Appending {card_s:.1f}s end card: “{text}”")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(tmp_out) \
                or os.path.getsize(tmp_out) == 0:
            log_func("   ⚠ End card render failed — shipping without card. "
                     f"ffmpeg: {result.stderr[-300:] if result.stderr else '?'}")
            return False
        os.replace(tmp_out, video_path)
        tmp_out = None
        log_func("   ✅ End card appended")
        return True
    except Exception as e:
        log_func(f"   ⚠ End card error — shipping without card: {e}")
        return False
    finally:
        for p in (tmp_out, text_file):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
