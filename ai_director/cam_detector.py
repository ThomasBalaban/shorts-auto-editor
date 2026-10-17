"""
Auto cam detection (opt-in via the auto_cam_detect setting).

One cheap Gemini Flash vision call locates the camera overlay (vtuber
avatar or facecam) and the primary gameplay area as normalized boxes,
replacing the manually dragged Layout-tab regions for that video.

Safety posture: the AI proposes, the deterministic zoompan math
consumes. A detection below CONFIDENCE_FLOOR, with an implausible box,
or any failure falls back to the operator's preset regions, and the
outcome (boxes, confidence, source) is stamped into metadata either way.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, Optional

from google.genai import types

from utils.models import (
    MODEL_FLASH,
    THINKING_VISION,
    MEDIA_RES_VISION,
    get_gemini_client,
    get_safety_settings,
)

CONFIDENCE_FLOOR = 0.6
MIN_BOX_AREA = 0.02   # a cam under 2% of frame area is a misdetection
MAX_CAM_AREA = 0.85   # ...and one covering ~everything probably is too

_PROMPT = """\
You are analyzing frames from a vertical gameplay recording that contains
a game feed and a streamer camera overlay (either a real facecam or an
animated VTuber avatar).

Across the frames shown, identify:
1. The streamer camera overlay region (facecam or VTuber avatar).
2. The primary gameplay region (the actual game content).

Coordinates are normalized to the frame: x,y = top-left corner, w,h =
size, all in 0..1.

Respond with ONLY this JSON:
{
  "cam_type": "vtuber" | "facecam" | "none",
  "camera_box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
  "gameplay_box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
  "confidence": 0.0,
  "reasoning": "one sentence"
}
Use cam_type "none" with confidence for your certainty if there is no
camera overlay."""


def _extract_frames(video_path: str, log_func):
    """3 PIL frames at 15/50/85% of the video."""
    from PIL import Image
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True)
    duration = float(probe.stdout.strip())
    frames = []
    for frac in (0.15, 0.5, 0.85):
        out = os.path.join(tempfile.gettempdir(),
                           f"camdet_{int(frac * 100)}.png")
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{duration * frac:.2f}",
             "-i", video_path, "-frames:v", "1",
             "-vf", "scale=540:-2", out],
            capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            frames.append(Image.open(out).copy())
            try:
                os.remove(out)
            except OSError:
                pass
    return frames


def _coord(v: float) -> float:
    """Repair the model's dropped-decimal numbers before clamping.

    Gemini occasionally emits "639" for "0.639" (same malformed-number
    family as its emoji escapes). Any coordinate > 1 in a normalized
    schema is that defect; /1000 restores three-digit mantissas, and
    larger values degrade to a clamp.
    """
    v = float(v)
    if 1.0 < v <= 1000.0:
        v = v / 1000.0
    return min(1.0, max(0.0, v))


def _valid_box(b) -> Optional[Dict[str, float]]:
    try:
        x = _coord(b["x"])
        y = _coord(b["y"])
        w = min(1.0 - x, _coord(b["w"]))
        h = min(1.0 - y, _coord(b["h"]))
    except (KeyError, TypeError, ValueError):
        return None
    if w * h < MIN_BOX_AREA:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def detect_regions(video_path: str, log_func) -> Optional[Dict]:
    """Detect camera + gameplay regions.

    Returns {"camera": box, "gameplay": box, "cam_type": str,
    "confidence": float} or None when detection should not be trusted
    (caller falls back to the preset).
    """
    try:
        frames = _extract_frames(video_path, log_func)
        if not frames:
            log_func("   📷 Cam detect: frame extraction failed")
            return None

        client = get_gemini_client()
        resp = client.models.generate_content(
            model=MODEL_FLASH,
            contents=[_PROMPT] + frames,
            config=types.GenerateContentConfig(
                safety_settings=get_safety_settings(),
                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_VISION),
                media_resolution=MEDIA_RES_VISION,
                response_mime_type="application/json",
            ),
        )
        raw = (resp.text or "").strip().replace("```json", "").replace("```", "")
        parsed = json.loads(raw)
        conf = float(parsed.get("confidence") or 0.0)
        cam_type = str(parsed.get("cam_type") or "none").lower()

        if cam_type == "none":
            log_func(f"   📷 Cam detect: no camera overlay found "
                     f"(confidence {conf:.2f}) — using preset")
            return None
        if conf < CONFIDENCE_FLOOR:
            log_func(f"   📷 Cam detect: confidence {conf:.2f} below floor "
                     f"{CONFIDENCE_FLOOR} — using preset")
            return None

        cam = _valid_box(parsed.get("camera_box"))
        game = _valid_box(parsed.get("gameplay_box"))
        if cam is None or game is None:
            log_func("   📷 Cam detect: implausible box — using preset")
            return None
        if cam["w"] * cam["h"] > MAX_CAM_AREA:
            log_func("   📷 Cam detect: camera box covers almost the whole "
                     "frame — using preset")
            return None

        log_func(
            f"   📷 Cam detected ({cam_type}, confidence {conf:.2f}): "
            f"cam x={cam['x']:.2f} y={cam['y']:.2f} "
            f"w={cam['w']:.2f} h={cam['h']:.2f} | "
            f"game y={game['y']:.2f} h={game['h']:.2f}")
        return {"camera": cam, "gameplay": game,
                "cam_type": cam_type, "confidence": conf,
                "reasoning": parsed.get("reasoning")}
    except Exception as e:
        log_func(f"   📷 Cam detect failed ({e}) — using preset")
        return None
