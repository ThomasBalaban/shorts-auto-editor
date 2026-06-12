# gemini_vision_analyzer.py
"""
Per-event video analysis used by the onomatopoeia pipeline.

Migrated to the `google-genai` SDK and Gemini 3 Flash. We keep thinking at
`minimal` and media resolution at `low` because this is a high-volume
captioning task where latency and cost matter more than depth of reasoning.
"""

from typing import List, Dict, Optional
import json

import cv2  # type: ignore
from PIL import Image

from google.genai import types

from utils.models import (
    MODEL_FLASH,
    THINKING_VISION,
    MEDIA_RES_VISION,
    get_gemini_client_alpha,
    get_safety_settings,
)


class GeminiVisionAnalyzer:
    """Video scene analysis via Gemini 3 Flash."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.client = get_gemini_client_alpha()
        self.safety_settings = get_safety_settings()
        self.num_frames_for_caption = 4
        self.log_func(f"🎬 Gemini Vision analyzer initialized: {MODEL_FLASH}")

    # ── Convenience — kept for the title-less pipeline's back-compat usage ──
    #  modules still want raw frames.)
    def extract_frames_from_video(
        self,
        video_path: str,
        start_time: float,
        duration: float,
    ) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_func(f"ERROR: Could not open video file: {video_path}")
            return []

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[Image.Image] = []
        step = duration / max(self.num_frames_for_caption, 1)

        for i in range(self.num_frames_for_caption):
            frame_time = start_time + (i * step)
            frame_pos = int(frame_time * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    # ── Core captioning call ───────────────────────────────────────────────
    def generate_analysis(self, frames: List[Image.Image]) -> Dict:
        """Caption + scene context + dramatic intensity for a batch of frames."""
        analysis: Dict = {
            "caption": "", "scene_context": set(), "intensity": "normal"}
        if not frames:
            return analysis

        prompt = (
            "You are a video game scene analyzer. Analyze these sequential "
            "frames from a gameplay clip. Ignore any UI elements or facecam "
            "overlays.\n\n"
            "Respond with a JSON object containing exactly three keys:\n"
            '1. "caption": a concise one-sentence description of the primary '
            "gameplay action.\n"
            '2. "scene_context": a list of relevant environmental keywords. '
            'If the scene is underwater, include the word "underwater". '
            "Otherwise the list may be empty.\n"
            '3. "intensity": classify the dramatic weight of this moment as '
            "exactly one of:\n"
            "   - \"epic\": a big, exciting gameplay beat — a kill, boss, "
            "explosion, narrow escape, reveal, major damage, intense action.\n"
            "   - \"suspenseful\": tense or eerie dread — exploring the "
            "unknown, a horror/threat building, creeping through danger, an "
            "ominous quiet before something happens.\n"
            "   - \"normal\": ordinary gameplay, menus, idle, or routine "
            "traversal with no dramatic charge.\n\n"
            'Example: {"caption": "The player shoots a rifle at an enemy.", '
            '"scene_context": [], "intensity": "epic"}\n'
            'Example: {"caption": "The player creeps down a dark hallway '
            'toward a flickering light.", "scene_context": [], '
            '"intensity": "suspenseful"}\n'
            'Example: {"caption": "The player walks through an empty field.", '
            '"scene_context": [], "intensity": "normal"}\n\n'
            "Here are the frames:"
        )

        # Build the multipart content. Each PIL image becomes a Part.
        contents: List = [prompt]
        contents.extend(frames)

        try:
            response = self.client.models.generate_content(
                model=MODEL_FLASH,
                contents=contents,
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_VISION,
                    ),
                    media_resolution=MEDIA_RES_VISION,
                    response_mime_type="application/json",
                ),
            )

            if not response or not getattr(response, "text", None):
                self.log_func("💥 Gemini Vision returned empty response")
                return analysis

            raw_text = response.text.strip().replace(
                "```json", "").replace("```", "")
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                parsed = next((p for p in parsed if isinstance(p, dict)), {})
            if not isinstance(parsed, dict):
                parsed = {}

            analysis["caption"] = parsed.get("caption", "")
            analysis["scene_context"] = set(parsed.get("scene_context", []))
            intensity = str(parsed.get("intensity", "normal")).strip().lower()
            analysis["intensity"] = (
                intensity if intensity in ("epic", "suspenseful", "normal")
                else "normal")

            self.log_func(
                f"👁️ Vision caption: '{analysis['caption']}' "
                f"[{analysis['intensity']}]")
            if analysis["scene_context"]:
                self.log_func(
                    f"💧 Scene context: {analysis['scene_context']}")
            return analysis

        except json.JSONDecodeError as e:
            self.log_func(f"💥 Gemini Vision JSON parse error: {e}")
            if response and getattr(response, "text", None):
                self.log_func(f"   Raw response was: {response.text}")
                analysis["caption"] = response.text.strip()
            return analysis
        except Exception as e:
            self.log_func(f"💥 Gemini Vision analysis error: {e}")
            return analysis

    # ── Event-map orchestration ────────────────────────────────────────────
    def analyze_video_at_timestamps(
        self,
        video_path: str,
        events: List[Dict],
        window_duration: float = 5.0,
    ) -> Dict[float, Dict]:
        """
        Run vision analysis once per event and return a
        {event_time: analysis} map keyed by each event's original time.
        """
        analysis_map: Dict[float, Dict] = {}
        for event in events:
            event_time = event["time"]
            start_time = max(0.0, event_time - 2.0)

            frames = self.extract_frames_from_video(
                video_path, start_time, window_duration)
            if not frames:
                continue

            result = self.generate_analysis(frames)
            caption = result["caption"]
            scene_context = result["scene_context"]
            if not caption:
                continue

            analysis_map[event_time] = {
                "video_caption": caption,
                "scene_context": scene_context,
                "intensity": result.get("intensity", "normal"),
                "confidence": 0.95,
            }
        return analysis_map

    # ── Camera-region gaze check ───────────────────────────────────────────
    @staticmethod
    def _crop_region(img: Image.Image, region: Dict) -> Image.Image:
        """Crop a PIL image to a normalized {x,y,w,h} region of the frame."""
        w, h = img.size
        x0 = int(max(0.0, region.get("x", 0.0)) * w)
        y0 = int(max(0.0, region.get("y", 0.0)) * h)
        x1 = int(min(1.0, region.get("x", 0.0) + region.get("w", 1.0)) * w)
        y1 = int(min(1.0, region.get("y", 0.0) + region.get("h", 1.0)) * h)
        if x1 <= x0 or y1 <= y0:
            return img
        return img.crop((x0, y0, x1, y1))

    def analyze_camera_gaze(
        self,
        video_path: str,
        timestamp: float,
        region: Dict,
        camera_mode: str = "vtuber",
    ) -> Optional[float]:
        """Judge whether the avatar in the camera region looks engaged at a
        moment, returning a quality in [0,1]:

            1.0  — face visible and looking toward the viewer (good to zoom)
            0.25 — face visible but looking away/down (likely model bug)
            0.1  — face not clearly visible / glitched / obscured

        Returns None on any failure so the caller treats it as "unknown" and
        applies no penalty (don't punish good reactions on a flaky call).
        """
        try:
            frames = self.extract_frames_from_video(
                video_path, max(0.0, timestamp - 0.3), 0.6)
            if not frames:
                return None
            cropped = [self._crop_region(f, region) for f in frames][:2]
            if not cropped:
                return None

            subject = (
                "the streamer's real face on a webcam" if camera_mode == "facecam"
                else "the VTuber/streamer avatar")
            prompt = (
                f"This is a cropped view of {subject} from a webcam region. "
                "Judge the face only.\n"
                "Respond with a JSON object with exactly two boolean keys:\n"
                '1. "face_visible": is the character\'s face clearly visible '
                "and not glitched, cut off, or obscured?\n"
                '2. "looking_forward": are the eyes open and oriented toward '
                "the viewer/camera (engaged), as opposed to looking down, away, "
                "or eyes closed/rolled?\n\n"
                'Example: {"face_visible": true, "looking_forward": true}'
            )
            contents: List = [prompt]
            contents.extend(cropped)

            response = self.client.models.generate_content(
                model=MODEL_FLASH,
                contents=contents,
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_VISION,
                    ),
                    media_resolution=MEDIA_RES_VISION,
                    response_mime_type="application/json",
                ),
            )
            if not response or not getattr(response, "text", None):
                return None
            raw = response.text.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(raw)
            # Gemini sometimes wraps the object in a list — unwrap to the first
            # dict so a list response doesn't crash the check. A genuinely
            # unreadable shape returns None (unknown → no penalty), so a garbage
            # call never suppresses a good reaction.
            if isinstance(parsed, list):
                parsed = next(
                    (p for p in parsed if isinstance(p, dict)), None)
            if not isinstance(parsed, dict):
                return None
            face_visible = bool(parsed.get("face_visible", False))
            looking_forward = bool(parsed.get("looking_forward", False))

            if not face_visible:
                quality = 0.1
            elif looking_forward:
                quality = 1.0
            else:
                quality = 0.25
            self.log_func(
                f"   👁 camera gaze @ {timestamp:.2f}s: "
                f"visible={face_visible} forward={looking_forward} "
                f"→ q={quality:.2f}")
            return quality
        except Exception as e:
            self.log_func(f"   ⚠ camera gaze check failed: {e}")
            return None