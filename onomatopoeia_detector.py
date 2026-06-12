# onomatopoeia_detector.py - UPDATED
import os
from typing import List, Dict, Tuple

from audio.onset_detector import GamingOnsetDetector
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
from processing.multimodal_fusion import MultimodalFusionEngine
from processing.gaming_optimizer import GamingOptimizer
from subtitle_generator import SubtitleGenerator
from utils.file_processor import FileProcessor


class OnomatopoeiaDetector:
    """Cross‑modal onomatopoeia detection system."""
    VERIFY_WINDOW_SEC: float = 1.0
    NMS_RADIUS_SEC: float = 0.4
    COOLDOWN_SEC: float = 0.5

    def __init__(self, sensitivity: float = 0.5, device: str = "cpu", log_func=None,
                 game_track_index: str = "a:2"):
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print
        self.game_track_index = game_track_index
        self.event_history: List[Dict[str, float]] = []
        self._initialize_components()
        self.log_func("✅ Cross‑modal multimodal system ready!")

    def _initialize_components(self) -> None:
        self.onset_detector = GamingOnsetDetector(sensitivity=self.sensitivity, log_func=self.log_func)
        self.video_analyzer = GeminiVisionAnalyzer(log_func=self.log_func)
        self.fusion_engine = MultimodalFusionEngine(log_func=self.log_func)
        self.gaming_optimizer = GamingOptimizer(log_func=self.log_func)
        self.subtitle_generator = SubtitleGenerator(log_func=self.log_func)
        self.file_processor = FileProcessor(log_func=self.log_func)

    def _analyze_video_file(self, video_path: str, animation_type: str, sync_offset: float = -0.15) -> Tuple[List[Dict], Dict[float, Dict]]:
        """ Analyzes the video with specific sync offset. """
        audio_path = None
        video_map: Dict[float, Dict] = {}
        try:
            self.log_func(f"\n{'='*60}\nSTARTING CROSS‑MODAL ANALYSIS\n{'='*60}")
            audio_path = self.file_processor.extract_audio_from_video(video_path, track_index=self.game_track_index)

            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            if not audio_events: return [], {}

            video_map = self._create_synchronized_video_analysis(video_path, audio_events)

            verified_events = self._verify_and_filter_events(audio_events, video_map)
            if not verified_events: return [], video_map

            # PASS SYNC OFFSET HERE
            final_effects = self.fusion_engine.process_multimodal_events(
                verified_events, video_map, animation_type, sync_offset=sync_offset
            )
            
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            return optimized_effects, video_map

        except Exception as e:
            self.log_func(f"ERROR during cross‑modal analysis: {e}")
            return [], video_map
        finally:
            if audio_path: self.file_processor.cleanup_temp_file(audio_path)

    def _verify_and_filter_events(self, events: List[Dict], video_map: Dict[float, Dict]) -> List[Dict]:
        verified: List[Dict] = []
        for event in sorted(events, key=lambda e: e['time']):
            matching_analysis = None
            for v_time, analysis in video_map.items():
                if abs(event['time'] - v_time) <= self.VERIFY_WINDOW_SEC:
                    matching_analysis = analysis
                    break
            if not matching_analysis: continue
            
            score = event.get('confidence', 0.5)
            if any(abs(event['time'] - ex['time']) < self.NMS_RADIUS_SEC and ex.get('score', 0) >= score for ex in verified): continue
            if any(0 < event['time'] - h['time'] < self.COOLDOWN_SEC for h in self.event_history): continue
            
            event['score'] = score
            verified.append(event)
            self.event_history.append({'time': event['time'], 'score': score})
            self.event_history = self.event_history[-50:]
        return verified

    def _create_synchronized_video_analysis(self, video_path: str, audio_events: List[Dict]) -> Dict[float, Dict]:
        groups = self._group_nearby_events(audio_events, 0.5)
        video_map: Dict[float, Dict] = {}
        for group in groups:
            primary_event = max(group, key=lambda e: e.get('energy', 0))
            t = primary_event['time']
            analysis = self.video_analyzer.analyze_video_at_timestamps(video_path, [primary_event])
            if analysis:
                for evt in group:
                    video_map[evt['time']] = analysis[t]
        return video_map

    def _group_nearby_events(self, events: List[Dict], max_group_span: float) -> List[List[Dict]]:
        if not events: return []
        sorted_events = sorted(events, key=lambda e: e['time'])
        groups: List[List[Dict]] = []
        current_group: List[Dict] = [sorted_events[0]]
        for ev in sorted_events[1:]:
            if ev['time'] - current_group[0]['time'] <= max_group_span:
                current_group.append(ev)
            else:
                groups.append(current_group)
                current_group = [ev]
        groups.append(current_group)
        return groups

    def create_subtitle_file(self, input_path: str, output_path: str, animation_type: str = "Random") -> Tuple[bool, List[Dict], Dict[float, Dict]]:
        """ Public API. NOTE: sync_offset must be set via analyze_file if used directly. """
        try:
            events, video_map = self.analyze_file(input_path, animation_type)
            if not events:
                self.log_func("No events detected for subtitle generation.")
                return False, [], {}
            success = self.subtitle_generator.create_subtitle_file(events, output_path, animation_type)
            return success, events, video_map
        except Exception as e:
            self.log_func(f"Error creating subtitle file: {e}")
            return False, [], {}

    def analyze_file(self, input_path: str, animation_type: str, sync_offset: float = -0.15) -> Tuple[List[Dict], Dict[float, Dict]]:
        """ Main analysis entry point. """
        file_type = self.file_processor.detect_file_type(input_path)
        if file_type == 'video':
            return self._analyze_video_file(input_path, animation_type, sync_offset)
        self.log_func(f"Unsupported file type for this analysis: {input_path}")
        return [], {}