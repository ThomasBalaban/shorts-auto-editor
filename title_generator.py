"""
Title generator for shorts-auto-editor.

Generates a single YouTube metadata title for a finished short, primed by
the channel's analyzer corpus (synthesis + per-video analysis). The output
is the title text plus a detailed analysis record explaining what reference
shorts and patterns shaped the choice.

Strict no-fallback policy (Phase 1.3 of gameplan.md):
    - Analyzer unreachable, synthesis missing, Gemini failure, or empty
      transcript → return None. Caller writes no title field at all.
"""
import datetime
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from analyzer_client import AnalyzerClient, CHANNEL_HANDLE
from utils.models import (
    MODEL_PRO,
    get_gemini_client,
    get_safety_settings,
)


# Shorts titles are competing for swipe-time on a tiny phone screen. Keep them
# tight — 70 chars is the YouTube short-title display ceiling on mobile.
MAX_TITLE_CHARS = 70

# Hard-deny list — chosen titles containing any of these (whole-word, case-
# insensitive) are rejected and the title is dropped. Note: "ass" alone is
# allowed; only the explicit anatomical/sexual terms below trip the gate.
# The prompt itself does the heavy lifting (creative euphemism); this is the
# safety net for when the model ignores instructions.
_BANNED_PATTERNS: List["re.Pattern[str]"] = [
    re.compile(r"\bfuck\w*\b", re.IGNORECASE),     # fuck, fucking, fucked, …
    re.compile(r"\bcum(s|ming|med)?\b", re.IGNORECASE),
    re.compile(r"\bdicks?\b", re.IGNORECASE),
    re.compile(r"\bassholes?\b", re.IGNORECASE),
    re.compile(r"\bpuss(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bclit\w*\b", re.IGNORECASE),     # clit, clits, clitoris
    re.compile(r"\bvagina\w*\b", re.IGNORECASE),   # vagina, vaginas, vaginal
]


def _banned_match(title: str) -> Optional[str]:
    """Return the first banned term matched in the title, or None."""
    for pat in _BANNED_PATTERNS:
        m = pat.search(title)
        if m:
            return m.group(0)
    return None

# JSON schema we ask Gemini to fill in. Captured here (not just inline in the
# prompt) so structured-output validation happens server-side.
_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "clip_interpretation",
        "chosen_title",
        "chosen_title_reasoning",
        "candidates_considered",
        "patterns_applied",
        "patterns_avoided",
        "reference_comparisons",
    ],
    "properties": {
        "clip_interpretation": {
            "type": "object",
            "description": (
                "Fill this in BEFORE proposing any title. The chosen "
                "title must be true to literal_event — a title that "
                "misrepresents the in-game event is rejected regardless "
                "of how clever it sounds."
            ),
            "required": [
                "literal_event",
                "streamer_reaction",
                "named_entities",
                "comedic_premise",
            ],
            "properties": {
                "literal_event": {
                    "type": "object",
                    "required": ["description", "evidence_quote"],
                    "description": (
                        "What is actually happening on screen. May be "
                        "coarse — 'streamer dies during a boss fight' "
                        "is acceptable when the precise cause is not "
                        "verifiable. NEVER name an entity, action, or "
                        "spoken line that you cannot quote verbatim "
                        "from the transcripts. NEVER mention 'chat', "
                        "'viewers', or 'stream chat' — there is no "
                        "chat data in this pipeline."
                    ),
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "One sentence describing what "
                                "happened, at the level of confidence "
                                "the evidence supports. If the "
                                "transcripts don't establish a "
                                "specific event, write 'unclear from "
                                "transcripts — streamer reacting to "
                                "[broad situational cue]'."
                            ),
                        },
                        "evidence_quote": {
                            "type": "string",
                            "description": (
                                "Verbatim line from mic or desktop "
                                "transcript that supports the "
                                "description, with source label "
                                "(e.g. 'mic: \"...\"' or "
                                "'desktop: \"...\"'). If no quote "
                                "supports a specific event, write "
                                "'no transcript evidence' and keep "
                                "the description coarse."
                            ),
                        },
                    },
                },
                "streamer_reaction": {
                    "type": "string",
                    "description": (
                        "Verbatim or near-verbatim of what the "
                        "streamer says, framed as social commentary "
                        "directed at friends in voice chat — NOT as "
                        "narration of the event. The streamer "
                        "addressing someone by name is talking TO "
                        "them. This is often the most reliable signal "
                        "in the clip and a strong title hook."
                    ),
                },
                "named_entities": {
                    "type": "array",
                    "description": (
                        "Every proper noun in the transcripts. RULES:"
                        " (1) Names appearing ONLY in desktop audio "
                        "are presumed Whisper mishearings of NPC "
                        "voice lines or game sound effects — mark "
                        "them 'likely_mishearing' and DO NOT use them "
                        "in the title. (2) Names in mic dialogue, "
                        "especially vocative ('rest in peace, "
                        "rabbit'), are friends in voice chat. (3) "
                        "Never invent 'chat' or 'viewers' as an "
                        "entity. (4) An entity is only 'in_game' if "
                        "it is well-known canon for the game (e.g. "
                        "'Radahn' in Elden Ring) AND verifiable; "
                        "otherwise it is 'unknown'."
                    ),
                    "items": {
                        "type": "object",
                        "required": [
                            "name", "role", "evidence_quote",
                        ],
                        "properties": {
                            "name": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "friend_voice_chat",
                                    "in_game_canonical",
                                    "likely_mishearing",
                                    "unknown",
                                ],
                            },
                            "evidence_quote": {
                                "type": "string",
                                "description": (
                                    "Verbatim transcript line where "
                                    "the name appears, with source "
                                    "('mic:' or 'desktop:')."
                                ),
                            },
                        },
                    },
                },
                "comedic_premise": {
                    "type": "string",
                    "description": (
                        "One sentence: what makes this clip funny or "
                        "shareable. Usually the gap between the "
                        "literal event and the streamer's reaction."
                    ),
                },
            },
        },
        "chosen_title": {
            "type": "string",
            "description": (
                "The single YouTube metadata title, short and snappy, "
                f"no longer than {MAX_TITLE_CHARS} characters. Must be "
                "true to clip_interpretation.literal_event. Must NOT "
                "include any friend handle (any entity tagged "
                "friend_voice_chat) — substitute with a generic role "
                "('my friend', 'my teammate', 'my buddy') or omit. "
                "This channel is small; named friends carry no "
                "recognition value to swipe-by viewers and weaken the "
                "hook."
            ),
        },
        "chosen_title_reasoning": {
            "type": "string",
            "description": (
                "2–4 sentences. FIRST sentence: confirm the title is "
                "true to literal_event and does not misread any "
                "named_entity. THEN cite synthesis patterns and "
                "reference shorts that support it."
            ),
        },
        "candidates_considered": {
            "type": "array",
            "description": (
                "Every candidate title that was generated, including the "
                "chosen one. The chosen one MUST appear here with "
                "verdict='chosen'. Aim for 3–5 candidates total."
            ),
            "items": {
                "type": "object",
                "required": ["text", "verdict", "reasoning"],
                "properties": {
                    "text": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["chosen", "rejected"],
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Why this candidate was chosen or rejected, "
                            "in concrete terms (which pattern, which "
                            "reference)."
                        ),
                    },
                },
            },
        },
        "patterns_applied": {
            "type": "array",
            "description": (
                "Synthesis-derived patterns this title leans into. "
                "Each entry: 'pattern_name — concrete reason it fits this clip'."
            ),
            "items": {"type": "string"},
        },
        "patterns_avoided": {
            "type": "array",
            "description": (
                "Synthesis-derived anti-patterns explicitly avoided. "
                "Each entry: 'pattern_name — why we avoided it'."
            ),
            "items": {"type": "string"},
        },
        "reference_comparisons": {
            "type": "array",
            "description": (
                "How specific reference shorts (by exact title) "
                "informed the chosen title."
            ),
            "items": {
                "type": "object",
                "required": ["reference_title", "takeaway"],
                "properties": {
                    "reference_title": {"type": "string"},
                    "takeaway": {"type": "string"},
                },
            },
        },
    },
}


class TitleGenerator:
    def __init__(
        self,
        analyzer_client: Optional[AnalyzerClient] = None,
        log_func=print,
    ) -> None:
        self._log = log_func
        self._analyzer = analyzer_client or AnalyzerClient(log_func=log_func)
        self._client = None  # lazy — avoid Gemini-client init if we bail early

    # ── Public entry point ────────────────────────────────────────────────────

    def generate(
        self,
        mic_transcriptions_raw: List[str],
        desktop_transcriptions_raw: List[str],
        original_duration: Optional[float],
        final_duration: Optional[float],
        trim_segments: Optional[List[Tuple[float, float]]],
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a dict on success:
            {
              "text": "<chosen title string>",
              "analysis": { ...detailed reasoning... },
              "provenance": { model, generated_at, ... }
            }
        Returns None on any failure (analyzer down, missing synthesis,
        empty transcript, Gemini failure, malformed response).
        """
        # ── Strict-fail preconditions ──────────────────────────────────────────
        if not self._has_usable_dialogue(
                mic_transcriptions_raw, desktop_transcriptions_raw):
            self._log(
                "[title] no usable dialogue in transcripts — "
                "skipping title generation")
            return None

        synthesis = self._analyzer.read_result(
            f"{CHANNEL_HANDLE}.synthesis.json")
        if not synthesis:
            self._log(
                "[title] analyzer synthesis unavailable — "
                "skipping title generation")
            return None

        per_video = self._analyzer.read_result(f"{CHANNEL_HANDLE}.json")
        if not per_video or not per_video.get("shorts"):
            self._log(
                "[title] analyzer per-video data unavailable — "
                "skipping title generation")
            return None

        # ── Build prompt + call Gemini ─────────────────────────────────────────
        prompt = self._build_prompt(
            synthesis=synthesis,
            shorts=per_video["shorts"],
            mic_transcriptions=mic_transcriptions_raw,
            desktop_transcriptions=desktop_transcriptions_raw,
            original_duration=original_duration,
            final_duration=final_duration,
            trim_segments=trim_segments,
        )

        try:
            if self._client is None:
                self._client = get_gemini_client()
            response = self._client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=get_safety_settings(),
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.HIGH,
                    ),
                    response_mime_type="application/json",
                    response_json_schema=_RESPONSE_SCHEMA,
                    temperature=0.3,
                ),
            )
        except Exception as e:
            self._log(f"[title] Gemini call failed: {e}")
            return None

        text = getattr(response, "text", None)
        if not text:
            self._log("[title] Gemini returned empty response")
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self._log(f"[title] could not parse Gemini JSON: {e}")
            return None

        chosen = (data.get("chosen_title") or "").strip()
        if not chosen:
            self._log("[title] Gemini response missing chosen_title")
            return None

        # Post-validate: hard-deny banned terms (creative substitution is
        # supposed to happen in the prompt; this is the safety net).
        banned_hit = _banned_match(chosen)
        if banned_hit:
            self._log(
                f"[title] rejected — chosen title contains banned term "
                f"'{banned_hit}': \"{chosen}\""
            )
            return None

        # Length cap is softer — flag in the analysis but still ship the
        # title so the operator can decide whether to trim it.
        too_long = len(chosen) > MAX_TITLE_CHARS

        analysis = {
            "clip_interpretation": data.get("clip_interpretation", {}),
            "chosen_title_reasoning": data.get("chosen_title_reasoning", ""),
            "candidates_considered": data.get("candidates_considered", []),
            "patterns_applied": data.get("patterns_applied", []),
            "patterns_avoided": data.get("patterns_avoided", []),
            "reference_comparisons": data.get("reference_comparisons", []),
        }
        if too_long:
            analysis["length_warning"] = (
                f"Chosen title is {len(chosen)} chars "
                f"(cap is {MAX_TITLE_CHARS})."
            )

        provenance = {
            "model": MODEL_PRO,
            "generated_at": datetime.datetime.now().isoformat(),
            "channel_handle": CHANNEL_HANDLE,
            "synthesis_total_shorts": (
                synthesis.get("metadata", {}).get("total_shorts")),
            "synthesis_small_corpus_warning": (
                synthesis.get("metadata", {}).get("small_corpus_warning")),
            "synthesis_generated_at": (
                synthesis.get("metadata", {}).get("generated_at")),
        }

        self._log(
            f"[title] chose: \"{chosen}\" "
            f"(considered {len(analysis['candidates_considered'])} candidates)"
        )
        return {
            "text": chosen,
            "analysis": analysis,
            "provenance": provenance,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _has_usable_dialogue(
        mic: List[str], desktop: List[str],
    ) -> bool:
        # Transcriber output is "start-end: word" strings. Strip timestamps
        # and check there's any actual content.
        for line in (mic or []) + (desktop or []):
            text = re.sub(r"^\s*\d+\.\d+-\d+\.\d+:\s*", "", line).strip()
            if text:
                return True
        return False

    @staticmethod
    def _strip_timestamps(lines: List[str]) -> str:
        out = []
        for line in lines:
            text = re.sub(r"^\s*\d+\.\d+-\d+\.\d+:\s*", "", line).strip()
            if text:
                out.append(text)
        return " ".join(out)

    @staticmethod
    def _trim_summary(
        original_duration: Optional[float],
        final_duration: Optional[float],
        trim_segments: Optional[List[Tuple[float, float]]],
    ) -> str:
        parts = []
        if original_duration:
            parts.append(f"raw: {original_duration:.1f}s")
        if final_duration:
            parts.append(f"final: {final_duration:.1f}s")
        if trim_segments:
            seg_str = ", ".join(
                f"{s:.1f}-{e:.1f}" for s, e in trim_segments[:8])
            if len(trim_segments) > 8:
                seg_str += f", … (+{len(trim_segments) - 8} more)"
            parts.append(f"kept: [{seg_str}]")
        return "; ".join(parts) if parts else "duration/trim unknown"

    def _build_prompt(
        self,
        synthesis: Dict[str, Any],
        shorts: List[Dict[str, Any]],
        mic_transcriptions: List[str],
        desktop_transcriptions: List[str],
        original_duration: Optional[float],
        final_duration: Optional[float],
        trim_segments: Optional[List[Tuple[float, float]]],
    ) -> str:
        narrative = synthesis.get("narrative", {}) or {}
        meta = synthesis.get("metadata", {}) or {}

        # Reference block: each of the 5 shorts with title, breakout score,
        # hook, and what made it work. The model uses these as both worked
        # examples and a reverse-engineering target.
        reference_blocks = []
        for s in shorts:
            ga = s.get("gemini_analysis", {}) or {}
            title_obj = ga.get("title", {}) or {}
            hook_obj = ga.get("hook", {}) or {}
            tags_obj = ga.get("tags", {}) or {}
            ref_lines = [
                f"### {s.get('title', '(untitled)')}",
                (f"breakout_score: {s.get('breakout_score')} | "
                 f"views: {s.get('views')} | "
                 f"duration: {s.get('duration_seconds')}s"),
                (f"title_why_it_worked: "
                 f"{title_obj.get('why_it_worked', '(none)')}"),
                f"hook: {hook_obj.get('description', '(none)')}",
                (f"why_the_video_worked: "
                 f"{(ga.get('why_the_video_worked') or '')[:600]}"),
            ]
            tag_summary = []
            for axis in ("title_mechanics", "title_video_relationship",
                         "hook_type"):
                vals = tags_obj.get(axis) or []
                if vals:
                    tag_summary.append(f"{axis}={','.join(vals)}")
            if tag_summary:
                ref_lines.append("relevant_tags: " + " | ".join(tag_summary))
            reference_blocks.append("\n".join(ref_lines))

        synthesis_block = "\n\n".join([
            (f"corpus: {meta.get('total_shorts')} shorts "
             f"(small_corpus_warning="
             f"{bool(meta.get('small_corpus_warning'))})"),
            f"top_quintile_signature:\n{narrative.get('top_quintile_signature', '')}",
            (f"bottom_quintile_signature:\n"
             f"{narrative.get('bottom_quintile_signature', '')}"),
            (f"load_bearing_patterns:\n"
             f"{narrative.get('load_bearing_patterns', '')}"),
            (f"conditional_insights:\n"
             f"{narrative.get('conditional_insights', '')}"),
            f"cautions:\n{narrative.get('cautions', '')}",
        ])

        mic_text = self._strip_timestamps(mic_transcriptions)
        desktop_text = self._strip_timestamps(desktop_transcriptions)
        trim_str = self._trim_summary(
            original_duration, final_duration, trim_segments)

        # Cap transcripts to keep the prompt sane. Mic is the streamer's voice
        # and is the most signal-rich for titling; desktop is game/ambient.
        if len(mic_text) > 3000:
            mic_text = mic_text[:3000] + " …(truncated)"
        if len(desktop_text) > 1500:
            desktop_text = desktop_text[:1500] + " …(truncated)"

        return f"""You are crafting a single YouTube short metadata title for the channel
@{CHANNEL_HANDLE}. Your job is to study what has worked on this channel and
propose a title that maximises the chance of a swipe-stop on the YouTube
shorts feed.

This title is metadata only — it is shown under the video on YouTube. It is
NOT burned into the video. Subtitles, on-screen text, and overlays are
handled elsewhere and must be ignored when crafting the title.

Hard constraints on the title:
- Single line, no line breaks.
- {MAX_TITLE_CHARS} characters or fewer (count carefully — shorts feed
  truncates aggressively).
- Snappy, scroll-stopping, mobile-readable.
- No clickbait deception — the title must be honest about the clip content.
- Do NOT prepend or append "#shorts" — that's added downstream if needed.

Content policy (these will get the title auto-rejected by the publisher):
- The word "fuck" (and any inflection: fucking, fucked, etc.) is BANNED.
  Substitute with creative alternatives that capture the same energy
  ("freaking", "absolutely cooked", "wrecked", "lost it", etc.) or
  rewrite the title to drop the intensifier entirely.
- Explicit sexual anatomy terms are BANNED — including but not limited to
  "cum", "dick", "asshole", "pussy", "clit", "vagina". The word "ass" by
  itself is fine in non-sexual contexts (e.g. "got my ass kicked").
- For any clip content that is suggestive or sexual in nature, get
  CREATIVE with the language. Invent playful, original euphemisms that
  fit the moment rather than reaching for the literal anatomical word.
  Think on the level of a comedian writing a TV-safe punchline. Different
  clips deserve different invented terms — do not fall back to a stock
  list. The goal is humor, not censorship; a great euphemism makes the
  title funnier than the literal word would have.
- When in doubt, write a title that a YouTube ads-eligible channel would
  ship. If even your euphemism feels like a near-miss of the banned word,
  pick a different angle entirely.

# Channel insights — synthesis from the analyzer
The synthesis below is your PRIMARY guide for picking the strongest
title once you have honest candidates. Use it actively — what scores
on this channel, what tanks, which mechanics differentiate breakouts.
Two-layer rule: (1) FIDELITY is the floor — any candidate that
misreads the clip or names a friend handle is rejected before any
pattern reasoning happens. (2) Among the candidates that clear the
floor, the synthesis is the deciding factor: pick the one that best
matches load-bearing breakout patterns and avoids bottom-quintile
traits. Do NOT retrofit a title to a pattern; do select between
honest candidates using the data.

{synthesis_block}

# Reference shorts ({len(shorts)}) — what has actually shipped on this channel
Concrete worked examples of what has scored on this channel. Use them
actively to compare candidates — which references does each candidate
echo? Which bottom-quintile traps does it avoid? If a candidate has
no genuine analogue in the references, that's a signal worth weighing
(novel ≠ wrong, but it lacks evidence). Do not force a comparison
that isn't real.

{chr(10).join(reference_blocks)}

# THIS clip — what we are titling
trim_summary: {trim_str}

# IMPORTANT — source reliability
mic_dialogue is the streamer's voice. It is the MOST RELIABLE signal in
this clip. Treat it as social commentary directed at friends in voice
chat — not as narration. The streamer's reaction is often the actual
hook of the short.

desktop_dialogue is Whisper's transcription of chaotic game audio
(music, NPC voice lines, sound effects, multiple overlapping sources).
It is UNRELIABLE for naming. Use it ONLY for vibe/atmosphere
("dramatic music", "boss roar"). Proper-noun-looking strings that
appear ONLY in desktop are almost certainly mishearings — do NOT treat
them as real entities, do NOT put them in the title.

There is NO chat data in this pipeline. Never invent "chat", "viewers",
"stream chat", or attribute any action to them. They do not exist in
your inputs.

mic_dialogue:
{mic_text or '(silent)'}

desktop_dialogue (UNRELIABLE — vibe only):
{desktop_text or '(silent)'}

# Your task
1. INTERPRET the clip first — fill in clip_interpretation. Honesty over
   creativity. Every claim needs a transcript quote backing it.
   - literal_event: state ONLY what the transcripts can support. If
     mic + desktop don't establish a specific event, say so coarsely
     ("streamer reacts to a death during a boss fight, exact cause
     unclear"). DO NOT invent an event to make a better story.
   - streamer_reaction: quote or near-quote what the streamer says.
     This is the strongest evidence in the clip.
   - named_entities: apply the rules from the schema strictly.
     Desktop-only names are likely_mishearing — do not promote them.
   - comedic_premise: usually the streamer's reaction itself, or the
     gap between reaction and event.

2. Generate 3–5 candidate titles. Because the in-game event is often
   uncertain, REACTION-FIRST titles (echoing or paraphrasing what the
   streamer said, or naming the emotion) are usually safer than
   event-first titles. Channel breakouts often hook on reaction.
   Hard rules:
   - Never name an entity flagged 'likely_mishearing' or 'unknown'.
   - Never reference 'chat' or 'viewers'.
   - Never claim a spoken line you cannot quote from the transcripts.
   - Friend handles (entities tagged friend_voice_chat) MUST NOT
     appear in the title. Substitute a generic role ('my friend',
     'my teammate', 'my buddy', 'homie') or omit. This channel is
     small; viewers do not know your friends by name. Generic
     phrasing is NOT a downgrade — it is usually stronger for
     swipe-by viewers, because it lets them map themselves into the
     scene. Reject the bias that "specific = better" here.

3. For each candidate, name the synthesis pattern(s) and reference
   short(s) it genuinely echoes. The synthesis is your primary tool
   for ranking candidates — be specific about which load-bearing
   pattern each candidate leans on and which bottom-quintile traps it
   avoids. "No relevant pattern" is acceptable but should be rare; if
   you write it, note that the candidate lacks channel evidence.

4. Pick exactly one as the chosen title. In chosen_title_reasoning,
   first confirm fidelity to literal_event and named_entities, THEN
   make the data-driven case: which load-bearing pattern wins it,
   which bottom-quintile trait the rejected candidates fell into,
   which reference shorts it most closely echoes.

5. patterns_applied: the load-bearing patterns the chosen title
   actually echoes. Each entry must justify why THIS clip supports
   the pattern (genuine echo, not retrofit).

6. patterns_avoided: bottom-quintile anti-patterns you steered around.

7. reference_comparisons: only references that are genuine analogues.
   An empty list is fine if none apply.

Return only the JSON object matching the response schema. No prose outside it.
"""
