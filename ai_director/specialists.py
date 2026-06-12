# ai_director/specialists.py
"""
Reaction classifier for the content-driven AI Director.

This used to host a fleet of specialists (text/audio/visual + an editorial
conflict-resolver). The director is now candidate-driven: vision intensity is
classified by the vision analyzer, audio is handled by energy envelopes, and
conflicts are settled by scoring — so the only LLM judgment left here is the
high-volume "is this spoken line a standout reaction?" call, kept on local
Ollama where it belongs.
"""

from typing import Tuple

from llm.ollama_integration import OllamaLLM


class ReactionClassifier:
    """Classifies a single player utterance as wild / awkward / normal."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.llm = OllamaLLM(log_func=self.log_func)
        self.log_func("🤖 AI Director reaction classifier ready.")

    @property
    def available(self) -> bool:
        return self.llm.available

    def classify(self, text: str) -> Tuple[str, float]:
        """Return (label, confidence) where label ∈ {wild, awkward, normal}.

        'wild' is reserved for genuinely standout/off-the-wall reactions, not
        any profanity or routine commentary — this is the gate the operator
        wants tightened so the camera stops over-firing.
        """
        text = (text or "").strip()
        if not text or not self.llm.available:
            return "normal", 0.0

        prompt = (
            "You are an editor for a gaming highlights channel. A streamer "
            "said the line below. Classify how zoom-worthy the player's "
            "REACTION is into exactly one word:\n"
            "'wild': genuinely off-the-wall — outrageous, shocking, hilarious, "
            "or a big emotional outburst (scream, freakout, unhinged take). "
            "NOT routine swearing or mild commentary.\n"
            "'awkward': social cringe, an uncomfortable non-sequitur, a "
            "conversational whiff.\n"
            "'normal': ordinary commentary — most lines are normal.\n\n"
            f"Line: \"{text}\"\n\n"
            "Respond with ONLY the classification word.\nClassification:"
        )
        try:
            raw = (self.llm.generate(prompt) or "").lower()
        except Exception as e:
            self.log_func(f"   ⚠ reaction classify failed: {e}")
            return "normal", 0.0

        if "wild" in raw:
            return "wild", 0.85
        if "awkward" in raw:
            return "awkward", 0.75
        return "normal", 0.9
