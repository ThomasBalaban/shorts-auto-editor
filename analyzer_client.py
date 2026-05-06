"""
Thin client for the shorts_analyzer API (default :9021).

Phase 0 of the integration laid out in gameplan.md. Every method degrades
gracefully — if the analyzer is unreachable, callers get None / [] / False
and a single warning log line. The shorts-auto-editor pipeline must continue
to work without it.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = os.environ.get(
    "SHORTS_ANALYZER_URL", "http://localhost:9021")
HEALTH_TIMEOUT = 2.0
READ_TIMEOUT = 10.0

# Only channel shorts-auto-editor ever cuts for. Hardcoded by design — if this
# ever needs to change, edit it here, not in settings.
CHANNEL_HANDLE = "PeepingOtter"

# Sibling-file suffixes the analyzer writes alongside <handle>.json
_SIBLING_SUFFIXES = (
    ".synthesis.json",
    ".tailwind.json",
    ".context.json",
)


class AnalyzerClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        log_func=print,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._log = log_func
        # Last reachability state — used to dedupe poll-driven log spam.
        # None = unknown, True = reachable, False = unreachable.
        self._last_reachable: Optional[bool] = None

    # ── HTTP plumbing ──────────────────────────────────────────────────────────

    def _note_reachable(self, reachable: bool, detail: str = "") -> None:
        if self._last_reachable == reachable:
            return
        self._last_reachable = reachable
        if reachable:
            self._log(f"[analyzer] connected ({self.base_url})")
        else:
            self._log(f"[analyzer] unreachable ({self.base_url}): {detail}")

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, str]] = None,
        timeout: float = READ_TIMEOUT,
    ) -> Optional[Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._note_reachable(True)
                return data
        except urllib.error.URLError as e:
            self._note_reachable(False, str(e.reason))
            return None
        except Exception as e:
            self._note_reachable(False, str(e))
            return None

    # ── Endpoints ──────────────────────────────────────────────────────────────

    def health(self) -> bool:
        data = self._get("/health", timeout=HEALTH_TIMEOUT)
        return bool(data and data.get("status") == "ok")

    def list_results(self) -> List[Dict[str, Any]]:
        data = self._get("/results")
        if not data:
            return []
        return data.get("files", []) or []

    def read_result(self, name: str) -> Optional[Dict[str, Any]]:
        return self._get("/results/read", params={"name": name})

    def get_videos(self, output: str) -> Optional[Dict[str, Any]]:
        return self._get("/videos", params={"output": output})

    # ── Higher-level helpers ──────────────────────────────────────────────────

    @staticmethod
    def _is_channel_handle_file(name: str) -> bool:
        if not name.endswith(".json"):
            return False
        return not any(name.endswith(s) for s in _SIBLING_SUFFIXES)

    def list_channels(self) -> List[str]:
        """Channel handles available in the analyzer's output dir."""
        return sorted({
            f["name"][:-5]  # strip ".json"
            for f in self.list_results()
            if self._is_channel_handle_file(f["name"])
        })

    def channel_file_presence(self, channel_handle: str) -> Dict[str, bool]:
        """Which sibling files exist for this channel, keyed by phase."""
        names = {f["name"] for f in self.list_results()}
        return {
            "analysis":  f"{channel_handle}.json" in names,
            "synthesis": f"{channel_handle}.synthesis.json" in names,
            "tailwind":  f"{channel_handle}.tailwind.json" in names,
        }

    def status_snapshot(self) -> Dict[str, Any]:
        """
        Single call the bridge UI hits — proves end-to-end connectivity and
        surfaces what's available for the hardcoded channel.
        """
        snapshot: Dict[str, Any] = {
            "base_url": self.base_url,
            "checked_at": time.time(),
            "connected": False,
            "channel_handle": CHANNEL_HANDLE,
            "channel_files": None,
        }
        if not self.health():
            return snapshot
        snapshot["connected"] = True
        snapshot["channel_files"] = self.channel_file_presence(CHANNEL_HANDLE)
        return snapshot
