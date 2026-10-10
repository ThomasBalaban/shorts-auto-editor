"""
Atomic JSON writes + race-free metadata path allocation.

MISSION Phase 1.3 / Phase 3.2: the editor used to write metadata with a
plain open("w") — a torn read on the strategist side stops the thinker —
and both the GUI and the API allocated shorts_metadata_<N> with an
unlocked scan-then-write.
"""

import json
import os
import re
import tempfile
from typing import Any, Tuple

_METADATA_RE = re.compile(r"shorts_metadata_(\d+)\.json$")


def atomic_json_dump(obj: Any, path: str) -> None:
    """Write JSON to ``path`` via tmp-file + os.replace (never torn).

    Mirrors the strategist's own recommendations-store write pattern.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def claim_next_metadata_path(shorts_data_dir: str) -> Tuple[str, int]:
    """Atomically claim the next shorts_metadata_<N>.json filename.

    O_CREAT|O_EXCL makes the claim itself atomic: if two writers race,
    one loses the create and retries with the next index. The claimed
    file exists (empty) on return; follow up with atomic_json_dump.

    Returns (path, index).
    """
    os.makedirs(shorts_data_dir, exist_ok=True)
    while True:
        max_index = 0
        for fname in os.listdir(shorts_data_dir):
            m = _METADATA_RE.match(fname)
            if m:
                max_index = max(max_index, int(m.group(1)))
        index = max_index + 1
        path = os.path.join(shorts_data_dir, f"shorts_metadata_{index}.json")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return path, index
        except FileExistsError:
            continue  # lost the race — rescan and try the next index
