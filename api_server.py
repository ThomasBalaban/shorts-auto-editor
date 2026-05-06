"""
shorts-auto-editor Headless API Server
REST API for the Hub to control video processing without the GUI.
Port: 9020
"""
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PORT = 9020
SETTINGS_FILE = os.path.join(_HERE, "..", "youtube_hub", "hub_settings.json")

# ─── Shared state ─────────────────────────────────────────────────────────────

_files: List[Dict[str, Any]] = []
_settings: Dict[str, Any] = {
    "animation_type": "Auto",
    "sync_offset": -0.15,
    "output_dir": os.path.join(os.path.expanduser("~"), "Desktop"),
    "enable_trimming": True,
}
_logs: deque = deque(maxlen=50000)
_processing = False
_stop_requested = False
_current_index = -1
_lock = threading.Lock()
_session_log_path: str = ""
_session_log_lock = threading.Lock()


def _load_settings() -> None:
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        import json
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k in _settings:
            if k in saved:
                _settings[k] = saved[k]
        print(f"[settings] Loaded from {SETTINGS_FILE}", flush=True)
    except Exception as e:
        print(f"[settings] Could not load {SETTINGS_FILE}: {e}", flush=True)


def _save_settings() -> None:
    try:
        import json
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_settings, f, indent=2)
    except Exception as e:
        print(f"[settings] Could not save {SETTINGS_FILE}: {e}", flush=True)


_load_settings()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _logs.append(line)
    print(line, flush=True)
    path = _session_log_path
    if path:
        try:
            with _session_log_lock, open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[session-log] write failed: {e}", flush=True)


def _unique_output_path(input_path: str, output_dir: str) -> str:
    name = os.path.splitext(os.path.basename(input_path))[0]
    base = os.path.join(output_dir, f"{name}-as.mp4")
    if not os.path.exists(base):
        return base
    i = 1
    while os.path.exists(os.path.join(output_dir, f"{name}-as-{i}.mp4")):
        i += 1
    return os.path.join(output_dir, f"{name}-as-{i}.mp4")


def _processing_worker() -> None:
    global _processing, _stop_requested, _current_index, _session_log_path

    try:
        from core.video_processor import VideoProcessor
    except Exception as e:
        _log(f"❌ Failed to import VideoProcessor: {e}")
        import traceback
        _log(traceback.format_exc())
        _processing = False
        return

    queued_indices = [
        i for i, f in enumerate(_files) if f["status"] == "queued"
    ]

    # Open a per-batch session .txt log so /session-logs/list picks it up,
    # mirroring main.py. Without this, only the in-memory ring buffer
    # captured run output, so logs vanished on restart and got truncated
    # after maxlen lines.
    output_dir = _settings.get("output_dir") or ""
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d_%H-%M")
            path = os.path.join(output_dir, f"{timestamp}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"--- BATCH PROCESSING STARTED: {timestamp} ---\n")
                f.write(f"Target Directory: {output_dir}\n")
                f.write(f"Files Queued: {len(queued_indices)}\n")
                f.write("Files to be processed:\n")
                for n, idx in enumerate(queued_indices, 1):
                    entry = _files[idx]
                    f.write(
                        f"  {n}. {os.path.basename(entry['input_path'])} -> "
                        f"{os.path.basename(entry['output_path'])}\n"
                    )
                f.write("=" * 60 + "\n\n")
            _session_log_path = path
        except Exception as e:
            print(f"[session-log] could not create log file: {e}", flush=True)
            _session_log_path = ""

    _log(f"=== Batch started: {len(queued_indices)} queued file(s) ===")
    if _session_log_path:
        _log(f"📄 Session log: {_session_log_path}")

    batch_metadata: List[Dict[str, Any]] = []

    for idx in queued_indices:
        if _stop_requested:
            _log("⏹  Stop requested — halting batch.")
            break

        with _lock:
            _current_index = idx
            _files[idx]["status"] = "processing"

        entry = _files[idx]
        _log(f"\n{'='*40}")
        _log(
            f"Processing {idx + 1}/{len(_files)}: "
            f"{os.path.basename(entry['input_path'])}"
        )
        _log(f"{'='*40}")

        try:
            env_flag = os.environ.get(
                "SHORTS_STRATEGIST_ITERATION_LOOP", ""
            ).strip().lower()
            iteration_loop_enabled = env_flag in ("1", "true", "yes", "on")
            _log(
                f"   iteration loop: "
                f"{'ENABLED' if iteration_loop_enabled else 'disabled'} "
                f"(env SHORTS_STRATEGIST_ITERATION_LOOP={env_flag!r})"
            )

            if iteration_loop_enabled:
                # Strategist-driven loop. Each video gets its own metadata
                # file (overwritten across iterations) so the strategist's
                # edit-review task can score it independently. The
                # orchestrator handles per-iteration output paths and
                # picks the final ship version.
                from core.iteration_loop import IterationOrchestrator
                project_root = os.path.dirname(os.path.abspath(__file__))
                shorts_data_dir = os.path.join(project_root, "shorts_data")
                os.makedirs(shorts_data_dir, exist_ok=True)

                # Pick the next available metadata filename. Mirrors the
                # logic in _save_batch_metadata.
                import re
                max_index = 0
                for fname in os.listdir(shorts_data_dir):
                    m = re.match(r"shorts_metadata_(\d+)\.json", fname)
                    if m:
                        max_index = max(max_index, int(m.group(1)))
                per_video_metadata_path = os.path.join(
                    shorts_data_dir, f"shorts_metadata_{max_index + 1}.json")

                out_dir, out_name = os.path.split(entry["output_path"])
                out_stem, out_ext = os.path.splitext(out_name)
                output_template = os.path.join(
                    out_dir, f"{out_stem}-v{{iter}}{out_ext}")

                orch = IterationOrchestrator(
                    max_iterations=3, log_func=_log)
                final_path, meta = orch.run(
                    input_file=entry["input_path"],
                    output_file_template=output_template,
                    metadata_path=per_video_metadata_path,
                    animation_type=_settings["animation_type"],
                    sync_offset=_settings["sync_offset"],
                    detailed_logs=True,
                    final_output_path=entry["output_path"],
                )
                # Iteration-loop path writes its own per-video metadata file;
                # do NOT add to batch_metadata so we don't double-write.
                with _lock:
                    _files[idx]["output_path"] = (
                        final_path or entry["output_path"])
                    _files[idx]["status"] = "done"
                    _files[idx]["title"] = (meta or {}).get("title", "") or ""
            else:
                final_path, _title, meta = VideoProcessor.process_single_video(
                    input_file=entry["input_path"],
                    output_file=entry["output_path"],
                    animation_type=_settings["animation_type"],
                    sync_offset=_settings["sync_offset"],
                    detailed_logs=True,
                    log_func=_log,
                    enable_trimming=_settings["enable_trimming"],
                )
                with _lock:
                    _files[idx]["output_path"] = (
                        final_path or entry["output_path"])
                    _files[idx]["status"] = "done"
                    _files[idx]["title"] = (meta or {}).get("title", "") or ""
                if meta:
                    batch_metadata.append(meta)
            _log(
                f"✅ Done: "
                f"{os.path.basename(_files[idx]['output_path'])}"
            )

        except Exception as e:
            import traceback
            err = str(e)
            with _lock:
                _files[idx]["status"] = "error"
                _files[idx]["error"] = err
            _log(
                f"❌ Error on "
                f"{os.path.basename(entry['input_path'])}: {err}"
            )
            _log(traceback.format_exc())

    if batch_metadata:
        try:
            _save_batch_metadata(batch_metadata)
        except Exception as e:
            _log(f"⚠️ Failed to save batch metadata: {e}")

    with _lock:
        _processing = False
        _current_index = -1
    _log("=== Batch complete ===")
    _session_log_path = ""


def _save_batch_metadata(batch_metadata: List[Dict[str, Any]]) -> None:
    """Mirror main.py: write the batch's per-video metadata to
    shorts_data/shorts_metadata_<N>.json with auto-incrementing index."""
    import json
    import re
    shorts_data_dir = os.path.join(_HERE, "shorts_data")
    os.makedirs(shorts_data_dir, exist_ok=True)
    max_index = 0
    for fname in os.listdir(shorts_data_dir):
        m = re.match(r"shorts_metadata_(\d+)\.json", fname)
        if m:
            max_index = max(max_index, int(m.group(1)))
    next_index = max_index + 1
    json_path = os.path.join(
        shorts_data_dir, f"shorts_metadata_{next_index}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, indent=2, default=str)
    _log(f"✅ Batch metadata saved to: {json_path}")


# ─── Pydantic models ──────────────────────────────────────────────────────────

class SubtitlerSettings(BaseModel):
    animation_type: str
    sync_offset: float
    output_dir: str
    enable_trimming: bool


class FilesPayload(BaseModel):
    paths: List[str]


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log(f"✅ shorts-auto-editor API ready on :{PORT}")
    yield
    _log("shorts-auto-editor API stopping.")


app = FastAPI(title="shorts-auto-editor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "port": PORT}


# ─── Files ────────────────────────────────────────────────────────────────────

@app.get("/files")
def get_files():
    return list(_files)


@app.post("/files")
def add_files(payload: FilesPayload):
    output_dir = _settings["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    existing = {f["input_path"] for f in _files}
    added = 0
    for raw in payload.paths:
        path = raw.strip()
        if not path:
            continue
        if not os.path.exists(path):
            _log(f"⚠️  Skipping missing file: {path}")
            continue
        if path in existing:
            continue
        with _lock:
            _files.append({
                "input_path": path,
                "output_path": _unique_output_path(path, output_dir),
                "status": "queued",
                "error": "",
                "title": "",
            })
        existing.add(path)
        added += 1
    return {"ok": True, "added": added, "total": len(_files)}


@app.delete("/files/{index}")
def remove_file(index: int):
    if _processing:
        raise HTTPException(400, "Cannot remove files while processing")
    if index < 0 or index >= len(_files):
        raise HTTPException(404, "Index out of range")
    with _lock:
        _files.pop(index)
    return {"ok": True, "total": len(_files)}


@app.delete("/files")
def clear_files():
    global _files
    if _processing:
        raise HTTPException(400, "Cannot clear while processing")
    with _lock:
        _files.clear()
    return {"ok": True}


@app.post("/files/reset")
def reset_files():
    """Set all done/error files back to queued."""
    if _processing:
        raise HTTPException(400, "Cannot reset while processing")
    with _lock:
        for f in _files:
            if f["status"] in ("done", "error"):
                f["status"] = "queued"
                f["error"] = ""
    return {"ok": True}


@app.get("/files/browse")
async def browse_files():
    import asyncio

    def _dialog():
        if sys.platform == "darwin":
            import subprocess
            script = (
                "set theFiles to choose file "
                'of type {"mp4", "mkv", "avi"} '
                "with multiple selections allowed "
                'with prompt "Select Video Files"\n'
                "set out to \"\"\n"
                "repeat with f in theFiles\n"
                "    set out to out & POSIX path of f & (ASCII character 10)\n"
                "end repeat\n"
                "return out"
            )
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return []
            return [p for p in r.stdout.strip().splitlines() if p.strip()]
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            paths = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=[("Video files", "*.mp4 *.mkv *.avi")],
            )
            root.destroy()
            return list(paths)

    try:
        paths = await asyncio.get_event_loop().run_in_executor(None, _dialog)
        return {"paths": paths}
    except Exception as e:
        raise HTTPException(500, f"File dialog failed: {e}")


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/settings")
def get_settings():
    return dict(_settings)


@app.post("/settings")
def post_settings(s: SubtitlerSettings):
    dir_changed = s.output_dir != _settings["output_dir"]
    _settings.update({
        "animation_type": s.animation_type,
        "sync_offset": round(s.sync_offset, 3),
        "output_dir": s.output_dir,
        "enable_trimming": s.enable_trimming,
    })
    if dir_changed:
        os.makedirs(s.output_dir, exist_ok=True)
        with _lock:
            for f in _files:
                if f["status"] == "queued":
                    f["output_path"] = _unique_output_path(
                        f["input_path"], s.output_dir)
    _save_settings()
    return {"ok": True}


@app.get("/settings/browse-dir")
async def browse_output_dir():
    import asyncio

    def _dialog():
        if sys.platform == "darwin":
            import subprocess
            script = (
                'POSIX path of (choose folder with prompt '
                '"Select Output Directory")'
            )
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return ""
            return r.stdout.strip().rstrip("/")
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            path = filedialog.askdirectory(title="Select Output Directory")
            root.destroy()
            return path or ""

    try:
        path = await asyncio.get_event_loop().run_in_executor(None, _dialog)
        return {"path": path}
    except Exception as e:
        raise HTTPException(500, f"Directory dialog failed: {e}")


# ─── Session log files ────────────────────────────────────────────────────────

@app.get("/session-logs/list")
def list_session_logs(dir: str = ""):
    target = dir.strip() or _settings.get("output_dir", "")
    if not target or not os.path.isdir(target):
        return {"files": [], "dir": target}
    try:
        files = []
        for name in os.listdir(target):
            if not name.endswith(".txt"):
                continue
            full = os.path.join(target, name)
            if not os.path.isfile(full):
                continue
            stat = os.stat(full)
            files.append({
                "name": name,
                "path": full,
                "modified": stat.st_mtime,
                "size": stat.st_size,
            })
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"files": files, "dir": target}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/session-logs/read")
def read_session_log(path: str = ""):
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"name": os.path.basename(path), "content": content}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Processing ───────────────────────────────────────────────────────────────

@app.post("/process/start")
def start_processing():
    global _processing, _stop_requested
    if _processing:
        return {"ok": False, "reason": "already_running"}
    queued = [f for f in _files if f["status"] == "queued"]
    if not queued:
        return {"ok": False, "reason": "no_queued_files"}
    _stop_requested = False
    _processing = True
    threading.Thread(target=_processing_worker, daemon=True).start()
    return {"ok": True}


@app.post("/process/stop")
def stop_processing():
    global _stop_requested
    _stop_requested = True
    return {
        "ok": True,
        "message": "Stop requested — will halt after current file.",
    }


@app.get("/process/status")
def process_status():
    return {
        "processing": _processing,
        "stop_requested": _stop_requested,
        "current_index": _current_index,
        "total": len(_files),
        "queued": sum(1 for f in _files if f["status"] == "queued"),
        "done": sum(1 for f in _files if f["status"] == "done"),
        "errors": sum(1 for f in _files if f["status"] == "error"),
    }


# ─── Logs ─────────────────────────────────────────────────────────────────────

@app.get("/logs")
def get_logs(last: int = 200):
    return {"lines": list(_logs)[-last:]}


@app.delete("/logs")
def clear_logs():
    _logs.clear()
    return {"ok": True}


# ─── Analyzer bridge (shorts_analyzer @ :9021) ───────────────────────────────

from analyzer_client import AnalyzerClient

_analyzer_client = AnalyzerClient(log_func=_log)


@app.get("/analyzer/status")
def analyzer_status():
    """
    Bridge health probe. Returns connectivity to the shorts_analyzer API plus
    what's available for the hardcoded PeepingOtter channel. Safe to poll —
    never raises, never starts work on the analyzer side.
    """
    return _analyzer_client.status_snapshot()


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")