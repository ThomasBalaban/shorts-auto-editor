"""
Shared Gemini Files-API upload (MISSION Phase 2.2).

The narrative planner used to upload the video, delete it in `finally`,
and the trimmer then re-uploaded the same file seconds later — two
uploads + two ACTIVE-polls per cut (per iteration!). This handle
uploads once, lazily, and both phases reference the same URI; the
processor deletes it after the last consumer.
"""

import os
import time

from google.genai import types


class SharedVideoUpload:
    """Lazy, shareable Files-API upload for one local video file."""

    def __init__(self, client, video_path: str, mime_type: str, log_func):
        self.client = client
        self.video_path = video_path
        self.mime_type = mime_type
        self.log_func = log_func
        self._file = None

    def ensure_active(self, max_wait_seconds: int = 300,
                      upload_retries: int = 3):
        """Upload (once) and poll to ACTIVE. Raises on failure."""
        if self._file is not None:
            return self._file

        last_err = None
        for attempt in range(1, upload_retries + 1):
            try:
                f = self.client.files.upload(
                    file=self.video_path,
                    config={"mime_type": self.mime_type},
                )
                break
            except Exception as e:
                last_err = e
                self.log_func(
                    f"   ⚠ upload attempt {attempt}/{upload_retries} "
                    f"failed: {e}")
                time.sleep(2 * attempt)
        else:
            raise last_err

        poll_start = time.time()
        while True:
            state = f.state.name
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise ValueError(
                    f"Gemini file processing FAILED: "
                    f"{getattr(f, 'error', 'no details')}")
            if time.time() - poll_start > max_wait_seconds:
                raise TimeoutError(
                    f"File did not become ACTIVE within "
                    f"{max_wait_seconds}s (last state: {state})")
            self.log_func(f"   Waiting for ACTIVE (state: {state}) …")
            time.sleep(3)
            f = self.client.files.get(name=f.name)

        size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
        self.log_func(
            f"   ✅ File ACTIVE (shared upload, {size_mb:.1f} MB): {f.name}")
        self._file = f
        return f

    def part(self) -> types.Part:
        f = self.ensure_active()
        return types.Part.from_uri(
            file_uri=f.uri, mime_type=f.mime_type or self.mime_type)

    def delete(self) -> None:
        """Idempotent best-effort cleanup."""
        if self._file is None:
            return
        try:
            self.client.files.delete(name=self._file.name)
            self.log_func("   🗑  Shared upload deleted")
        except Exception:
            pass
        self._file = None
