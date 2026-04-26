"""In-memory progress state and HTTP progress server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .io_utils import clip_string, now_iso


WEB_DIR = Path(__file__).parent / "web"

STATIC_FILES: Dict[str, tuple[str, str]] = {
    "/":           ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css":  ("style.css",  "text/css; charset=utf-8"),
    "/app.js":     ("app.js",     "application/javascript; charset=utf-8"),
}


PROGRESS_LOCK = threading.Lock()
PROGRESS_STATE: Dict[str, Any] = {
    "started_at": None,
    "updated_at": None,
    "status": "idle",
    "summary": {},
    "active": {},
    "latest_public_timeline": [],
    "latest_public_timeline_id": None,
    "last_persona": None,
    "last_controller": None,
    "last_actor": None,
    "events": [],
    "errors": [],
}


def progress_safe(obj: Any, *, max_string: int = 4000, max_list: int = 80) -> Any:
    if obj is None:
        return None

    if isinstance(obj, str):
        return clip_string(obj, max_string)

    if isinstance(obj, dict):
        return {
            str(k): progress_safe(v, max_string=max_string, max_list=max_list)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            progress_safe(v, max_string=max_string, max_list=max_list)
            for v in obj[-max_list:]
        ]

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def progress_update(
    *,
    status: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    current: Optional[Dict[str, Any]] = None,
    latest_public_timeline: Optional[List[Dict[str, Any]]] = None,
    last_persona: Optional[Dict[str, Any]] = None,
    last_controller: Optional[Dict[str, Any]] = None,
    last_actor: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    event: Optional[Dict[str, Any]] = None,
    remove_active: bool = False,
) -> None:
    now = now_iso()

    with PROGRESS_LOCK:
        if PROGRESS_STATE["started_at"] is None:
            PROGRESS_STATE["started_at"] = now

        PROGRESS_STATE["updated_at"] = now

        if status is not None:
            PROGRESS_STATE["status"] = status

        if summary is not None:
            PROGRESS_STATE["summary"] = progress_safe(summary)

        if conversation_id is not None:
            if remove_active:
                PROGRESS_STATE["active"].pop(conversation_id, None)
            else:
                prev = PROGRESS_STATE["active"].get(conversation_id) or {}
                prev_timeline = prev.get("public_timeline")
                if current is not None:
                    new_slot = progress_safe(current) or {}
                    if not isinstance(new_slot, dict):
                        new_slot = {"current": new_slot}
                else:
                    new_slot = prev
                if latest_public_timeline is not None:
                    new_slot["public_timeline"] = progress_safe(latest_public_timeline)
                elif prev_timeline is not None and current is not None:
                    new_slot["public_timeline"] = prev_timeline
                new_slot["updated_at"] = now
                if current is not None or latest_public_timeline is not None:
                    PROGRESS_STATE["active"][conversation_id] = new_slot

        if latest_public_timeline is not None:
            PROGRESS_STATE["latest_public_timeline"] = progress_safe(latest_public_timeline)
            if conversation_id is not None:
                PROGRESS_STATE["latest_public_timeline_id"] = conversation_id

        if last_persona is not None:
            PROGRESS_STATE["last_persona"] = progress_safe(last_persona)

        if last_controller is not None:
            PROGRESS_STATE["last_controller"] = progress_safe(last_controller)

        if last_actor is not None:
            PROGRESS_STATE["last_actor"] = progress_safe(last_actor)

        if error is not None:
            PROGRESS_STATE["errors"].append(progress_safe(error))
            PROGRESS_STATE["errors"] = PROGRESS_STATE["errors"][-100:]

        if event is not None:
            PROGRESS_STATE["events"].append(
                {
                    "time": now,
                    **progress_safe(event),
                }
            )
            PROGRESS_STATE["events"] = PROGRESS_STATE["events"][-300:]




PROGRESS_HTML = ""  # legacy placeholder; UI is now served from chara_ds/web/.


class ProgressHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/state":
            with PROGRESS_LOCK:
                state = json.loads(json.dumps(PROGRESS_STATE, ensure_ascii=False))
            data = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8")
            return

        static = STATIC_FILES.get(path)
        if static is not None:
            filename, content_type = static
            file_path = WEB_DIR / filename
            try:
                data = file_path.read_bytes()
            except FileNotFoundError:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"missing static asset")
                return
            self._send_bytes(data, content_type)
            return

        self.send_response(404)
        self.end_headers()


def start_progress_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ProgressHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
