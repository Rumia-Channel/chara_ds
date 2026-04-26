"""In-memory progress state and HTTP progress server."""

from __future__ import annotations

import hashlib
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


def _compute_static_version() -> str:
    """Hash all served static assets so we can bust intermediary caches.

    We hash file contents (not mtime) so reloads inside a single Python
    process pick up edits even when the filesystem timestamp is the same,
    and so two processes serving identical files share a version string.
    """
    h = hashlib.sha256()
    for filename in sorted({fn for fn, _ in STATIC_FILES.values()}):
        try:
            h.update(filename.encode("utf-8"))
            h.update(b"\0")
            h.update((WEB_DIR / filename).read_bytes())
            h.update(b"\0")
        except FileNotFoundError:
            continue
    return h.hexdigest()[:12]


STATIC_VERSION = _compute_static_version()


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
    "control": {
        "paused": False,
        "stop_requested": False,
    },
}


# -- Pause / stop control -----------------------------------------------------
# PAUSE_EVENT semantics: SET = workers may run, CLEAR = workers must wait.
PAUSE_EVENT = threading.Event()
PAUSE_EVENT.set()
STOP_EVENT = threading.Event()


def is_paused() -> bool:
    return not PAUSE_EVENT.is_set()


def is_stopped() -> bool:
    return STOP_EVENT.is_set()


def wait_if_paused(check_interval: float = 0.5) -> None:
    """Block while paused. Returns immediately if a stop has been requested."""
    while not PAUSE_EVENT.is_set():
        if STOP_EVENT.is_set():
            return
        PAUSE_EVENT.wait(timeout=check_interval)


def _set_control(key: str, value: bool) -> None:
    with PROGRESS_LOCK:
        PROGRESS_STATE["control"][key] = value
        PROGRESS_STATE["updated_at"] = now_iso()
        PROGRESS_STATE["events"].append(
            {
                "time": PROGRESS_STATE["updated_at"],
                "event": "control",
                "key": key,
                "value": value,
            }
        )
        PROGRESS_STATE["events"] = PROGRESS_STATE["events"][-300:]


def request_pause() -> None:
    PAUSE_EVENT.clear()
    _set_control("paused", True)


def request_resume() -> None:
    PAUSE_EVENT.set()
    _set_control("paused", False)


def request_stop() -> None:
    STOP_EVENT.set()
    PAUSE_EVENT.set()  # unblock workers waiting on pause so they can exit
    _set_control("stop_requested", True)
    _set_control("paused", False)


# -- Persona / situation queue registry --------------------------------------
BUFFER_REGISTRY: Dict[str, Any] = {
    "buffer": None,         # PersonaBuffer reference (set by runner)
    "format_path": None,    # path to format.txt
    "initial_count": 0,     # how many lines came from the initial file
    "lock": threading.Lock(),
}


def register_persona_buffer(buffer: Any, format_path: str, initial_count: int) -> None:
    BUFFER_REGISTRY["buffer"] = buffer
    BUFFER_REGISTRY["format_path"] = format_path
    BUFFER_REGISTRY["initial_count"] = int(initial_count)


def _format_lock() -> threading.Lock:
    return BUFFER_REGISTRY["lock"]


def safe_append_line(path: str, text: str) -> None:
    """Append a line to ``path`` while holding the format-file lock.

    Used by the background situation producer so its appends do not race
    with full-file rewrites triggered from the Web UI.
    """
    from .situation_gen import append_line as _append_line  # avoid import cycle

    with _format_lock():
        _append_line(path, text)


def _rewrite_format_file(path: str, items: List[Any]) -> None:
    from .io_utils import safe_mkdir_for_file

    safe_mkdir_for_file(path)
    with _format_lock():
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(getattr(it, "text", str(it)) + "\n")


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


def _read_completed_records(
    out_path: str,
    *,
    limit: int = 50,
    offset: int = 0,
    order: str = "desc",
    include_record: bool = False,
) -> Dict[str, Any]:
    """Stream the output JSONL and return a paginated slice."""
    p = Path(out_path)
    if not p.exists():
        return {"out": out_path, "exists": False, "total": 0, "items": []}

    lines: List[str] = []
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    lines.append(line)
    except Exception as e:
        return {
            "out": out_path,
            "exists": True,
            "total": 0,
            "items": [],
            "error": str(e),
        }

    total = len(lines)
    if order == "desc":
        lines = list(reversed(lines))
    sliced = lines[offset : offset + max(0, limit)]

    items: List[Dict[str, Any]] = []
    for raw in sliced:
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        summary = _summarize_record(rec)
        if include_record:
            summary["record"] = progress_safe(rec, max_string=20000, max_list=400)
        items.append(summary)

    return {
        "out": out_path,
        "exists": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "order": order,
        "items": items,
    }


def _summarize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary used in completed list view."""
    cid = rec.get("conversation_id") or rec.get("id") or "?"
    persona = rec.get("persona") or rec.get("personas") or {}
    if isinstance(persona, dict):
        persona_a = persona.get("A") or {}
        persona_b = persona.get("B") or {}
        persona_summary = {
            "A": (persona_a or {}).get("name") if isinstance(persona_a, dict) else None,
            "B": (persona_b or {}).get("name") if isinstance(persona_b, dict) else None,
        }
    else:
        persona_summary = None
    timeline = (
        rec.get("public_timeline")
        or rec.get("timeline")
        or rec.get("dialogue")
        or []
    )
    if not isinstance(timeline, list):
        timeline = []
    last_msg = ""
    if timeline:
        m = timeline[-1] or {}
        if isinstance(m, dict):
            last_msg = m.get("utterance") or m.get("public_utterance") or ""
    return {
        "conversation_id": cid,
        "turns": len(timeline),
        "personas": persona_summary,
        "situation": clip_string(str(rec.get("situation") or rec.get("scene") or ""), 240),
        "last_utterance": clip_string(str(last_msg or ""), 240),
        "created_at": rec.get("created_at") or rec.get("finished_at") or None,
    }


class ProgressHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = 200,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        # Defeat any intermediary cache (nginx, browser, etc.). The
        # generated HTML also embeds ?v=<STATIC_VERSION> on every static
        # asset so even badly behaved proxies cannot serve a stale bundle.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Static-Version", STATIC_VERSION)
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status=status)

    def _query(self) -> Dict[str, str]:
        from urllib.parse import parse_qs

        q = urlparse(self.path).query
        return {k: v[-1] for k, v in parse_qs(q, keep_blank_values=True).items()}

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/state":
            with PROGRESS_LOCK:
                state = json.loads(json.dumps(PROGRESS_STATE, ensure_ascii=False))
            data = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8")
            return

        if path == "/completed":
            qs = self._query()
            with PROGRESS_LOCK:
                default_out = ((PROGRESS_STATE.get("summary") or {}).get("out")) or ""
            out_path = qs.get("out") or default_out
            if not out_path:
                self._send_json({"error": "no out path known yet"}, status=400)
                return
            try:
                limit = max(1, min(500, int(qs.get("limit") or 50)))
            except Exception:
                limit = 50
            try:
                offset = max(0, int(qs.get("offset") or 0))
            except Exception:
                offset = 0
            order = qs.get("order") or "desc"
            include_record = (qs.get("include_record") or "0") in ("1", "true", "yes")
            payload = _read_completed_records(
                out_path,
                limit=limit,
                offset=offset,
                order=order,
                include_record=include_record,
            )
            self._send_json(payload)
            return

        if path == "/control":
            self._send_json(
                {
                    "paused": is_paused(),
                    "stop_requested": is_stopped(),
                }
            )
            return

        if path == "/situations":
            buf = BUFFER_REGISTRY.get("buffer")
            if buf is None:
                self._send_json({"items": [], "format_path": None, "initial_count": 0})
                return
            items = buf.snapshot()
            initial_count = BUFFER_REGISTRY.get("initial_count", 0)
            with PROGRESS_LOCK:
                summary = dict(PROGRESS_STATE.get("summary") or {})
                active_ids = list((PROGRESS_STATE.get("active") or {}).keys())
            consumed_estimate = int(summary.get("written") or 0) + len(active_ids)
            payload_items = []
            for idx, it in enumerate(items):
                payload_items.append(
                    {
                        "index": idx,
                        "line_number": it.line_number,
                        "text": it.text,
                        "sha256": it.sha256,
                        "origin": "initial" if idx < initial_count else "generated",
                        "likely_consumed": idx < consumed_estimate,
                    }
                )
            self._send_json(
                {
                    "items": payload_items,
                    "format_path": BUFFER_REGISTRY.get("format_path"),
                    "initial_count": initial_count,
                    "consumed_estimate": consumed_estimate,
                    "buffer_finished": buf.is_finished(),
                }
            )
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
            if filename.endswith(".html"):
                # Rewrite static asset URLs so each deploy / restart busts
                # any HTTP cache (nginx, Cloudflare, browser disk cache).
                text = data.decode("utf-8")
                text = text.replace('href="/style.css"', f'href="/style.css?v={STATIC_VERSION}"')
                text = text.replace('src="/app.js"', f'src="/app.js?v={STATIC_VERSION}"')
                # Expose the version to JS so future fetches/imports can append it.
                text = text.replace(
                    "</head>",
                    f'  <meta name="static-version" content="{STATIC_VERSION}">\n'
                    f'  <script>window.__STATIC_VERSION__ = "{STATIC_VERSION}";</script>\n'
                    f"</head>",
                    1,
                )
                data = text.encode("utf-8")
            self._send_bytes(data, content_type)
            return

        self.send_response(404)
        self.end_headers()

    def _read_body_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/control/pause":
            request_pause()
            self._send_json({"paused": True, "stop_requested": is_stopped()})
            return
        if path == "/control/resume":
            if is_stopped():
                self._send_json(
                    {"error": "stop already requested; cannot resume"},
                    status=409,
                )
                return
            request_resume()
            self._send_json({"paused": False, "stop_requested": is_stopped()})
            return
        if path == "/control/stop":
            request_stop()
            self._send_json({"paused": False, "stop_requested": True})
            return

        if path == "/situations":
            data = self._read_body_json()
            text = (data.get("text") or "").strip()
            if not text:
                self._send_json({"error": "text required"}, status=400)
                return
            buf = BUFFER_REGISTRY.get("buffer")
            fmt_path = BUFFER_REGISTRY.get("format_path")
            if buf is None or fmt_path is None:
                self._send_json({"error": "buffer not registered"}, status=503)
                return
            from .config import PersonaLine as _PL
            from .io_utils import sha256_text as _sha
            existing = buf.snapshot()
            next_ln = (max((it.line_number for it in existing), default=0) + 1)
            new_item = _PL(
                line_number=next_ln,
                text=text,
                sha256=_sha(text),
            )
            safe_append_line(fmt_path, text)
            buf.extend([new_item])
            self._send_json(
                {
                    "ok": True,
                    "added": {
                        "line_number": new_item.line_number,
                        "text": new_item.text,
                        "sha256": new_item.sha256,
                    },
                }
            )
            return

        # PATCH-via-POST fallback for environments that block PATCH
        if path.startswith("/situations/") and path.endswith("/edit"):
            try:
                ln = int(path[len("/situations/") : -len("/edit")])
            except ValueError:
                self._send_json({"error": "invalid line_number"}, status=400)
                return
            self._handle_edit(ln)
            return

        self.send_response(404)
        self.end_headers()

    def do_PATCH(self) -> None:  # noqa: N802 (HTTP verb name)
        path = urlparse(self.path).path
        if path.startswith("/situations/"):
            try:
                ln = int(path[len("/situations/") :])
            except ValueError:
                self._send_json({"error": "invalid line_number"}, status=400)
                return
            self._handle_edit(ln)
            return
        self.send_response(404)
        self.end_headers()

    def _handle_edit(self, ln: int) -> None:
        data = self._read_body_json()
        text = (data.get("text") or "").strip()
        if not text:
            self._send_json({"error": "text required"}, status=400)
            return
        buf = BUFFER_REGISTRY.get("buffer")
        fmt_path = BUFFER_REGISTRY.get("format_path")
        if buf is None or fmt_path is None:
            self._send_json({"error": "buffer not registered"}, status=503)
            return
        from .io_utils import sha256_text as _sha
        ok = buf.replace(ln, text, _sha(text))
        if not ok:
            self._send_json({"error": "line_number not found"}, status=404)
            return
        _rewrite_format_file(fmt_path, buf.snapshot())
        self._send_json({"ok": True, "line_number": ln, "text": text})


def start_progress_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ProgressHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
