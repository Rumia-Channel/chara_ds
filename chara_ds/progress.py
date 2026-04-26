"""In-memory progress state and HTTP progress server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .io_utils import clip_string, now_iso


PROGRESS_LOCK = threading.Lock()
PROGRESS_STATE: Dict[str, Any] = {
    "started_at": None,
    "updated_at": None,
    "status": "idle",
    "summary": {},
    "active": {},
    "latest_public_timeline": [],
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
            elif current is not None:
                PROGRESS_STATE["active"][conversation_id] = progress_safe(current)

        if latest_public_timeline is not None:
            PROGRESS_STATE["latest_public_timeline"] = progress_safe(latest_public_timeline)

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


PROGRESS_HTML = r"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>Dialogue Generation Progress</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; background: #f7f7f7; color: #222; }
    h1 { font-size: 20px; margin-bottom: 8px; }
    h2 { font-size: 16px; margin-top: 18px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .status { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
    pre { background: #111; color: #eee; padding: 10px; border-radius: 6px; overflow: auto; max-height: 420px; font-size: 12px; line-height: 1.35; }
    .timeline { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px; max-height: 500px; overflow: auto; }
    .msg { margin: 8px 0; padding: 8px; border-radius: 6px; background: #f0f0f0; }
    .speaker { font-weight: 700; margin-right: 6px; }
    .action { color: #734; margin-left: 10px; font-size: 12px; }
  </style>
</head>
<body>
  <h1>DeepSeek Dialogue Generation Progress</h1>
  <div class="card">
    <div id="summary" class="status">loading...</div>
  </div>

  <h2>Active conversations</h2>
  <pre id="active"></pre>

  <h2>Latest public timeline</h2>
  <div id="timeline" class="timeline"></div>

  <div class="grid">
    <div class="card"><h2>Last actor output</h2><pre id="last_actor"></pre></div>
    <div class="card"><h2>Last controller output</h2><pre id="last_controller"></pre></div>
    <div class="card"><h2>Persona</h2><pre id="last_persona"></pre></div>
    <div class="card"><h2>Recent events</h2><pre id="events"></pre></div>
    <div class="card"><h2>Errors</h2><pre id="errors"></pre></div>
  </div>

<script>
async function refresh() {
  const res = await fetch('/state?t=' + Date.now());
  const s = await res.json();
  const summary = s.summary || {};
  document.getElementById('summary').textContent =
    'status: ' + (s.status || '') + '\n' +
    'updated_at: ' + (s.updated_at || '') + '\n' +
    'written: ' + (summary.written ?? '') + ' / ' + (summary.total_requested ?? '') + '\n' +
    'workers: ' + (summary.workers ?? '') + '\n' +
    'out: ' + (summary.out || '');

  document.getElementById('active').textContent = JSON.stringify(s.active || {}, null, 2);

  const timeline = document.getElementById('timeline');
  timeline.innerHTML = '';
  (s.latest_public_timeline || []).forEach(m => {
    const div = document.createElement('div');
    div.className = 'msg';
    const sp = document.createElement('span');
    sp.className = 'speaker';
    sp.textContent = (m.speaker || '?') + ':';
    const tx = document.createElement('span');
    tx.textContent = ' ' + (m.utterance || '');
    div.appendChild(sp);
    div.appendChild(tx);
    if (m.visible_action) {
      const ac = document.createElement('div');
      ac.className = 'action';
      ac.textContent = 'action: ' + JSON.stringify(m.visible_action);
      div.appendChild(ac);
    }
    timeline.appendChild(div);
  });
  timeline.scrollTop = timeline.scrollHeight;

  document.getElementById('last_actor').textContent = JSON.stringify(s.last_actor, null, 2);
  document.getElementById('last_controller').textContent = JSON.stringify(s.last_controller, null, 2);
  document.getElementById('last_persona').textContent = JSON.stringify(s.last_persona, null, 2);
  document.getElementById('events').textContent = JSON.stringify((s.events || []).slice(-80), null, 2);
  document.getElementById('errors').textContent = JSON.stringify((s.errors || []).slice(-30), null, 2);
}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""


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

        if path in ("/", "/index.html"):
            self._send_bytes(PROGRESS_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        self.send_response(404)
        self.end_headers()


def start_progress_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ProgressHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
