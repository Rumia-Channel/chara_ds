"""Per-turn conversation cache for resumable generation.

When a conversation is interrupted mid-flight (process killed, stop request,
machine reboot, etc.) the partially-generated turns are lost because we only
write to the output jsonl after the entire conversation finishes. To avoid
re-doing all the expensive turn-controller / actor calls on the next run, we
persist a compact snapshot after the persona controller finishes and after
every successful turn into ``<out>.cache/<conversation_id>.json``.

Each cache file is independent (one per conversation_id), so concurrent
workers never contend on the same file. Writes are atomic: we write to a
``.tmp`` sibling and ``os.replace`` it.

A ``signature`` is embedded in every cache file. The signature combines all
inputs that, if changed, would invalidate the partial conversation
(prompt text hashes, model, seed, target_turns, thinking flags, max-token
policy, persona line sha, etc.). On resume the caller computes the current
signature and compares; on mismatch the cache is dropped and the
conversation is regenerated from scratch.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# Lock to make per-file writes serialised within a single process. Different
# conversations write to different files, but we still want to be sure that
# two threads writing the same file (which can happen if a retry path saves
# again before the previous save returns) cannot interleave.
_CACHE_WRITE_LOCK = threading.Lock()


def _safe_filename(conversation_id: str) -> str:
    # conversation_id is already a safe identifier
    # (e.g. "persona_deepseek_triple_ja_00000022") but be defensive.
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in conversation_id)


def cache_path_for(cache_dir: str, conversation_id: str) -> str:
    return str(Path(cache_dir) / f"{_safe_filename(conversation_id)}.json")


def compute_signature(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ensure_cache_dir(cache_dir: str) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


def save_turn_cache(cache_dir: str, conversation_id: str, payload: Dict[str, Any]) -> None:
    ensure_cache_dir(cache_dir)
    path = cache_path_for(cache_dir, conversation_id)
    tmp = path + ".tmp"
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _CACHE_WRITE_LOCK:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)


def load_turn_cache(cache_dir: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    path = cache_path_for(cache_dir, conversation_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_turn_cache(cache_dir: str, conversation_id: str) -> None:
    path = cache_path_for(cache_dir, conversation_id)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass
