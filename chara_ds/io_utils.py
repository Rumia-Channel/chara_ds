"""File IO, hashing, JSON parsing, and small helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import PersonaLine, PromptBundle
from .norms import hash_norm_source, load_norm_index


JSONL_WRITE_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompts(prompt_dir: str) -> PromptBundle:
    base = Path(prompt_dir)
    norm_dir = base / "age_gender_norms"
    legacy_norms = base / "age_gender_norms.txt"
    files = {
        "persona_controller": base / "persona_controller.txt",
        "turn_controller": base / "turn_controller.txt",
        "actor": base / "actor.txt",
        "actor_guard": base / "actor_guard.txt",
        "age_gender_norms": legacy_norms,
    }

    required_keys = ("persona_controller", "turn_controller", "actor")
    missing = [str(files[k]) for k in required_keys if not files[k].exists()]
    if missing:
        raise FileNotFoundError(f"missing prompt files: {missing}")

    legacy_norm_text = read_text(str(legacy_norms)).strip() if legacy_norms.exists() else ""
    norm_index = load_norm_index(norm_dir)
    return PromptBundle(
        persona_controller=read_text(str(files["persona_controller"])).strip(),
        turn_controller=read_text(str(files["turn_controller"])).strip(),
        actor=read_text(str(files["actor"])).strip(),
        actor_guard=read_text(str(files["actor_guard"])).strip()
        if files["actor_guard"].exists()
        else "",
        age_gender_norms=legacy_norm_text,
        age_gender_norms_dir=str(norm_dir) if norm_dir.exists() else "",
        age_gender_norms_index=norm_index,
        age_gender_norms_sha256=hash_norm_source(norm_dir, legacy_norm_text),
    )


def load_persona_lines(path: str) -> List[PersonaLine]:
    items: List[PersonaLine] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()

            if not text:
                continue

            if text.startswith("#"):
                continue

            items.append(
                PersonaLine(
                    line_number=line_number,
                    text=text,
                    sha256=sha256_text(text),
                )
            )

    if not items:
        raise ValueError(f"no persona seeds found in {path}")

    return items


def parse_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("empty model content")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def safe_mkdir_for_file(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    safe_mkdir_for_file(path)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

    with JSONL_WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


_CID_INDEX_RE = re.compile(r"persona_deepseek_triple_ja_(\d+)")


def read_done_indices(path: str) -> set:
    """Return the set of zero-based idx0 values already present in the output
    jsonl. Used by --resume so that, with parallel workers, we skip exactly
    the conversations that completed and re-run any indices that were in
    flight when the previous run was stopped.

    Note: conversation_id encodes conversation_index = idx0 + 1
    (see runner.run_one_conversation_task), so we subtract 1 here to align
    with work_indices, which iterates over zero-based idx0.
    """

    if not os.path.exists(path):
        return set()

    done: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = rec.get("id") or rec.get("conversation_id")
            if not isinstance(cid, str):
                continue
            m = _CID_INDEX_RE.search(cid)
            if not m:
                continue
            try:
                conversation_index = int(m.group(1))
            except ValueError:
                continue
            if conversation_index <= 0:
                continue
            done.add(conversation_index - 1)
    return done


def sort_jsonl_by_conversation_id(path: str) -> None:
    if not os.path.exists(path):
        return

    with JSONL_WRITE_LOCK:
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = [ln for ln in (line.rstrip("\n") for line in f) if ln.strip()]

        if not raw_lines:
            return

        decoded: list[tuple[str, int, str]] = []
        for idx, ln in enumerate(raw_lines):
            try:
                obj = json.loads(ln)
                cid = obj.get("id") or obj.get("conversation_id") or ""
            except Exception:
                cid = ""
            decoded.append((cid, idx, ln))

        decoded.sort(key=lambda t: (t[0] == "", t[0], t[1]))

        tmp_path = path + ".sort.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for _, _, ln in decoded:
                f.write(ln + "\n")
        os.replace(tmp_path, path)


def clip_string(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"... [truncated {len(s) - max_chars} chars]"
