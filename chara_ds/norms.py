"""Lookup helpers for age/gender norm snippets."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_norm_index(norm_dir: Path) -> Dict[str, Any]:
    path = norm_dir / "index.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def hash_norm_source(norm_dir: Path, legacy_text: str) -> str:
    if norm_dir.exists():
        parts: List[str] = []
        for path in sorted(norm_dir.glob("*.json")):
            try:
                parts.append(path.name)
                parts.append(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        if parts:
            return _sha256_text("\n".join(parts))
    return _sha256_text(legacy_text or "")


def _norm_entries(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = index.get("attributes")
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def _profile_text(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    fields = []
    for key in ("role", "age", "age_band", "gender", "occupation", "public_profile", "personality", "speech_style"):
        value = profile.get(key)
        if isinstance(value, str):
            fields.append(value)
        elif isinstance(value, dict):
            fields.extend(str(v) for v in value.values() if isinstance(v, str))
    return " ".join(fields)


def select_norm_ids_from_text(index: Dict[str, Any], text: str, *, limit: int = 3) -> List[str]:
    hay = (text or "").lower()
    selected: List[str] = []
    for item in _norm_entries(index):
        norm_id = str(item.get("id") or "")
        if not norm_id:
            continue
        needles = [
            str(item.get("label_ja") or ""),
            str(item.get("gender") or ""),
            str(item.get("age_band") or ""),
            str(item.get("school_stage") or ""),
            norm_id.replace("_", " "),
        ]
        aliases = item.get("aliases")
        if isinstance(aliases, list):
            needles.extend(str(a) for a in aliases)
        if any(n and n.lower() in hay for n in needles):
            selected.append(norm_id)
        if len(selected) >= limit:
            break
    return selected


def select_norm_ids_for_profile(index: Dict[str, Any], profile: Any, *, limit: int = 2) -> List[str]:
    return select_norm_ids_from_text(index, _profile_text(profile), limit=limit)


def load_selected_norms(norm_dir: str, index: Dict[str, Any], ids: List[str]) -> List[Dict[str, Any]]:
    base = Path(norm_dir)
    by_id = {str(item.get("id") or ""): item for item in _norm_entries(index)}
    loaded: List[Dict[str, Any]] = []
    for norm_id in ids:
        item = by_id.get(norm_id)
        if not item:
            continue
        filename = str(item.get("file") or "")
        if not filename:
            continue
        path = (base / filename).resolve()
        try:
            if base.resolve() not in path.parents:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            loaded.append(data)
    return loaded
