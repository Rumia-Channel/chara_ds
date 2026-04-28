"""CLI entrypoint, work scheduling, and per-task wrapper."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .api_client import get_thread_client
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
    FLASH_MODEL,
    PRO_MODEL,
    PersonaLine,
    PromptBundle,
)
from .conversation import estimate_ending_pacing_floor, generate_one_conversation
from .io_utils import (
    append_jsonl,
    count_jsonl_lines,
    load_persona_lines,
    load_prompts,
    now_iso,
    JSONL_WRITE_LOCK,
    read_done_indices,
    sha256_text,
    sort_jsonl_by_conversation_id,
)
from .persona_buffer import PersonaBuffer
from .turn_cache import delete_turn_cache, ensure_cache_dir, load_turn_cache
from .turn_cache import compute_signature, save_turn_cache
from .progress import (
    is_stopped,
    progress_update,
    register_persona_buffer,
    start_progress_server,
    wait_if_paused,
)
from .situation_gen import SITUATION_GEN_MODEL_DEFAULT
from .situation_producer import start_background_producer


_CID_INDEX_RE = re.compile(r"persona_deepseek_triple_ja_(\d+)")


def pick_persona_line_for_index(
    *,
    idx0: int,
    args: argparse.Namespace,
    buffer: PersonaBuffer,
    pool_size: int,
) -> Tuple[PersonaLine, int]:
    if args.sampling == "random":
        rng = random.Random(args.seed + idx0 * 7919)
        line_index = rng.randrange(pool_size)
        variation = 1 + rng.randrange(args.variations_per_line)
    else:
        line_index = (idx0 // args.variations_per_line) % pool_size
        variation = (idx0 % args.variations_per_line) + 1

    persona_line = buffer.wait_for_index(line_index)
    if persona_line is None:
        raise RuntimeError(
            f"persona line {line_index} unavailable: "
            f"buffer size={len(buffer)}, finished={buffer.is_finished()}"
        )
    return persona_line, variation


def conversation_index_from_id(conversation_id: str) -> Optional[int]:
    m = _CID_INDEX_RE.search(conversation_id)
    if not m:
        return None
    try:
        value = int(m.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def actual_turn_count(record: Dict[str, Any]) -> int:
    counts: List[int] = []
    cfg = record.get("generation_config") or {}
    value = cfg.get("actual_turns")
    if isinstance(value, int):
        counts.append(value)
    timeline = record.get("public_timeline")
    if isinstance(timeline, list):
        counts.append(len(timeline))
    turns = record.get("turns")
    if isinstance(turns, list):
        counts.append(len(turns))
    return max(counts) if counts else 0


def read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception as e:
                raise ValueError(f"invalid jsonl at {path}:{line_number}: {e}") from e
            if isinstance(obj, dict):
                records.append(obj)
    return records


def find_duplicate_record_ids(records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    positions: Dict[str, List[int]] = {}
    for pos, record in enumerate(records):
        cid = record.get("id") or record.get("conversation_id")
        if not isinstance(cid, str) or not cid:
            continue
        positions.setdefault(cid, []).append(pos)
    return {cid: pos_list for cid, pos_list in positions.items() if len(pos_list) > 1}


def rewrite_jsonl_records(path: str, records: List[Dict[str, Any]]) -> None:
    tmp_path = path + ".rewrite.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp_path, path)


def replace_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    cid = record.get("id") or record.get("conversation_id")
    if not isinstance(cid, str) or not cid:
        raise ValueError("record missing id for replacement")

    with JSONL_WRITE_LOCK:
        records = read_jsonl_records(path)
        replaced = False
        for pos, existing in enumerate(records):
            existing_cid = existing.get("id") or existing.get("conversation_id")
            if existing_cid == cid:
                records[pos] = record
                replaced = True
                break
        if not replaced:
            records.append(record)
        rewrite_jsonl_records(path, records)


def backfill_short_records_to_cache(
    *,
    records: List[Dict[str, Any]],
    cache_dir: str,
    args: argparse.Namespace,
    prompts: PromptBundle,
    persona_thinking_enabled: bool,
    turn_controller_thinking_enabled: bool,
    actor_thinking_enabled: bool,
    actor_guard_enabled: bool,
    actor_guard_thinking_enabled: bool,
    backup_existing_cache: bool,
) -> Tuple[List[str], set[int]]:
    """Copy short out.jsonl records back into per-turn cache for resumption.

    The output file itself is left untouched here. The caller decides which
    records should be re-run and, on success, replaces those records in out.
    """

    if not records:
        return [], set()

    backfilled_ids: List[str] = []
    backfilled_done_indices: set[int] = set()

    for record in records:
        if actual_turn_count(record) >= args.finish_min_turns:
            continue

        conversation_id = record.get("id") or record.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            continue

        m = _CID_INDEX_RE.search(conversation_id)
        if not m:
            continue

        try:
            conversation_index = int(m.group(1))
        except ValueError:
            continue

        if conversation_index > 0:
            backfilled_done_indices.add(conversation_index - 1)

        source = dict(record.get("source") or {})
        variation = source.get("variation")
        if not isinstance(variation, int):
            variation = (record.get("generation_config") or {}).get("variation")
        if not isinstance(variation, int):
            variation = 1

        line_number = source.get("line_number")
        if not isinstance(line_number, int):
            line_number = 0
        source_text = source.get("text")
        if not isinstance(source_text, str):
            source_text = ""
        source_sha256 = source.get("sha256")
        if not isinstance(source_sha256, str) or not source_sha256:
            source_sha256 = sha256_text(source_text)

        rng = random.Random(args.seed + conversation_index * 1009 + variation * 9173)
        target_turns = max(rng.randint(args.min_turns, args.max_turns), 0)
        ending_pacing_floor = estimate_ending_pacing_floor(source_text, args.min_turns, args.max_turns)
        if ending_pacing_floor is not None and ending_pacing_floor > target_turns:
            target_turns = ending_pacing_floor

        persona_generation = record.get("persona_generation") or {}
        persona_content = persona_generation.get("controller_content") or {
            "persona_seed": record.get("persona_seed") or {}
        }
        if not isinstance(persona_content, dict):
            persona_content = {"persona_seed": record.get("persona_seed") or {}}

        persona_reasoning = persona_generation.get("controller_reasoning_content")
        persona_usage = persona_generation.get("usage") or {}
        persona_raw = persona_generation.get("raw_content") or ""
        public_timeline = list(record.get("public_timeline") or [])
        turns = list(record.get("turns") or [])
        usage_summary = record.get("usage") or {
            "persona_controller": persona_usage,
            "turn_controller": [],
            "actors": [],
            "actor_guard": [],
        }

        cache_signature = compute_signature(
            {
                "conversation_id": conversation_id,
                "conversation_index": conversation_index,
                "variation": variation,
                "seed": args.seed,
                "min_turns": args.min_turns,
                "max_turns": args.max_turns,
                "target_turns": target_turns,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "persona_thinking_enabled": persona_thinking_enabled,
                "turn_controller_thinking_enabled": turn_controller_thinking_enabled,
                "state_memory_tool_enabled": not args.disable_state_memory_tool,
                "actor_thinking_enabled": actor_thinking_enabled,
                "actor_guard_enabled": actor_guard_enabled,
                "actor_guard_model": args.actor_guard_model,
                "actor_guard_thinking_enabled": actor_guard_thinking_enabled,
                "controller_temperature": args.controller_temperature,
                "controller_top_p": args.controller_top_p,
                "persona_max_tokens": args.persona_max_tokens,
                "controller_max_tokens": args.controller_max_tokens,
                "actor_max_tokens": args.actor_max_tokens,
                "actor_guard_max_tokens": args.actor_guard_max_tokens,
                "persona_line_sha256": source_sha256,
                "persona_line_number": line_number,
                "prompts": {
                    "persona_controller_sha256": sha256_text(prompts.persona_controller),
                    "turn_controller_sha256": sha256_text(prompts.turn_controller),
                    "actor_sha256": sha256_text(prompts.actor),
                    "actor_guard_sha256": sha256_text(prompts.actor_guard),
                },
            }
        )

        existing_cache = load_turn_cache(cache_dir, conversation_id)
        if (
            isinstance(existing_cache, dict)
            and existing_cache.get("signature") == cache_signature
            and len(existing_cache.get("turns") or []) >= len(turns)
        ):
            pass
        else:
            save_turn_cache(
                cache_dir,
                conversation_id,
                {
                    "signature": cache_signature,
                    "conversation_id": conversation_id,
                    "conversation_index": conversation_index,
                    "variation": variation,
                    "target_turns": target_turns,
                    "persona_content": persona_content,
                    "persona_reasoning": persona_reasoning,
                    "persona_usage": persona_usage,
                    "persona_raw": persona_raw if persona_raw else "",
                    "public_timeline": public_timeline,
                    "turns": turns,
                    "usage_summary": usage_summary,
                    "early_end": True,
                    "saved_at": now_iso(),
                },
                backup_existing=backup_existing_cache,
            )
        backfilled_ids.append(conversation_id)

    return backfilled_ids, backfilled_done_indices


def persona_line_from_record(record: Dict[str, Any], persona_txt_path: str) -> PersonaLine:
    source = record.get("source") or {}
    text = source.get("text")
    if not isinstance(text, str):
        text = ""
    line_number = source.get("line_number")
    if not isinstance(line_number, int):
        line_number = 0
    sha = source.get("sha256")
    if not isinstance(sha, str) or not sha:
        sha = sha256_text(text)
    return PersonaLine(line_number=line_number, text=text, sha256=sha)


def persona_line_from_current_file(
    record: Dict[str, Any],
    persona_lines: List[PersonaLine],
    persona_txt_path: str,
) -> PersonaLine:
    source = record.get("source") or {}
    line_number = source.get("line_number")
    if not isinstance(line_number, int) or line_number <= 0:
        raise ValueError(
            f"record {record.get('id') or record.get('conversation_id')} has no valid source.line_number"
        )
    for item in persona_lines:
        if item.line_number == line_number:
            return item
    raise ValueError(
        f"source.line_number {line_number} for "
        f"{record.get('id') or record.get('conversation_id')} not found in {persona_txt_path}"
    )


def expand_id_args(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for part in value.split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out


def run_one_conversation_task(
    *,
    idx0: int,
    args: argparse.Namespace,
    prompts: PromptBundle,
    buffer: PersonaBuffer,
    pool_size: int,
    errors_out: str,
    persona_thinking_enabled: bool,
    turn_controller_thinking_enabled: bool,
    actor_thinking_enabled: bool,
    actor_guard_enabled: bool,
    actor_guard_thinking_enabled: bool,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    conversation_index = idx0 + 1

    # Honour pause / stop control before opening any expensive resources.
    wait_if_paused()
    if is_stopped():
        return {
            "ok": False,
            "idx0": idx0,
            "record": None,
            "error": {"error": "stopped"},
            "skipped": True,
        }

    persona_line, variation = pick_persona_line_for_index(
        idx0=idx0,
        args=args,
        buffer=buffer,
        pool_size=pool_size,
    )

    client = get_thread_client(args.base_url)

    try:
        record = generate_one_conversation(
            client=client,
            prompts=prompts,
            model=args.model,
            persona_txt_path=args.persona_txt,
            persona_line=persona_line,
            conversation_index=conversation_index,
            variation=variation,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            persona_thinking_enabled=persona_thinking_enabled,
            turn_controller_thinking_enabled=turn_controller_thinking_enabled,
            state_memory_tool_enabled=not args.disable_state_memory_tool,
            resume_accept_stale_cache=args.resume_accept_stale_cache,
            actor_thinking_enabled=actor_thinking_enabled,
            actor_guard_enabled=actor_guard_enabled,
            actor_guard_model=args.actor_guard_model,
            actor_guard_thinking_enabled=actor_guard_thinking_enabled,
            controller_temperature=args.controller_temperature,
            controller_top_p=args.controller_top_p,
            persona_max_tokens=args.persona_max_tokens,
            controller_max_tokens=args.controller_max_tokens,
            actor_max_tokens=args.actor_max_tokens,
            actor_guard_max_tokens=args.actor_guard_max_tokens,
            keep_raw_content=args.keep_raw_content,
            errors_out=errors_out,
            retries=args.retries,
            retry_base_sleep=args.retry_base_sleep,
            cache_dir=cache_dir,
            cache_diagnostics=args.resume,
            backup_existing_cache=not args.no_turn_cache_backup,
        )
        return {
            "ok": True,
            "idx0": idx0,
            "record": record,
            "error": None,
        }

    except Exception as e:
        err = {
            "created_at": now_iso(),
            "stage": "conversation",
            "conversation_index": conversation_index,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(limit=10),
            "source": {
                "line_number": persona_line.line_number,
                "text": persona_line.text,
                "sha256": persona_line.sha256,
                "variation": variation,
            },
        }
        append_jsonl(errors_out, err)
        progress_update(
            status="error",
            error=err,
            event={
                "type": "conversation_error",
                "conversation_index": conversation_index,
                "error": str(e),
            },
        )
        return {
            "ok": False,
            "idx0": idx0,
            "record": None,
            "error": err,
        }


def rewrite_one_conversation_task(
    *,
    record: Dict[str, Any],
    record_position: int,
    args: argparse.Namespace,
    prompts: PromptBundle,
    persona_lines: List[PersonaLine],
    errors_out: str,
    persona_thinking_enabled: bool,
    turn_controller_thinking_enabled: bool,
    actor_thinking_enabled: bool,
    actor_guard_enabled: bool,
    actor_guard_thinking_enabled: bool,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    conversation_id = str(record.get("id") or record.get("conversation_id") or "")
    conversation_index = conversation_index_from_id(conversation_id)
    if conversation_index is None:
        return {
            "ok": False,
            "record_position": record_position,
            "record": None,
            "error": {
                "error": f"cannot parse conversation index from id: {conversation_id!r}",
            },
        }

    source = record.get("source") or {}
    variation = source.get("variation")
    if not isinstance(variation, int):
        variation = (record.get("generation_config") or {}).get("variation")
    if not isinstance(variation, int):
        variation = 1

    if args.rewrite_use_current_persona_txt:
        persona_line = persona_line_from_current_file(record, persona_lines, args.persona_txt)
    else:
        persona_line = persona_line_from_record(record, args.persona_txt)
    client = get_thread_client(args.base_url)

    try:
        if cache_dir and args.delete_turn_cache_on_success:
            delete_turn_cache(cache_dir, conversation_id)
        rewritten = generate_one_conversation(
            client=client,
            prompts=prompts,
            model=args.model,
            persona_txt_path=args.persona_txt,
            persona_line=persona_line,
            conversation_index=conversation_index,
            variation=variation,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            persona_thinking_enabled=persona_thinking_enabled,
            turn_controller_thinking_enabled=turn_controller_thinking_enabled,
            state_memory_tool_enabled=not args.disable_state_memory_tool,
            resume_accept_stale_cache=args.resume_accept_stale_cache,
            actor_thinking_enabled=actor_thinking_enabled,
            actor_guard_enabled=actor_guard_enabled,
            actor_guard_model=args.actor_guard_model,
            actor_guard_thinking_enabled=actor_guard_thinking_enabled,
            controller_temperature=args.controller_temperature,
            controller_top_p=args.controller_top_p,
            persona_max_tokens=args.persona_max_tokens,
            controller_max_tokens=args.controller_max_tokens,
            actor_max_tokens=args.actor_max_tokens,
            actor_guard_max_tokens=args.actor_guard_max_tokens,
            keep_raw_content=args.keep_raw_content,
            errors_out=errors_out,
            retries=args.retries,
            retry_base_sleep=args.retry_base_sleep,
            cache_dir=cache_dir,
            backup_existing_cache=not args.no_turn_cache_backup,
            conversation_id_override=conversation_id,
        )
        return {
            "ok": True,
            "record_position": record_position,
            "record": rewritten,
            "error": None,
        }
    except Exception as e:
        err = {
            "created_at": now_iso(),
            "stage": "rewrite_conversation",
            "conversation_id": conversation_id,
            "record_position": record_position,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(limit=10),
        }
        append_jsonl(errors_out, err)
        progress_update(
            status="error",
            error=err,
            event={
                "type": "rewrite_conversation_error",
                "conversation_id": conversation_id,
                "error": str(e),
            },
        )
        return {
            "ok": False,
            "record_position": record_position,
            "record": None,
            "error": err,
        }


def finish_one_conversation_task(
    *,
    record: Dict[str, Any],
    record_position: int,
    args: argparse.Namespace,
    prompts: PromptBundle,
    errors_out: str,
    persona_thinking_enabled: bool,
    turn_controller_thinking_enabled: bool,
    actor_thinking_enabled: bool,
    actor_guard_enabled: bool,
    actor_guard_thinking_enabled: bool,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    conversation_id = str(record.get("id") or record.get("conversation_id") or "")
    conversation_index = conversation_index_from_id(conversation_id)
    if conversation_index is None:
        return {
            "ok": False,
            "record_position": record_position,
            "record": None,
            "error": {
                "error": f"cannot parse conversation index from id: {conversation_id!r}",
            },
        }

    source = record.get("source") or {}
    variation = source.get("variation")
    if not isinstance(variation, int):
        variation = (record.get("generation_config") or {}).get("variation")
    if not isinstance(variation, int):
        variation = 1

    persona_line = persona_line_from_record(record, args.persona_txt)
    client = get_thread_client(args.base_url)

    try:
        extended = generate_one_conversation(
            client=client,
            prompts=prompts,
            model=args.model,
            persona_txt_path=args.persona_txt,
            persona_line=persona_line,
            conversation_index=conversation_index,
            variation=variation,
            min_turns=args.finish_min_turns,
            max_turns=max(args.max_turns, args.finish_min_turns),
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            persona_thinking_enabled=persona_thinking_enabled,
            turn_controller_thinking_enabled=turn_controller_thinking_enabled,
            state_memory_tool_enabled=not args.disable_state_memory_tool,
            resume_accept_stale_cache=args.resume_accept_stale_cache,
            actor_thinking_enabled=actor_thinking_enabled,
            actor_guard_enabled=actor_guard_enabled,
            actor_guard_model=args.actor_guard_model,
            actor_guard_thinking_enabled=actor_guard_thinking_enabled,
            controller_temperature=args.controller_temperature,
            controller_top_p=args.controller_top_p,
            persona_max_tokens=args.persona_max_tokens,
            controller_max_tokens=args.controller_max_tokens,
            actor_max_tokens=args.actor_max_tokens,
            actor_guard_max_tokens=args.actor_guard_max_tokens,
            keep_raw_content=args.keep_raw_content,
            errors_out=errors_out,
            retries=args.retries,
            retry_base_sleep=args.retry_base_sleep,
            cache_dir=cache_dir,
            backup_existing_cache=not args.no_turn_cache_backup,
            existing_record=record,
            target_turns_override=args.finish_min_turns,
        )
        return {
            "ok": True,
            "record_position": record_position,
            "record": extended,
            "error": None,
        }
    except Exception as e:
        err = {
            "created_at": now_iso(),
            "stage": "finish_conversation",
            "conversation_id": conversation_id,
            "record_position": record_position,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(limit=10),
        }
        append_jsonl(errors_out, err)
        progress_update(
            status="error",
            error=err,
            event={
                "type": "finish_conversation_error",
                "conversation_id": conversation_id,
                "error": str(e),
            },
        )
        return {
            "ok": False,
            "record_position": record_position,
            "record": None,
            "error": err,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona-controlled Japanese multi-agent dialogue JSONL with DeepSeek."
    )

    parser.add_argument("--persona-txt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-dir", default="./prompts")
    parser.add_argument("--errors-out", default=None)

    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--flash",
        action="store_true",
        help=(
            f"Shortcut for --model {FLASH_MODEL}. If --auto-generate-situations "
            "is used and --situation-model is omitted, the producer also uses flash."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)

    parser.add_argument("--num-conversations", type=int, default=None)
    parser.add_argument("--variations-per-line", type=int, default=1)
    parser.add_argument("--sampling", choices=["round_robin", "random"], default="round_robin")

    parser.add_argument("--min-turns", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--rewrite-id",
        action="append",
        default=[],
        help=(
            "Regenerate existing record(s) in --out by id, preserving the same id. "
            "May be repeated or given as comma-separated ids."
        ),
    )
    parser.add_argument(
        "--rewrite-ids-file",
        default=None,
        help="Optional file containing one id per line for --rewrite-id mode.",
    )
    parser.add_argument(
        "--rewrite-dry-run",
        action="store_true",
        help="Only report records matching --rewrite-id / --rewrite-ids-file; do not call the API or rewrite the jsonl.",
    )
    parser.add_argument(
        "--rewrite-use-current-persona-txt",
        action="store_true",
        help=(
            "In rewrite mode, use the current line text from --persona-txt by "
            "source.line_number instead of the saved source.text in the jsonl."
        ),
    )
    parser.add_argument(
        "--rewrite-remove-duplicates",
        action="store_true",
        help=(
            "In rewrite mode, if an id appears multiple times, remove all "
            "existing records for that id and regenerate exactly one replacement."
        ),
    )
    parser.add_argument(
        "--rewrite-all-duplicates",
        action="store_true",
        help=(
            "Detect every duplicated id in --out and regenerate exactly one "
            "replacement per duplicated id. Implies --rewrite-remove-duplicates."
        ),
    )
    parser.add_argument(
        "--finish-min-turns",
        type=int,
        default=0,
        help=(
            "Finish existing records in --out whose actual_turns are below this "
            "count. Matching conversations are continued with the same id and "
            "rewritten in-place in the jsonl instead of creating new ids. "
            "0 disables finish mode."
        ),
    )
    parser.add_argument(
        "--finish-dry-run",
        action="store_true",
        help=(
            "With --finish-min-turns, only detect and report matching short "
            "records. Do not call the API and do not rewrite the jsonl."
        ),
    )
    parser.add_argument(
        "--finish-dry-run-format",
        choices=["json", "lines", "ids"],
        default="json",
        help=(
            "Output format for --finish-dry-run. json = one summary object, "
            "lines = one detected record per line with details, ids = one id per line."
        ),
    )

    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument(
        "--thinking",
        choices=["default", "on", "off"],
        default="default",
        help=(
            "Global thinking mode for persona, turn controller, and actors. "
            "default keeps the legacy defaults: persona/actor on, turn controller off. "
            "Per-agent thinking flags below override this value."
        ),
    )

    parser.add_argument("--disable-persona-thinking", action="store_true")
    parser.add_argument("--enable-turn-controller-thinking", action="store_true")
    parser.add_argument("--disable-actor-thinking", action="store_true")
    parser.add_argument(
        "--actor-guard",
        action="store_true",
        help=(
            "After each actor turn, use a third-person DeepSeek judge to "
            "explain age/body/tone inconsistencies back to the actor and rewrite the turn."
        ),
    )
    parser.add_argument("--actor-guard-model", default=PRO_MODEL)
    parser.add_argument(
        "--actor-guard-thinking",
        choices=["default", "on", "off"],
        default="off",
        help="Thinking mode for --actor-guard. default/off keeps guard judging cheaper.",
    )

    parser.add_argument("--controller-temperature", type=float, default=0.9)
    parser.add_argument("--controller-top-p", type=float, default=0.95)

    parser.add_argument(
        "--persona-max-tokens",
        type=int,
        default=DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
        help="Defaults to DeepSeek V4 max output (384K). 0 means omit max_tokens for persona controller.",
    )
    parser.add_argument(
        "--controller-max-tokens",
        type=int,
        default=DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
        help="Defaults to DeepSeek V4 max output (384K). 0 means omit max_tokens for turn controller.",
    )
    parser.add_argument(
        "--actor-max-tokens",
        type=int,
        default=DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
        help="Defaults to DeepSeek V4 max output (384K). 0 means omit max_tokens for actor.",
    )
    parser.add_argument(
        "--actor-guard-max-tokens",
        type=int,
        default=DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
        help="Defaults to DeepSeek V4 max output (384K). 0 means omit max_tokens for actor guard.",
    )

    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)

    parser.add_argument("--keep-raw-content", action="store_true")
    parser.add_argument(
        "--turn-cache-dir",
        type=str,
        default="",
        help=(
            "Directory for per-turn conversation caches. Empty (default) means "
            "<out>.cache. Each in-flight conversation writes a snapshot after "
            "every successful turn so --resume can continue mid-conversation. "
            "Caches are kept after success unless --delete-turn-cache-on-success is set."
        ),
    )
    parser.add_argument(
        "--no-turn-cache",
        action="store_true",
        help="Disable per-turn caching (in-flight conversations are restarted from turn 1).",
    )
    parser.add_argument(
        "--delete-turn-cache-on-success",
        action="store_true",
        help="Delete a per-turn cache file after its conversation is successfully written to out.jsonl.",
    )
    parser.add_argument(
        "--no-turn-cache-backup",
        action="store_true",
        help="Disable timestamped backup of an existing per-turn cache file before it is overwritten.",
    )
    parser.add_argument(
        "--disable-state-memory-tool",
        action="store_true",
        help=(
            "Disable the new Turn Controller strict tool/state_memory path and use the legacy JSON controller path. "
            "Useful when resuming old caches."
        ),
    )
    parser.add_argument(
        "--resume-accept-stale-cache",
        action="store_true",
        help=(
            "With --resume, accept an existing turn cache even when its signature differs. "
            "Use only for explicitly restoring old backup caches after prompt/schema changes."
        ),
    )
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--progress-server", action="store_true")
    parser.add_argument("--progress-host", default="127.0.0.1")
    parser.add_argument("--progress-port", type=int, default=8765)

    # ---- Auto situation generation (background DeepSeek flash producer) ----
    parser.add_argument(
        "--auto-generate-situations",
        action="store_true",
        help=(
            "While dialogues are being generated, run a background producer "
            "that grows --persona-txt with new situations using DeepSeek flash. "
            "Workers needing a situation that doesn't exist yet will block "
            "until the producer appends one."
        ),
    )
    parser.add_argument(
        "--situation-batch-size",
        type=int,
        default=8,
        help="Situations per producer API call.",
    )
    parser.add_argument(
        "--situation-target",
        type=int,
        default=0,
        help=(
            "Stop the producer once format.txt has this many lines (after dedup). "
            "0 = no fixed cap; the producer keeps generating in parallel until "
            "either dialogue generation finishes or --situation-max-iterations is hit."
        ),
    )
    parser.add_argument(
        "--situation-max-iterations",
        type=int,
        default=200,
        help="Hard cap on producer iterations (safety net).",
    )
    parser.add_argument(
        "--situation-model",
        default=None,
        help="Model used by the background situation producer.",
    )
    parser.add_argument(
        "--situation-prompt-file",
        default=None,
        help=(
            "Prompt file for the background situation producer. "
            "Defaults to <prompt-dir>/situation_gen.txt."
        ),
    )
    parser.add_argument("--situation-temperature", type=float, default=1.1)
    parser.add_argument("--situation-top-p", type=float, default=0.95)
    parser.add_argument("--situation-max-tokens", type=int, default=DEEPSEEK_V4_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--situation-seed",
        action="append",
        default=[],
        help="Extra seed situation for the producer (repeatable).",
    )
    parser.add_argument(
        "--situation-seed-file",
        default=None,
        help="Optional file: one seed situation per line.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model is None:
        args.model = FLASH_MODEL if args.flash else DEFAULT_MODEL

    if args.flash and args.model != FLASH_MODEL:
        raise ValueError("--flash cannot be combined with --model other than deepseek-v4-flash")

    if args.situation_model is None:
        args.situation_model = FLASH_MODEL if args.flash else SITUATION_GEN_MODEL_DEFAULT
    if args.situation_prompt_file is None:
        args.situation_prompt_file = os.path.join(args.prompt_dir, "situation_gen.txt")

    if args.min_turns <= 0:
        raise ValueError("--min-turns must be positive")

    if args.max_turns < args.min_turns:
        raise ValueError("--max-turns must be >= --min-turns")

    if args.variations_per_line <= 0:
        raise ValueError("--variations-per-line must be positive")

    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    if args.finish_min_turns < 0:
        raise ValueError("--finish-min-turns must be >= 0")

    if args.resume_accept_stale_cache and not args.resume:
        raise ValueError("--resume-accept-stale-cache requires --resume")

    if args.resume_accept_stale_cache and not args.disable_state_memory_tool:
        raise ValueError("--resume-accept-stale-cache requires --disable-state-memory-tool")

    persona_lines = load_persona_lines(args.persona_txt)
    prompts = load_prompts(args.prompt_dir)
    if args.actor_guard and not prompts.actor_guard.strip():
        raise FileNotFoundError(
            f"--actor-guard requires {os.path.join(args.prompt_dir, 'actor_guard.txt')}"
        )

    initial_pool = len(persona_lines)
    auto_gen_active = bool(args.auto_generate_situations)

    if args.num_conversations is not None:
        total_requested = args.num_conversations
    elif auto_gen_active and args.situation_target > 0:
        # Producer will grow format.txt to situation_target lines; dispatch
        # one batch of dialogues per line, multiplied by variations.
        total_requested = args.situation_target * args.variations_per_line
    else:
        total_requested = initial_pool * args.variations_per_line

    needed_situations = max(
        initial_pool, math.ceil(total_requested / args.variations_per_line)
    )

    if (
        not args.auto_generate_situations
        and needed_situations > initial_pool
    ):
        capped_total = initial_pool * args.variations_per_line
        print(
            json.dumps(
                {
                    "event": "capped_total_to_pool",
                    "persona_lines": initial_pool,
                    "variations_per_line": args.variations_per_line,
                    "requested": total_requested,
                    "capped_to": capped_total,
                    "hint": "pass --auto-generate-situations to grow format.txt on the fly",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        total_requested = capped_total
        needed_situations = initial_pool

    persona_buffer = PersonaBuffer(initial=persona_lines)
    pool_size_for_indexing = needed_situations
    register_persona_buffer(persona_buffer, args.persona_txt, initial_pool)

    errors_out = args.errors_out or args.out + ".errors.jsonl"
    done_indices: set[int] = set()
    already_done = 0
    if args.resume:
        done_indices = read_done_indices(args.out)
        already_done = len(done_indices)

    if args.no_turn_cache:
        cache_dir: Optional[str] = None
    else:
        cache_dir = args.turn_cache_dir or (args.out + ".cache")
        ensure_cache_dir(cache_dir)

    if args.thinking == "on":
        persona_thinking_enabled = True
        turn_controller_thinking_enabled = True
        actor_thinking_enabled = True
    elif args.thinking == "off":
        persona_thinking_enabled = False
        turn_controller_thinking_enabled = False
        actor_thinking_enabled = False
    else:
        persona_thinking_enabled = True
        turn_controller_thinking_enabled = False
        actor_thinking_enabled = True

    if args.disable_persona_thinking:
        persona_thinking_enabled = False
    if args.enable_turn_controller_thinking:
        turn_controller_thinking_enabled = True
    if args.disable_actor_thinking:
        actor_thinking_enabled = False

    if args.actor_guard_thinking == "on":
        actor_guard_thinking_enabled = True
    else:
        actor_guard_thinking_enabled = False

    resume_backfilled_done_indices: set[int] = set()
    if args.resume and args.finish_min_turns > 0:
        if args.min_turns < args.finish_min_turns:
            args.min_turns = args.finish_min_turns
        if args.max_turns < args.finish_min_turns:
            args.max_turns = args.finish_min_turns
        if cache_dir is not None:
            records = read_jsonl_records(args.out)
            backfilled_ids, resume_backfilled_done_indices = backfill_short_records_to_cache(
                records=records,
                cache_dir=cache_dir,
                args=args,
                prompts=prompts,
                persona_thinking_enabled=persona_thinking_enabled,
                turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                actor_thinking_enabled=actor_thinking_enabled,
                actor_guard_enabled=args.actor_guard,
                actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                backup_existing_cache=not args.no_turn_cache_backup,
            )
            if backfilled_ids:
                print(
                    json.dumps(
                        {
                            "event": "resume_cache_backfill",
                            "out": args.out,
                            "backfilled": len(backfilled_ids),
                            "ids": backfilled_ids,
                            "finish_min_turns": args.finish_min_turns,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    work_indices: List[int] = [
        i for i in range(total_requested) if i not in done_indices or i in resume_backfilled_done_indices
    ]

    rewrite_ids = expand_id_args(args.rewrite_id)
    if args.rewrite_ids_file:
        with open(args.rewrite_ids_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    rewrite_ids.append(s)

    rewrite_mode_requested = bool(rewrite_ids) or bool(args.rewrite_all_duplicates)
    if rewrite_mode_requested:
        records = read_jsonl_records(args.out)
        duplicate_positions = find_duplicate_record_ids(records)
        if args.rewrite_all_duplicates:
            rewrite_ids.extend(sorted(duplicate_positions.keys()))
            args.rewrite_remove_duplicates = True

        requested_ids = set(rewrite_ids)
        if args.rewrite_remove_duplicates:
            targets = []
            for cid in rewrite_ids:
                pos_list = duplicate_positions.get(cid)
                if pos_list:
                    first_pos = pos_list[0]
                    targets.append((first_pos, records[first_pos]))
                else:
                    for pos, rec in enumerate(records):
                        if (rec.get("id") or rec.get("conversation_id")) == cid:
                            targets.append((pos, rec))
                            break
        else:
            targets = [
                (pos, rec)
                for pos, rec in enumerate(records)
                if (rec.get("id") or rec.get("conversation_id")) in requested_ids
            ]
        found_ids = {
            str(rec.get("id") or rec.get("conversation_id"))
            for _, rec in targets
        }
        missing_ids = sorted(requested_ids - found_ids)

        if args.rewrite_dry_run:
            for pos, rec in targets:
                print(
                    json.dumps(
                        {
                            "id": rec.get("id") or rec.get("conversation_id"),
                            "record_position": pos,
                            "actual_turns": actual_turn_count(rec),
                            "target_turns": (rec.get("generation_config") or {}).get("target_turns"),
                            "source_line_number": (rec.get("source") or {}).get("line_number"),
                            "variation": (rec.get("source") or {}).get("variation"),
                            "duplicate_positions": duplicate_positions.get(
                                str(rec.get("id") or rec.get("conversation_id")),
                                [],
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "event": "rewrite_dry_run_summary",
                        "out": args.out,
                        "requested": len(requested_ids),
                        "targets": len(targets),
                        "missing_ids": missing_ids,
                        "rewrite_remove_duplicates": args.rewrite_remove_duplicates,
                        "rewrite_use_current_persona_txt": args.rewrite_use_current_persona_txt,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            return

        if not targets:
            print(
                json.dumps(
                    {
                        "event": "rewrite_nothing_to_do",
                        "out": args.out,
                        "requested": len(requested_ids),
                        "missing_ids": missing_ids,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            return

        if args.progress_server:
            start_progress_server(args.progress_host, args.progress_port)
            url = f"http://{args.progress_host}:{args.progress_port}"
            print(
                json.dumps(
                    {
                        "event": "progress_server_started",
                        "url": url,
                        "urls": [url],
                        "bind": args.progress_host,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

        start_event = {
            "event": "rewrite_start",
            "out": args.out,
            "errors_out": errors_out,
            "requested": len(requested_ids),
            "targets": len(targets),
            "missing_ids": missing_ids,
            "workers": args.workers,
            "model": args.model,
        }
        print(json.dumps(start_event, ensure_ascii=False), file=sys.stderr, flush=True)
        progress_update(
            status="rewrite_started",
            summary={
                "written": 0,
                "total_requested": len(targets),
                "workers": args.workers,
                "out": args.out,
            },
            event=start_event,
        )

        replacements: Dict[int, Dict[str, Any]] = {}
        rewritten_count = 0

        if args.workers <= 1:
            get_thread_client(args.base_url)
            with tqdm(total=len(targets)) as pbar:
                for pos, rec in targets:
                    if is_stopped():
                        break
                    wait_if_paused()
                    result = rewrite_one_conversation_task(
                        record=rec,
                        record_position=pos,
                        args=args,
                        prompts=prompts,
                        persona_lines=persona_lines,
                        errors_out=errors_out,
                        persona_thinking_enabled=persona_thinking_enabled,
                        turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                        actor_thinking_enabled=actor_thinking_enabled,
                        actor_guard_enabled=args.actor_guard,
                        actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                        cache_dir=cache_dir,
                    )
                    if result["ok"]:
                        replacements[pos] = result["record"]
                        rewritten_count += 1
                        if cache_dir and args.delete_turn_cache_on_success:
                            delete_turn_cache(cache_dir, result["record"]["id"])
                    else:
                        print(
                            json.dumps(
                                {
                                    "event": "rewrite_task_failed",
                                    "record_position": result["record_position"],
                                    "error": result["error"]["error"],
                                },
                                ensure_ascii=False,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    progress_update(
                        summary={
                            "written": rewritten_count,
                            "total_requested": len(targets),
                            "workers": args.workers,
                            "out": args.out,
                        }
                    )
                    pbar.update(1)
                    if args.sleep > 0:
                        time.sleep(args.sleep)
        else:
            with tqdm(total=len(targets)) as pbar:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            rewrite_one_conversation_task,
                            record=rec,
                            record_position=pos,
                            args=args,
                            prompts=prompts,
                            persona_lines=persona_lines,
                            errors_out=errors_out,
                            persona_thinking_enabled=persona_thinking_enabled,
                            turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                            actor_thinking_enabled=actor_thinking_enabled,
                            actor_guard_enabled=args.actor_guard,
                            actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                            cache_dir=cache_dir,
                        )
                        for pos, rec in targets
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        if result["ok"]:
                            replacements[result["record_position"]] = result["record"]
                            rewritten_count += 1
                            if cache_dir and args.delete_turn_cache_on_success:
                                delete_turn_cache(cache_dir, result["record"]["id"])
                        else:
                            print(
                                json.dumps(
                                    {
                                        "event": "rewrite_task_failed",
                                        "record_position": result["record_position"],
                                        "error": result["error"]["error"],
                                    },
                                    ensure_ascii=False,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                        progress_update(
                            summary={
                                "written": rewritten_count,
                                "total_requested": len(targets),
                                "workers": args.workers,
                                "out": args.out,
                            }
                        )
                        pbar.update(1)
                        if args.sleep > 0:
                            time.sleep(args.sleep)

        if args.rewrite_remove_duplicates:
            replacement_by_id = {
                str(rec.get("id") or rec.get("conversation_id")): rec
                for rec in replacements.values()
            }
            new_records: List[Dict[str, Any]] = []
            inserted_ids: set[str] = set()
            for rec in records:
                cid = rec.get("id") or rec.get("conversation_id")
                if cid in replacement_by_id:
                    if cid not in inserted_ids:
                        new_records.append(replacement_by_id[str(cid)])
                        inserted_ids.add(str(cid))
                    continue
                new_records.append(rec)
            records = new_records
        else:
            for pos, rec in replacements.items():
                records[pos] = rec
        rewrite_jsonl_records(args.out, records)
        sort_jsonl_by_conversation_id(args.out)

        done_event = {
            "event": "rewrite_done",
            "rewritten": rewritten_count,
            "targets": len(targets),
            "missing_ids": missing_ids,
            "out": args.out,
            "errors_out": errors_out,
        }
        print(json.dumps(done_event, ensure_ascii=False), file=sys.stderr, flush=True)
        progress_update(
            status="rewrite_done",
            summary={
                "written": rewritten_count,
                "total_requested": len(targets),
                "workers": args.workers,
                "out": args.out,
            },
            event=done_event,
        )
        return

    if args.finish_min_turns > 0 and not args.resume:
        records = read_jsonl_records(args.out)
        targets = [
            (pos, rec)
            for pos, rec in enumerate(records)
            if actual_turn_count(rec) < args.finish_min_turns
        ]

        if args.finish_dry_run:
            report_items = []
            for pos, rec in targets:
                cfg = rec.get("generation_config") or {}
                report_items.append(
                    {
                        "record_position": pos,
                        "id": rec.get("id") or rec.get("conversation_id"),
                        "actual_turns": actual_turn_count(rec),
                        "target_turns": cfg.get("target_turns"),
                        "source_line_number": (rec.get("source") or {}).get("line_number"),
                        "variation": (rec.get("source") or {}).get("variation"),
                    }
                )
            if args.finish_dry_run_format == "ids":
                for item in report_items:
                    print(item["id"], flush=True)
                print(
                    json.dumps(
                        {
                            "event": "finish_dry_run_summary",
                            "out": args.out,
                            "finish_min_turns": args.finish_min_turns,
                            "records": len(records),
                            "targets": len(targets),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            elif args.finish_dry_run_format == "lines":
                for item in report_items:
                    print(
                        json.dumps(
                            {
                                "id": item["id"],
                                "actual_turns": item["actual_turns"],
                                "target_turns": item["target_turns"],
                                "record_position": item["record_position"],
                                "source_line_number": item["source_line_number"],
                                "variation": item["variation"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                print(
                    json.dumps(
                        {
                            "event": "finish_dry_run_summary",
                            "out": args.out,
                            "finish_min_turns": args.finish_min_turns,
                            "records": len(records),
                            "targets": len(targets),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    json.dumps(
                        {
                            "event": "finish_dry_run",
                            "out": args.out,
                            "finish_min_turns": args.finish_min_turns,
                            "records": len(records),
                            "targets": len(targets),
                            "items": report_items,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return

        if args.progress_server:
            start_progress_server(args.progress_host, args.progress_port)
            url = f"http://{args.progress_host}:{args.progress_port}"
            print(
                json.dumps(
                    {
                        "event": "progress_server_started",
                        "url": url,
                        "urls": [url],
                        "bind": args.progress_host,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

        start_event = {
            "event": "finish_start",
            "out": args.out,
            "errors_out": errors_out,
            "finish_min_turns": args.finish_min_turns,
            "records": len(records),
            "targets": len(targets),
            "workers": args.workers,
            "model": args.model,
        }
        print(json.dumps(start_event, ensure_ascii=False), file=sys.stderr, flush=True)
        progress_update(
            status="finish_started",
            summary={
                "written": 0,
                "total_requested": len(targets),
                "workers": args.workers,
                "out": args.out,
            },
            event=start_event,
        )

        if not targets:
            print(
                json.dumps(
                    {
                        "event": "finish_nothing_to_do",
                        "finish_min_turns": args.finish_min_turns,
                        "records": len(records),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            return

        replacements: Dict[int, Dict[str, Any]] = {}
        finished = 0

        if args.workers <= 1:
            get_thread_client(args.base_url)
            with tqdm(total=len(targets)) as pbar:
                for pos, rec in targets:
                    if is_stopped():
                        break
                    wait_if_paused()
                    result = finish_one_conversation_task(
                        record=rec,
                        record_position=pos,
                        args=args,
                        prompts=prompts,
                        errors_out=errors_out,
                        persona_thinking_enabled=persona_thinking_enabled,
                        turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                        actor_thinking_enabled=actor_thinking_enabled,
                        actor_guard_enabled=args.actor_guard,
                        actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                        cache_dir=cache_dir,
                    )
                    if result["ok"]:
                        replacements[pos] = result["record"]
                        finished += 1
                        if cache_dir and args.delete_turn_cache_on_success:
                            delete_turn_cache(cache_dir, result["record"]["id"])
                    else:
                        print(
                            json.dumps(
                                {
                                    "event": "finish_task_failed",
                                    "record_position": result["record_position"],
                                    "error": result["error"]["error"],
                                },
                                ensure_ascii=False,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    progress_update(
                        summary={
                            "written": finished,
                            "total_requested": len(targets),
                            "workers": args.workers,
                            "out": args.out,
                        }
                    )
                    pbar.update(1)
                    if args.sleep > 0:
                        time.sleep(args.sleep)
        else:
            with tqdm(total=len(targets)) as pbar:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            finish_one_conversation_task,
                            record=rec,
                            record_position=pos,
                            args=args,
                            prompts=prompts,
                            errors_out=errors_out,
                            persona_thinking_enabled=persona_thinking_enabled,
                            turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                            actor_thinking_enabled=actor_thinking_enabled,
                            actor_guard_enabled=args.actor_guard,
                            actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                            cache_dir=cache_dir,
                        )
                        for pos, rec in targets
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        if result["ok"]:
                            replacements[result["record_position"]] = result["record"]
                            finished += 1
                            if cache_dir and args.delete_turn_cache_on_success:
                                delete_turn_cache(cache_dir, result["record"]["id"])
                        else:
                            print(
                                json.dumps(
                                    {
                                        "event": "finish_task_failed",
                                        "record_position": result["record_position"],
                                        "error": result["error"]["error"],
                                    },
                                    ensure_ascii=False,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                        progress_update(
                            summary={
                                "written": finished,
                                "total_requested": len(targets),
                                "workers": args.workers,
                                "out": args.out,
                            }
                        )
                        pbar.update(1)
                        if args.sleep > 0:
                            time.sleep(args.sleep)

        for pos, rec in replacements.items():
            records[pos] = rec
        rewrite_jsonl_records(args.out, records)
        sort_jsonl_by_conversation_id(args.out)

        done_event = {
            "event": "finish_done",
            "finished": finished,
            "targets": len(targets),
            "out": args.out,
            "errors_out": errors_out,
        }
        print(json.dumps(done_event, ensure_ascii=False), file=sys.stderr, flush=True)
        progress_update(
            status="finish_done",
            summary={
                "written": finished,
                "total_requested": len(targets),
                "workers": args.workers,
                "out": args.out,
            },
            event=done_event,
        )
        return

    if not work_indices:
        print(
            f"nothing to do: work_indices=0 (done={len(done_indices)}, backfilled={len(resume_backfilled_done_indices)}, total={total_requested})",
            file=sys.stderr,
        )
        return

    producer_stop_event: Optional[threading.Event] = None
    producer_thread: Optional[threading.Thread] = None

    if auto_gen_active:
        seeds: List[str] = list(args.situation_seed or [])
        if args.situation_seed_file:
            from .io_utils import read_text as _read_text
            for line in _read_text(args.situation_seed_file).splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    seeds.append(s)
        if not seeds:
            seeds = [pl.text for pl in persona_lines[:8]]

        producer_stop_event = threading.Event()

        # If user gave an explicit target, honour it; otherwise let the
        # producer run alongside dialogue generation until we set the
        # stop event after all conversations finish.
        target_count = (
            args.situation_target if args.situation_target > 0 else None
        )

        producer_thread = start_background_producer(
            buffer=persona_buffer,
            out_path=args.persona_txt,
            prompt_file=args.situation_prompt_file,
            seeds=seeds,
            target_count=target_count,
            stop_event=producer_stop_event,
            batch_size=args.situation_batch_size,
            max_iterations=args.situation_max_iterations,
            model=args.situation_model,
            base_url=args.base_url,
            temperature=args.situation_temperature,
            top_p=args.situation_top_p,
            max_tokens=(
                args.situation_max_tokens if args.situation_max_tokens > 0 else None
            ),
            retries=args.retries,
            retry_base_sleep=args.retry_base_sleep,
            seed=args.seed,
        )
    else:
        # No producer: persona_buffer is fixed; mark it finished so callers
        # asking for an out-of-range index (which shouldn't happen) fail fast.
        persona_buffer.mark_finished()

    if args.progress_server:
        start_progress_server(args.progress_host, args.progress_port)
        urls: list[str] = []
        if args.progress_host in ("0.0.0.0", ""):
            urls.append(f"http://127.0.0.1:{args.progress_port}")
            try:
                import socket as _socket

                hostname = _socket.gethostname()
                lan_ip = _socket.gethostbyname(hostname)
                if lan_ip and lan_ip != "127.0.0.1":
                    urls.append(f"http://{lan_ip}:{args.progress_port}")
                if hostname:
                    urls.append(f"http://{hostname}:{args.progress_port}")
            except Exception:
                pass
        else:
            urls.append(f"http://{args.progress_host}:{args.progress_port}")
        print(
            json.dumps(
                {
                    "event": "progress_server_started",
                    "url": urls[0],
                    "urls": urls,
                    "bind": args.progress_host,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    start_event = {
        "event": "start",
        "out": args.out,
        "errors_out": errors_out,
        "persona_txt": args.persona_txt,
        "prompt_dir": args.prompt_dir,
        "persona_lines": initial_pool,
        "needed_situations": needed_situations,
        "auto_generate_situations": auto_gen_active,
        "total_requested": total_requested,
        "already_done": already_done,
        "work_items": len(work_indices),
        "workers": args.workers,
        "model": args.model,
        "persona_thinking_enabled": persona_thinking_enabled,
        "turn_controller_thinking_enabled": turn_controller_thinking_enabled,
        "state_memory_tool_enabled": not args.disable_state_memory_tool,
        "resume_accept_stale_cache": args.resume_accept_stale_cache,
        "actor_thinking_enabled": actor_thinking_enabled,
        "reasoning_effort": args.reasoning_effort,
        "python_quality_filtering": False,
        "resume_backfilled": len(resume_backfilled_done_indices),
        "max_tokens_policy": {
            "persona_max_tokens": args.persona_max_tokens,
            "controller_max_tokens": args.controller_max_tokens,
            "actor_max_tokens": args.actor_max_tokens,
            "zero_means_omitted": True,
        },
        "prompt_hashes": {
            "persona_controller_sha256": sha256_text(prompts.persona_controller),
            "turn_controller_sha256": sha256_text(prompts.turn_controller),
            "actor_sha256": sha256_text(prompts.actor),
        },
    }

    print(json.dumps(start_event, ensure_ascii=False), file=sys.stderr, flush=True)

    progress_update(
        status="started",
        summary={
            "written": already_done,
            "total_requested": total_requested,
            "work_items": len(work_indices),
            "workers": args.workers,
            "out": args.out,
        },
        event=start_event,
    )

    written = already_done

    if args.workers <= 1:
        get_thread_client(args.base_url)

        with tqdm(total=len(work_indices)) as pbar:
            for idx0 in work_indices:
                if is_stopped():
                    break
                result = run_one_conversation_task(
                    idx0=idx0,
                    args=args,
                    prompts=prompts,
                    buffer=persona_buffer,
                    pool_size=pool_size_for_indexing,
                    errors_out=errors_out,
                    persona_thinking_enabled=persona_thinking_enabled,
                    turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                    actor_thinking_enabled=actor_thinking_enabled,
                    actor_guard_enabled=args.actor_guard,
                    actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                    cache_dir=cache_dir,
                )

                if result.get("skipped"):
                    pbar.update(1)
                    continue

                if result["ok"]:
                    if result.get("idx0") in resume_backfilled_done_indices:
                        replace_jsonl_record(args.out, result["record"])
                    else:
                        append_jsonl(args.out, result["record"])
                        written += 1
                    if cache_dir and args.delete_turn_cache_on_success:
                        delete_turn_cache(cache_dir, result["record"]["id"])

                progress_update(
                    summary={
                        "written": written,
                        "total_requested": total_requested,
                        "workers": args.workers,
                        "out": args.out,
                    }
                )

                pbar.update(1)

                if args.sleep > 0:
                    time.sleep(args.sleep)

    else:
        print(
            json.dumps(
                {
                    "event": "parallel_start",
                    "workers": args.workers,
                    "tasks": len(work_indices),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

        with tqdm(total=len(work_indices)) as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        run_one_conversation_task,
                        idx0=idx0,
                        args=args,
                        prompts=prompts,
                        buffer=persona_buffer,
                        pool_size=pool_size_for_indexing,
                        errors_out=errors_out,
                        persona_thinking_enabled=persona_thinking_enabled,
                        turn_controller_thinking_enabled=turn_controller_thinking_enabled,
                        actor_thinking_enabled=actor_thinking_enabled,
                        actor_guard_enabled=args.actor_guard,
                        actor_guard_thinking_enabled=actor_guard_thinking_enabled,
                        cache_dir=cache_dir,
                    )
                    for idx0 in work_indices
                ]

                for future in as_completed(futures):
                    result = future.result()

                    if result.get("skipped"):
                        pbar.update(1)
                        continue

                    if result["ok"]:
                        if result.get("idx0") in resume_backfilled_done_indices:
                            replace_jsonl_record(args.out, result["record"])
                        else:
                            append_jsonl(args.out, result["record"])
                            written += 1
                        if cache_dir and args.delete_turn_cache_on_success:
                            delete_turn_cache(cache_dir, result["record"]["id"])
                    else:
                        print(
                            json.dumps(
                                {
                                    "event": "task_failed",
                                    "idx0": result["idx0"],
                                    "error": result["error"]["error"],
                                },
                                ensure_ascii=False,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )

                    progress_update(
                        summary={
                            "written": written,
                            "total_requested": total_requested,
                            "workers": args.workers,
                            "out": args.out,
                        }
                    )

                    pbar.update(1)

                    if args.sleep > 0:
                        time.sleep(args.sleep)

    if producer_stop_event is not None:
        producer_stop_event.set()
    if producer_thread is not None:
        producer_thread.join(timeout=120)

    sort_jsonl_by_conversation_id(args.out)

    done_event = {
        "event": "done",
        "written_total": written,
        "out": args.out,
        "errors_out": errors_out,
    }

    print(json.dumps(done_event, ensure_ascii=False), file=sys.stderr, flush=True)

    progress_update(
        status="done",
        summary={
            "written": written,
            "total_requested": total_requested,
            "workers": args.workers,
            "out": args.out,
        },
        event=done_event,
    )
