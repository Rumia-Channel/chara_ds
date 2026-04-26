"""CLI entrypoint, work scheduling, and per-task wrapper."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from .api_client import get_thread_client
from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, PersonaLine, PromptBundle
from .conversation import generate_one_conversation
from .io_utils import (
    append_jsonl,
    count_jsonl_lines,
    load_persona_lines,
    load_prompts,
    now_iso,
    sha256_text,
)
from .persona_buffer import PersonaBuffer
from .progress import progress_update, start_progress_server
from .situation_gen import SITUATION_GEN_MODEL_DEFAULT
from .situation_producer import start_background_producer


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
) -> Dict[str, Any]:
    conversation_index = idx0 + 1

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
            actor_thinking_enabled=actor_thinking_enabled,
            controller_temperature=args.controller_temperature,
            controller_top_p=args.controller_top_p,
            persona_max_tokens=args.persona_max_tokens,
            controller_max_tokens=args.controller_max_tokens,
            actor_max_tokens=args.actor_max_tokens,
            keep_raw_content=args.keep_raw_content,
            errors_out=errors_out,
            retries=args.retries,
            retry_base_sleep=args.retry_base_sleep,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona-controlled Japanese multi-agent dialogue JSONL with DeepSeek."
    )

    parser.add_argument("--persona-txt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-dir", default="./prompts")
    parser.add_argument("--errors-out", default=None)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)

    parser.add_argument("--num-conversations", type=int, default=None)
    parser.add_argument("--variations-per-line", type=int, default=1)
    parser.add_argument("--sampling", choices=["round_robin", "random"], default="round_robin")

    parser.add_argument("--min-turns", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")

    parser.add_argument("--disable-persona-thinking", action="store_true")
    parser.add_argument("--enable-turn-controller-thinking", action="store_true")
    parser.add_argument("--disable-actor-thinking", action="store_true")

    parser.add_argument("--controller-temperature", type=float, default=0.9)
    parser.add_argument("--controller-top-p", type=float, default=0.95)

    parser.add_argument(
        "--persona-max-tokens",
        type=int,
        default=0,
        help="0 means omit max_tokens for persona controller.",
    )
    parser.add_argument(
        "--controller-max-tokens",
        type=int,
        default=0,
        help="0 means omit max_tokens for turn controller.",
    )
    parser.add_argument(
        "--actor-max-tokens",
        type=int,
        default=0,
        help="0 means omit max_tokens for actor.",
    )

    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)

    parser.add_argument("--keep-raw-content", action="store_true")
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
        "--situation-max-iterations",
        type=int,
        default=200,
        help="Hard cap on producer iterations (safety net).",
    )
    parser.add_argument(
        "--situation-model",
        default=SITUATION_GEN_MODEL_DEFAULT,
        help="Model used by the background situation producer.",
    )
    parser.add_argument(
        "--situation-prompt-file",
        default="./prompts/situation_gen.txt",
    )
    parser.add_argument("--situation-temperature", type=float, default=1.1)
    parser.add_argument("--situation-top-p", type=float, default=0.95)
    parser.add_argument("--situation-max-tokens", type=int, default=0)
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

    if args.min_turns <= 0:
        raise ValueError("--min-turns must be positive")

    if args.max_turns < args.min_turns:
        raise ValueError("--max-turns must be >= --min-turns")

    if args.variations_per_line <= 0:
        raise ValueError("--variations-per-line must be positive")

    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    persona_lines = load_persona_lines(args.persona_txt)
    prompts = load_prompts(args.prompt_dir)

    initial_pool = len(persona_lines)
    if args.num_conversations is not None:
        total_requested = args.num_conversations
    else:
        total_requested = initial_pool * args.variations_per_line

    needed_situations = max(
        initial_pool, math.ceil(total_requested / args.variations_per_line)
    )

    auto_gen_active = (
        args.auto_generate_situations and needed_situations > initial_pool
    )

    if needed_situations > initial_pool and not args.auto_generate_situations:
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

    errors_out = args.errors_out or args.out + ".errors.jsonl"
    already_done = count_jsonl_lines(args.out) if args.resume else 0

    if already_done >= total_requested:
        print(f"nothing to do: {already_done} >= {total_requested}", file=sys.stderr)
        return

    persona_thinking_enabled = not args.disable_persona_thinking
    turn_controller_thinking_enabled = args.enable_turn_controller_thinking
    actor_thinking_enabled = not args.disable_actor_thinking

    if auto_gen_active:
        seeds: List[str] = list(args.situation_seed or [])
        if args.situation_seed_file:
            from .io_utils import read_text as _read_text
            for line in _read_text(args.situation_seed_file).splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    seeds.append(s)
        if not seeds:
            # Inspire from existing format.txt content.
            seeds = [pl.text for pl in persona_lines[:8]]

        start_background_producer(
            buffer=persona_buffer,
            out_path=args.persona_txt,
            prompt_file=args.situation_prompt_file,
            seeds=seeds,
            target_count=needed_situations,
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
        shown_host = (
            "127.0.0.1"
            if args.progress_host in ("0.0.0.0", "")
            else args.progress_host
        )
        print(
            json.dumps(
                {
                    "event": "progress_server_started",
                    "url": f"http://{shown_host}:{args.progress_port}",
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
        "workers": args.workers,
        "model": args.model,
        "persona_thinking_enabled": persona_thinking_enabled,
        "turn_controller_thinking_enabled": turn_controller_thinking_enabled,
        "actor_thinking_enabled": actor_thinking_enabled,
        "reasoning_effort": args.reasoning_effort,
        "python_quality_filtering": False,
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
            "workers": args.workers,
            "out": args.out,
        },
        event=start_event,
    )

    written = already_done
    work_indices = list(range(already_done, total_requested))

    if args.workers <= 1:
        get_thread_client(args.base_url)

        with tqdm(total=total_requested, initial=already_done) as pbar:
            for idx0 in work_indices:
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
                )

                if result["ok"]:
                    append_jsonl(args.out, result["record"])
                    written += 1

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

        with tqdm(total=total_requested, initial=already_done) as pbar:
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
                    )
                    for idx0 in work_indices
                ]

                for future in as_completed(futures):
                    result = future.result()

                    if result["ok"]:
                        append_jsonl(args.out, result["record"])
                        written += 1
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
