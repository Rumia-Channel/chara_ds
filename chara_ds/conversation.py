"""Single-conversation generation logic."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .agents import call_actor, call_persona_controller, call_turn_controller
from .api_client import call_with_retries
from .config import DATASET_NAME, PersonaLine, PromptBundle, SCHEMA_VERSION
from .io_utils import now_iso, sha256_json, sha256_text
from .progress import progress_update
from .turn_cache import (
    compute_signature,
    load_turn_cache,
    save_turn_cache,
)


def build_source_info(
    *,
    persona_txt_path: str,
    persona_line: PersonaLine,
    variation: int,
) -> Dict[str, Any]:
    return {
        "type": "line_txt",
        "filename": str(persona_txt_path),
        "line_number": persona_line.line_number,
        "variation": variation,
        "text": persona_line.text,
        "sha256": persona_line.sha256,
    }


def make_public_timeline_event(
    turn_index: int,
    speaker: str,
    actor_content: Dict[str, Any],
) -> Dict[str, Any]:
    physical_action = actor_content.get("physical_action")

    # physical_action is now a free-text string from the marker format. Treat
    # any non-empty action as visible (the model decides "特に動かない" etc.).
    visible_action: Optional[str] = None
    if isinstance(physical_action, str) and physical_action.strip():
        visible_action = physical_action.strip()

    return {
        "turn": turn_index,
        "speaker": speaker,
        "utterance": actor_content.get("public_utterance", ""),
        "visible_action": visible_action,
    }


def generate_one_conversation(
    *,
    client: OpenAI,
    prompts: PromptBundle,
    model: str,
    persona_txt_path: str,
    persona_line: PersonaLine,
    conversation_index: int,
    variation: int,
    min_turns: int,
    max_turns: int,
    seed: int,
    reasoning_effort: str,
    persona_thinking_enabled: bool,
    turn_controller_thinking_enabled: bool,
    actor_thinking_enabled: bool,
    controller_temperature: float,
    controller_top_p: float,
    persona_max_tokens: Optional[int],
    controller_max_tokens: Optional[int],
    actor_max_tokens: Optional[int],
    keep_raw_content: bool,
    errors_out: str,
    retries: int,
    retry_base_sleep: float,
    cache_dir: Optional[str] = None,
    existing_record: Optional[Dict[str, Any]] = None,
    target_turns_override: Optional[int] = None,
    conversation_id_override: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed + conversation_index * 1009 + variation * 9173)
    target_turns = max(rng.randint(min_turns, max_turns), target_turns_override or 0)
    conversation_id = (
        conversation_id_override
        if conversation_id_override is not None
        else
        str(existing_record.get("id") or existing_record.get("conversation_id"))
        if existing_record is not None
        else f"persona_deepseek_triple_ja_{conversation_index:08d}"
    )

    if existing_record is not None:
        source_info = dict(existing_record.get("source") or {})
        if not source_info:
            source_info = build_source_info(
                persona_txt_path=persona_txt_path,
                persona_line=persona_line,
                variation=variation,
            )
    else:
        source_info = build_source_info(
            persona_txt_path=persona_txt_path,
            persona_line=persona_line,
            variation=variation,
        )

    # ----- partial cache: signature & resume -----
    cache_signature = compute_signature(
        {
            "conversation_id": conversation_id,
            "conversation_index": conversation_index,
            "variation": variation,
            "seed": seed,
            "min_turns": min_turns,
            "max_turns": max_turns,
            "target_turns": target_turns,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "persona_thinking_enabled": persona_thinking_enabled,
            "turn_controller_thinking_enabled": turn_controller_thinking_enabled,
            "actor_thinking_enabled": actor_thinking_enabled,
            "controller_temperature": controller_temperature,
            "controller_top_p": controller_top_p,
            "persona_max_tokens": persona_max_tokens,
            "controller_max_tokens": controller_max_tokens,
            "actor_max_tokens": actor_max_tokens,
            "persona_line_sha256": persona_line.sha256,
            "persona_line_number": persona_line.line_number,
            "prompts": {
                "persona_controller_sha256": sha256_text(prompts.persona_controller),
                "turn_controller_sha256": sha256_text(prompts.turn_controller),
                "actor_sha256": sha256_text(prompts.actor),
            },
        }
    )

    cached: Optional[Dict[str, Any]] = None
    if cache_dir:
        c = load_turn_cache(cache_dir, conversation_id)
        if c is not None and c.get("signature") == cache_signature:
            cached = c

    if existing_record is not None:
        persona_generation = existing_record.get("persona_generation") or {}
        persona_content = persona_generation.get("controller_content") or {
            "persona_seed": existing_record.get("persona_seed") or {}
        }
        persona_reasoning = persona_generation.get("controller_reasoning_content")
        persona_usage = persona_generation.get("usage") or {}
        persona_raw = persona_generation.get("raw_content") or ""
        persona_seed = existing_record.get("persona_seed") or persona_content.get("persona_seed") or {}
        public_timeline = list(existing_record.get("public_timeline") or [])
        turns = list(existing_record.get("turns") or [])
        usage_summary = existing_record.get("usage") or {
            "persona_controller": persona_usage,
            "turn_controller": [],
            "actors": [],
        }
        start_turn = len(turns) + 1
        early_end = False

        progress_update(
            status="resuming_existing_record",
            conversation_id=conversation_id,
            current={
                "stage": "resuming_existing_record",
                "conversation_id": conversation_id,
                "conversation_index": conversation_index,
                "completed_turns": len(turns),
                "target_turns": target_turns,
            },
            latest_public_timeline=public_timeline,
            event={
                "type": "resuming_existing_record",
                "conversation_id": conversation_id,
                "completed_turns": len(turns),
                "target_turns": target_turns,
            },
        )
    elif cached is not None:
        persona_content = cached["persona_content"]
        persona_reasoning = cached.get("persona_reasoning")
        persona_usage = cached.get("persona_usage") or {}
        persona_raw = cached.get("persona_raw") or ""
        persona_seed = persona_content["persona_seed"]
        public_timeline: List[Dict[str, Any]] = list(cached.get("public_timeline") or [])
        turns: List[Dict[str, Any]] = list(cached.get("turns") or [])
        usage_summary: Dict[str, Any] = cached.get("usage_summary") or {
            "persona_controller": persona_usage,
            "turn_controller": [],
            "actors": [],
        }
        start_turn = len(turns) + 1
        early_end = bool(cached.get("early_end"))

        progress_update(
            status="resumed_from_cache",
            conversation_id=conversation_id,
            current={
                "stage": "resumed_from_cache",
                "conversation_id": conversation_id,
                "conversation_index": conversation_index,
                "completed_turns": len(turns),
                "target_turns": target_turns,
            },
            latest_public_timeline=public_timeline,
            event={
                "type": "resumed_from_cache",
                "conversation_id": conversation_id,
                "completed_turns": len(turns),
                "target_turns": target_turns,
            },
        )
    else:
        progress_update(
            status="persona_controller_running",
            conversation_id=conversation_id,
            current={
                "stage": "persona_controller",
                "conversation_id": conversation_id,
                "conversation_index": conversation_index,
                "source_line_number": persona_line.line_number,
                "variation": variation,
                "target_turns": target_turns,
            },
            latest_public_timeline=[],
            event={
                "type": "conversation_start",
                "conversation_id": conversation_id,
                "target_turns": target_turns,
            },
        )

        persona_content, persona_reasoning, persona_usage, persona_raw = call_with_retries(
            lambda: call_persona_controller(
                client,
                prompts=prompts,
                model=model,
                source_info=source_info,
                user_txt=persona_line.text,
                conversation_id=conversation_id,
                reasoning_effort=reasoning_effort,
                max_tokens=persona_max_tokens,
                thinking_enabled=persona_thinking_enabled,
            ),
            retries=retries,
            errors_out=errors_out,
            error_context={
                "stage": "persona_controller",
                "conversation_id": conversation_id,
                "source": source_info,
            },
            retry_base_sleep=retry_base_sleep,
        )

        persona_seed = persona_content["persona_seed"]

        progress_update(
            status="persona_controller_done",
            conversation_id=conversation_id,
            current={
                "stage": "persona_controller_done",
                "conversation_id": conversation_id,
            },
            last_persona=persona_content,
        )

        public_timeline = []
        turns = []
        usage_summary = {
            "persona_controller": persona_usage,
            "turn_controller": [],
            "actors": [],
        }
        start_turn = 1
        early_end = False

        if cache_dir:
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
                    "persona_raw": persona_raw if keep_raw_content else "",
                    "public_timeline": public_timeline,
                    "turns": turns,
                    "usage_summary": usage_summary,
                    "early_end": early_end,
                    "saved_at": now_iso(),
                },
            )

    if not early_end:
        for turn_index in range(start_turn, target_turns + 1):
            progress_update(
                status="turn_controller_running",
                conversation_id=conversation_id,
                current={
                    "stage": "turn_controller",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "target_turns": target_turns,
                },
                latest_public_timeline=public_timeline,
            )

            controller_content, controller_reasoning, controller_usage, controller_raw = call_with_retries(
                lambda: call_turn_controller(
                    client,
                    prompts=prompts,
                    model=model,
                    conversation_id=conversation_id,
                    persona_seed=persona_seed,
                    public_timeline=public_timeline,
                    turn_index=turn_index,
                    target_turns=target_turns,
                    reasoning_effort=reasoning_effort,
                    max_tokens=controller_max_tokens,
                    thinking_enabled=turn_controller_thinking_enabled,
                    temperature=controller_temperature,
                    top_p=controller_top_p,
                ),
                retries=retries,
                errors_out=errors_out,
                error_context={
                    "stage": "turn_controller",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                },
                retry_base_sleep=retry_base_sleep,
            )

            turn_control = controller_content["turn_control"]
            speaker = turn_control.get("next_speaker")

            if speaker not in ("A", "B"):
                speaker = "A" if turn_index % 2 == 1 else "B"
                turn_control["next_speaker"] = speaker

            progress_update(
                status="actor_running",
                conversation_id=conversation_id,
                current={
                    "stage": "actor",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "target_turns": target_turns,
                    "speaker": speaker,
                },
                latest_public_timeline=public_timeline,
                last_controller=controller_content,
            )

            actor_content, actor_reasoning, actor_usage, actor_raw = call_with_retries(
                lambda: call_actor(
                    client,
                    prompts=prompts,
                    model=model,
                    speaker=speaker,
                    persona_seed=persona_seed,
                    turn_control=turn_control,
                    public_timeline=public_timeline,
                    turn_index=turn_index,
                    reasoning_effort=reasoning_effort,
                    max_tokens=actor_max_tokens,
                    thinking_enabled=actor_thinking_enabled,
                ),
                retries=retries,
                errors_out=errors_out,
                error_context={
                    "stage": "actor",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "speaker": speaker,
                },
                retry_base_sleep=retry_base_sleep,
            )

            public_event = make_public_timeline_event(turn_index, speaker, actor_content)
            public_timeline.append(public_event)

            turn_record = {
                "turn": turn_index,
                "controller": {
                    "content": controller_content,
                    "reasoning_content": controller_reasoning,
                    "usage": controller_usage,
                    "thinking_enabled": turn_controller_thinking_enabled,
                },
                "actor": {
                    "speaker": speaker,
                    "content": actor_content,
                    "reasoning_content": actor_reasoning,
                    "usage": actor_usage,
                    "thinking_enabled": actor_thinking_enabled,
                },
                "public_event": public_event,
            }

            if keep_raw_content:
                turn_record["controller"]["raw_content"] = controller_raw
                turn_record["actor"]["raw_content"] = actor_raw

            turns.append(turn_record)
            usage_summary["turn_controller"].append(controller_usage)
            usage_summary["actors"].append(actor_usage)

            should_break = (
                turn_index >= min_turns and turn_control.get("should_end") is True
            )
            if should_break:
                early_end = True

            if cache_dir:
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
                        "persona_raw": persona_raw if keep_raw_content else "",
                        "public_timeline": public_timeline,
                        "turns": turns,
                        "usage_summary": usage_summary,
                        "early_end": early_end,
                        "saved_at": now_iso(),
                    },
                )

            progress_update(
                status="actor_done",
                conversation_id=conversation_id,
                current={
                    "stage": "actor_done",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "target_turns": target_turns,
                    "speaker": speaker,
                },
                latest_public_timeline=public_timeline,
                last_actor=actor_content,
                event={
                    "type": "actor_done",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "utterance": actor_content.get("public_utterance", ""),
                    "cache_hit_tokens": actor_usage.get("prompt_cache_hit_tokens"),
                    "cache_miss_tokens": actor_usage.get("prompt_cache_miss_tokens"),
                },
            )

            if should_break:
                break

    public_transcript = [
        {
            "speaker": ev["speaker"],
            "text": ev.get("utterance", ""),
        }
        for ev in public_timeline
    ]

    record = {
        "id": conversation_id,
        "dataset": DATASET_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "synthetic": True,
        "language": "ja",
        "source": source_info,
        "agents": {
            "persona_controller": {
                "provider": "deepseek",
                "model": model,
                "role": "persona_controller",
                "thinking": {
                    "enabled": persona_thinking_enabled,
                    "reasoning_effort": reasoning_effort if persona_thinking_enabled else None,
                },
            },
            "turn_controller": {
                "provider": "deepseek",
                "model": model,
                "role": "turn_controller",
                "thinking": {
                    "enabled": turn_controller_thinking_enabled,
                    "reasoning_effort": reasoning_effort if turn_controller_thinking_enabled else None,
                },
            },
            "actor_A": {
                "provider": "deepseek",
                "model": model,
                "role": "character_actor",
                "thinking": {
                    "enabled": actor_thinking_enabled,
                    "reasoning_effort": reasoning_effort if actor_thinking_enabled else None,
                },
            },
            "actor_B": {
                "provider": "deepseek",
                "model": model,
                "role": "character_actor",
                "thinking": {
                    "enabled": actor_thinking_enabled,
                    "reasoning_effort": reasoning_effort if actor_thinking_enabled else None,
                },
            },
        },
        "generation_config": {
            "target_turns": target_turns,
            "actual_turns": len(public_timeline),
            "min_turns": min_turns,
            "max_turns": max_turns,
            "seed": seed,
            "variation": variation,
            "controller_temperature": controller_temperature if not turn_controller_thinking_enabled else None,
            "controller_top_p": controller_top_p if not turn_controller_thinking_enabled else None,
            "python_quality_filtering": False,
            "max_tokens_policy": {
                "persona_max_tokens": persona_max_tokens,
                "controller_max_tokens": controller_max_tokens,
                "actor_max_tokens": actor_max_tokens,
                "zero_or_none_means_omitted": True,
            },
        },
        "prompt_hashes": {
            "persona_controller_sha256": sha256_text(prompts.persona_controller),
            "turn_controller_sha256": sha256_text(prompts.turn_controller),
            "actor_sha256": sha256_text(prompts.actor),
        },
        "persona_generation": {
            "controller_content": persona_content,
            "controller_reasoning_content": persona_reasoning,
            "usage": persona_usage,
            "thinking_enabled": persona_thinking_enabled,
        },
        "persona_seed": persona_seed,
        "turns": turns,
        "public_timeline": public_timeline,
        "public_transcript": public_transcript,
        "usage": usage_summary,
        "hashes": {
            "source_sha256": source_info.get("sha256") or persona_line.sha256,
            "persona_seed_sha256": sha256_json(persona_seed),
            "public_timeline_sha256": sha256_json(public_timeline),
            "conversation_sha256": sha256_json(
                {
                    "persona_seed": persona_seed,
                    "public_timeline": public_timeline,
                }
            ),
        },
    }

    if keep_raw_content:
        record["persona_generation"]["raw_content"] = persona_raw

    progress_update(
        status="conversation_done",
        conversation_id=conversation_id,
        current={
            "stage": "conversation_done",
            "conversation_id": conversation_id,
            "actual_turns": len(public_timeline),
        },
        latest_public_timeline=public_timeline,
        event={
            "type": "conversation_done",
            "conversation_id": conversation_id,
            "actual_turns": len(public_timeline),
        },
        remove_active=True,
    )

    return record
