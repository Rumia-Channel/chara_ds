"""Single-conversation generation logic."""

from __future__ import annotations

import math
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .agents import (
    call_actor,
    call_actor_guard,
    call_conversation_auditor,
    call_grand_controller,
    call_persona_controller,
    call_turn_controller,
)
from .api_client import call_with_retries
from .config import DATASET_NAME, PersonaLine, PromptBundle, SCHEMA_VERSION
from .io_utils import now_iso, sha256_json, sha256_text
from .progress import progress_update
from .turn_cache import (
    backup_turn_cache,
    cache_path_for,
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
    actor_guard_content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    physical_action = actor_content.get("physical_action")

    # physical_action is now a free-text string from the marker format. Treat
    # any non-empty action as visible (the model decides "特に動かない" etc.).
    visible_action: Optional[str] = None
    if isinstance(physical_action, str) and physical_action.strip():
        visible_action = physical_action.strip()

    event = {
        "turn": turn_index,
        "speaker": speaker,
        "utterance": actor_content.get("public_utterance", ""),
        "visible_action": visible_action,
    }
    if isinstance(actor_guard_content, dict) and isinstance(actor_guard_content.get("filler_analysis"), dict):
        event["filler_analysis"] = actor_guard_content["filler_analysis"]
    return event


def latest_scene_state(turns: List[Dict[str, Any]]) -> Optional[str]:
    """Return the latest controller scene_state saved in turn records."""
    for turn in reversed(turns):
        controller = turn.get("controller") if isinstance(turn, dict) else None
        content = controller.get("content") if isinstance(controller, dict) else None
        tc = content.get("turn_control") if isinstance(content, dict) else None
        state = tc.get("scene_state") if isinstance(tc, dict) else None
        if isinstance(state, str) and state.strip():
            return state.strip()
    return None


def latest_state_memory(turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the latest structured state memory saved in turn records."""
    for turn in reversed(turns):
        controller = turn.get("controller") if isinstance(turn, dict) else None
        content = controller.get("content") if isinstance(controller, dict) else None
        tc = content.get("turn_control") if isinstance(content, dict) else None
        memory = tc.get("state_memory") if isinstance(tc, dict) else None
        if isinstance(memory, dict):
            return memory
    return None


def estimate_ending_pacing_floor(user_txt: str, min_turns: int, max_turns: int) -> Optional[int]:
    """Return a conservative turn floor when the source text contains an explicit ending.

    This biases obvious end-anchored prompts toward the upper part of the turn band so
    the conversation has enough room to reach the stated ending instead of cutting off
    in the middle.
    """

    if min_turns <= 0 or max_turns < min_turns:
        return None

    text = user_txt.strip()
    if not text:
        return None

    strong_markers = (
        "決着",
        "結末",
        "終幕",
        "最後まで",
        "終わりまで",
        "場面まで",
        "勝敗がつく",
        "勝敗が決まる",
        "終わるところまで",
        "ラスト",
    )
    if not any(marker in text for marker in strong_markers):
        return None

    span = max_turns - min_turns
    if span <= 0:
        return min_turns

    if any(marker in text for marker in ("決着", "結末", "終幕", "最後まで", "終わりまで", "勝敗がつく", "勝敗が決まる")):
        ratio = 0.8
    else:
        ratio = 0.7

    floor = min_turns + math.ceil(span * ratio)
    return min(max(floor, min_turns), max_turns)


def normalize_persona_labels(persona_seed: Dict[str, Any]) -> Dict[str, Any]:
    """Keep dataset speaker labels as A/B and drop invented name fields."""
    if not isinstance(persona_seed, dict):
        return persona_seed

    characters = persona_seed.get("characters")
    if not isinstance(characters, dict):
        return persona_seed

    for label in ("A", "B"):
        profile = characters.get(label)
        if not isinstance(profile, dict):
            continue
        for key in (
            "name",
            "full_name",
            "display_name",
            "nickname",
            "given_name",
            "family_name",
        ):
            profile.pop(key, None)
    return persona_seed


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
    state_memory_tool_enabled: bool,
    resume_accept_stale_cache: bool,
    actor_thinking_enabled: bool,
    actor_guard_enabled: bool,
    actor_guard_model: str,
    actor_guard_provider: str,
    actor_guard_client: Optional[OpenAI],
    actor_guard_thinking_enabled: bool,
    conversation_audit_enabled: bool,
    conversation_audit_model: str,
    conversation_audit_provider: str,
    conversation_audit_client: Optional[OpenAI],
    controller_temperature: float,
    controller_top_p: float,
    persona_max_tokens: Optional[int],
    controller_max_tokens: Optional[int],
    actor_max_tokens: Optional[int],
    actor_guard_max_tokens: Optional[int],
    keep_raw_content: bool,
    errors_out: str,
    retries: int,
    retry_base_sleep: float,
    cache_dir: Optional[str] = None,
    cache_diagnostics: bool = False,
    backup_existing_cache: bool = True,
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
    ending_pacing_floor = estimate_ending_pacing_floor(persona_line.text, min_turns, max_turns)
    if ending_pacing_floor is not None and ending_pacing_floor > target_turns:
        target_turns = ending_pacing_floor
        progress_update(
            status="ending_pacing_floor_applied",
            conversation_id=conversation_id,
            current={
                "stage": "ending_pacing_floor_applied",
                "conversation_id": conversation_id,
                "conversation_index": conversation_index,
                "ending_pacing_floor": ending_pacing_floor,
                "target_turns": target_turns,
            },
            event={
                "type": "ending_pacing_floor_applied",
                "conversation_id": conversation_id,
                "ending_pacing_floor": ending_pacing_floor,
                "target_turns": target_turns,
            },
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
            "state_memory_tool_enabled": state_memory_tool_enabled,
            "actor_thinking_enabled": actor_thinking_enabled,
            "actor_guard_enabled": actor_guard_enabled,
            "actor_guard_model": actor_guard_model,
            "actor_guard_provider": actor_guard_provider,
            "actor_guard_thinking_enabled": actor_guard_thinking_enabled,
            "conversation_audit_enabled": conversation_audit_enabled,
            "conversation_audit_model": conversation_audit_model,
            "conversation_audit_provider": conversation_audit_provider,
            "controller_temperature": controller_temperature,
            "controller_top_p": controller_top_p,
            "persona_max_tokens": persona_max_tokens,
            "controller_max_tokens": controller_max_tokens,
            "actor_max_tokens": actor_max_tokens,
            "actor_guard_max_tokens": actor_guard_max_tokens,
            "persona_line_sha256": persona_line.sha256,
            "persona_line_number": persona_line.line_number,
            "prompts": {
                "persona_controller_sha256": sha256_text(prompts.persona_controller),
                "grand_controller_sha256": sha256_text(prompts.grand_controller),
                "turn_controller_sha256": sha256_text(prompts.turn_controller),
                "actor_sha256": sha256_text(prompts.actor),
                "actor_guard_sha256": sha256_text(prompts.actor_guard),
                "conversation_auditor_sha256": sha256_text(prompts.conversation_auditor),
                "age_gender_norms_sha256": prompts.age_gender_norms_sha256,
            },
        }
    )

    cached: Optional[Dict[str, Any]] = None
    if cache_dir:
        c = load_turn_cache(cache_dir, conversation_id)
        cache_file_exists = False
        cache_not_used_reason = "missing"
        if c is not None and (
            c.get("signature") == cache_signature
            or (resume_accept_stale_cache and not state_memory_tool_enabled)
        ):
            cached = c
            if cache_diagnostics:
                stale_cache = c.get("signature") != cache_signature
                print(
                    json.dumps(
                        {
                            "event": "turn_cache_used_stale" if stale_cache else "turn_cache_used",
                            "conversation_id": conversation_id,
                            "completed_turns": len(c.get("turns") or []),
                            "target_turns": target_turns,
                            "early_end": bool(c.get("early_end")),
                            "stale_signature_accepted": stale_cache,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        else:
            path = cache_path_for(cache_dir, conversation_id)
            cache_file_exists = os.path.exists(path)
            cache_not_used_reason = "missing" if not cache_file_exists else "signature_mismatch_or_unreadable"
            if cache_file_exists and backup_existing_cache:
                backup_turn_cache(cache_dir, conversation_id)
        if cache_diagnostics and cached is None:
            path = cache_path_for(cache_dir, conversation_id)
            event = {
                "event": "turn_cache_not_used",
                "conversation_id": conversation_id,
                "reason": cache_not_used_reason,
                "cache_path": path,
                "current_signature": cache_signature,
                "target_turns": target_turns,
                "backed_up": bool(cache_file_exists and backup_existing_cache),
            }
            if c is not None:
                event.update(
                    {
                        "cached_signature": c.get("signature"),
                        "cached_target_turns": c.get("target_turns"),
                        "cached_completed_turns": len(c.get("turns") or []),
                        "cached_early_end": bool(c.get("early_end")),
                    }
                )
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

    if existing_record is not None:
        persona_generation = existing_record.get("persona_generation") or {}
        persona_content = persona_generation.get("controller_content") or {
            "persona_seed": existing_record.get("persona_seed") or {}
        }
        persona_reasoning = persona_generation.get("controller_reasoning_content")
        persona_usage = persona_generation.get("usage") or {}
        persona_raw = persona_generation.get("raw_content") or ""
        persona_seed = existing_record.get("persona_seed") or persona_content.get("persona_seed") or {}
        persona_seed = normalize_persona_labels(persona_seed)
        persona_content["persona_seed"] = persona_seed
        public_timeline = list(existing_record.get("public_timeline") or [])
        turns = list(existing_record.get("turns") or [])
        usage_summary = existing_record.get("usage") or {
            "persona_controller": persona_usage,
            "turn_controller": [],
            "actors": [],
            "actor_guard": [],
        }
        start_turn = len(turns) + 1
        early_end = False
        previous_scene_state = latest_scene_state(turns)
        previous_state_memory = latest_state_memory(turns)

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
            history_persona=persona_content,
            history_turns=turns,
            history_conversation_audit=existing_record.get("conversation_audit")
            if isinstance(existing_record.get("conversation_audit"), dict)
            else None,
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
        persona_seed = normalize_persona_labels(persona_seed)
        persona_content["persona_seed"] = persona_seed
        public_timeline: List[Dict[str, Any]] = list(cached.get("public_timeline") or [])
        turns: List[Dict[str, Any]] = list(cached.get("turns") or [])
        usage_summary: Dict[str, Any] = cached.get("usage_summary") or {
            "persona_controller": persona_usage,
            "turn_controller": [],
            "actors": [],
            "actor_guard": [],
        }
        start_turn = len(turns) + 1
        early_end = bool(cached.get("early_end"))
        previous_scene_state = latest_scene_state(turns)
        previous_state_memory = latest_state_memory(turns)

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
            history_persona=persona_content,
            history_turns=turns,
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
                min_turns=min_turns,
                max_turns=max_turns,
                target_turns=target_turns,
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

        persona_seed = normalize_persona_labels(persona_content["persona_seed"])
        persona_content["persona_seed"] = persona_seed

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
            "actor_guard": [],
            "conversation_audit": [],
        }
        start_turn = 1
        early_end = False
        previous_scene_state = None
        previous_state_memory = None

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
                backup_existing=backup_existing_cache,
            )

    if not early_end:
        for turn_index in range(start_turn, target_turns + 1):
            progress_update(
                status="grand_controller_running",
                conversation_id=conversation_id,
                current={
                    "stage": "grand_controller",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "target_turns": target_turns,
                },
                latest_public_timeline=public_timeline,
            )

            grand_content, grand_reasoning, grand_usage, grand_raw = call_with_retries(
                lambda: call_grand_controller(
                    client,
                    prompts=prompts,
                    model=model,
                    conversation_id=conversation_id,
                    persona_seed=persona_seed,
                    public_timeline=public_timeline,
                    previous_scene_state=previous_scene_state,
                    previous_state_memory=previous_state_memory,
                    turn_index=turn_index,
                    target_turns=target_turns,
                    reasoning_effort=reasoning_effort,
                    max_tokens=controller_max_tokens,
                    thinking_enabled=turn_controller_thinking_enabled,
                ),
                retries=retries,
                errors_out=errors_out,
                error_context={
                    "stage": "grand_controller",
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                },
                retry_base_sleep=retry_base_sleep,
            )

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
                    previous_scene_state=previous_scene_state,
                    previous_state_memory=previous_state_memory,
                    grand_strategy=grand_content,
                    state_memory_tool_enabled=state_memory_tool_enabled,
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
            state = turn_control.get("scene_state")
            if isinstance(state, str) and state.strip():
                previous_scene_state = state.strip()
            memory = turn_control.get("state_memory")
            if isinstance(memory, dict):
                previous_state_memory = memory
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
                last_grand_controller=grand_content,
                last_controller=controller_content,
            )

            def generate_actor_and_guard():
                if not actor_guard_enabled:
                    actor_result = call_actor(
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
                    )
                    return (*actor_result, None, None, {}, "")

                feedback: Optional[Dict[str, Any]] = None
                guard_attempts: List[Dict[str, Any]] = []
                last_guard = None

                for guard_round in range(1, 3):
                    actor_result = call_actor(
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
                        actor_guard_feedback=feedback,
                    )

                    actor_content_local = actor_result[0]
                    progress_update(
                        status="actor_guard_running",
                        conversation_id=conversation_id,
                        current={
                            "stage": "actor_guard",
                            "conversation_id": conversation_id,
                            "turn_index": turn_index,
                            "speaker": speaker,
                            "guard_round": guard_round,
                        },
                        latest_public_timeline=public_timeline,
                        last_actor=actor_content_local,
                        event={
                            "type": "actor_guard_running",
                            "conversation_id": conversation_id,
                            "turn_index": turn_index,
                            "speaker": speaker,
                            "guard_round": guard_round,
                        },
                    )
                    guard_client = actor_guard_client if actor_guard_provider == "sakura" else client
                    guard_content, guard_reasoning, guard_usage, guard_raw = call_actor_guard(
                        guard_client,
                        prompts=prompts,
                        model=actor_guard_model,
                        speaker=speaker,
                        persona_seed=persona_seed,
                        turn_control=turn_control,
                        public_timeline=public_timeline,
                        conversation_pressure=turn_control.get("conversation_pressure"),
                        actor_content=actor_content_local,
                        turn_index=turn_index,
                        reasoning_effort=reasoning_effort,
                        max_tokens=actor_guard_max_tokens,
                        thinking_enabled=actor_guard_thinking_enabled
                        if actor_guard_provider == "deepseek"
                        else None,
                        tool_strict=actor_guard_provider == "deepseek",
                    )
                    last_guard = (guard_content, guard_reasoning, guard_usage, guard_raw)
                    guard_attempts.append(
                        {
                            "round": guard_round,
                            # Keep a snapshot here. If we store guard_content itself and later
                            # attach guard_attempts back onto it, the final record becomes
                            # self-referential and json.dumps() will fail with
                            # "Circular reference detected".
                            "content": dict(guard_content),
                            "usage": guard_usage,
                        }
                    )
                    if guard_content.get("pass") is True:
                        guard_content["attempts"] = guard_attempts
                        progress_update(
                            status="actor_guard_done",
                            conversation_id=conversation_id,
                            current={
                                "stage": "actor_guard_done",
                                "conversation_id": conversation_id,
                                "turn_index": turn_index,
                                "speaker": speaker,
                                "guard_round": guard_round,
                            },
                            latest_public_timeline=public_timeline,
                            last_actor_guard=guard_content,
                            event={
                                "type": "actor_guard_done",
                                "conversation_id": conversation_id,
                                "turn_index": turn_index,
                                "speaker": speaker,
                                "guard_round": guard_round,
                                "pass": True,
                                "severity": guard_content.get("severity"),
                                "reason": guard_content.get("reason_ja"),
                            },
                        )
                        return (*actor_result, guard_content, guard_reasoning, guard_usage, guard_raw)

                    feedback = {
                        "severity": guard_content.get("severity") or "unknown",
                        "reason_ja": guard_content.get("reason_ja") or "",
                        "suggested_fix_ja": guard_content.get("suggested_fix_ja") or "",
                            "instruction": (
                                "第三者監視役の指摘を反映し、同じターンを人物設定・年齢・"
                                "身体能力・口調・直前文脈に合う形で書き直す。"
                            ),
                        }
                    progress_update(
                        status="actor_guard_retrying",
                        conversation_id=conversation_id,
                        current={
                            "stage": "actor_guard_retrying",
                            "conversation_id": conversation_id,
                            "turn_index": turn_index,
                            "speaker": speaker,
                            "guard_round": guard_round,
                        },
                        latest_public_timeline=public_timeline,
                        last_actor_guard=guard_content,
                        event={
                            "type": "actor_guard_failed",
                            "conversation_id": conversation_id,
                            "turn_index": turn_index,
                            "speaker": speaker,
                            "guard_round": guard_round,
                            "pass": False,
                            "severity": guard_content.get("severity"),
                            "reason": guard_content.get("reason_ja"),
                            "suggested_fix": guard_content.get("suggested_fix_ja"),
                        },
                    )

                reason = (last_guard[0].get("reason_ja") if last_guard else None) or "actor guard rejected output"
                severity = (last_guard[0].get("severity") if last_guard else None) or "unknown"
                raise ValueError(f"actor guard rejected corrected output ({severity}): {reason}")

            (
                actor_content,
                actor_reasoning,
                actor_usage,
                actor_raw,
                guard_content,
                guard_reasoning,
                guard_usage,
                guard_raw,
            ) = call_with_retries(
                generate_actor_and_guard,
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

            public_event = make_public_timeline_event(
                turn_index,
                speaker,
                actor_content,
                guard_content if actor_guard_enabled else None,
            )
            public_timeline.append(public_event)

            turn_record = {
                "turn": turn_index,
                "controller": {
                    "grand_controller": {
                        "content": grand_content,
                        "reasoning_content": grand_reasoning,
                        "usage": grand_usage,
                        "thinking_enabled": turn_controller_thinking_enabled,
                    },
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
            if actor_guard_enabled:
                turn_record["actor_guard"] = {
                    "content": guard_content,
                    "reasoning_content": guard_reasoning,
                    "usage": guard_usage,
                    "thinking_enabled": actor_guard_thinking_enabled,
                    "model": actor_guard_model,
                    "provider": actor_guard_provider,
                }

            if keep_raw_content:
                turn_record["controller"]["grand_controller"]["raw_content"] = grand_raw
                turn_record["controller"]["raw_content"] = controller_raw
                turn_record["actor"]["raw_content"] = actor_raw
                if actor_guard_enabled:
                    turn_record["actor_guard"]["raw_content"] = guard_raw

            turns.append(turn_record)
            if grand_usage:
                usage_summary.setdefault("grand_controller", []).append(grand_usage)
            usage_summary["turn_controller"].append(controller_usage)
            usage_summary["actors"].append(actor_usage)
            if actor_guard_enabled:
                usage_summary.setdefault("actor_guard", []).append(guard_usage)

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
                    backup_existing=backup_existing_cache,
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
                clear_last_actor_guard=not actor_guard_enabled,
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
                "state_memory_tool_enabled": state_memory_tool_enabled,
                "thinking": {
                    "enabled": turn_controller_thinking_enabled,
                    "reasoning_effort": reasoning_effort if turn_controller_thinking_enabled else None,
                },
            },
            "grand_controller": {
                "provider": "deepseek",
                "model": model if prompts.grand_controller.strip() else None,
                "role": "grand_controller",
                "enabled": bool(prompts.grand_controller.strip()),
                "thinking": {
                    "enabled": turn_controller_thinking_enabled if prompts.grand_controller.strip() else False,
                    "reasoning_effort": reasoning_effort
                    if prompts.grand_controller.strip() and turn_controller_thinking_enabled
                    else None,
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
            "python_quality_filtering": True,
            "actor_guard_enabled": actor_guard_enabled,
            "actor_guard_model": actor_guard_model if actor_guard_enabled else None,
            "actor_guard_provider": actor_guard_provider if actor_guard_enabled else None,
            "conversation_audit_enabled": conversation_audit_enabled,
            "conversation_audit_model": conversation_audit_model if conversation_audit_enabled else None,
            "conversation_audit_provider": conversation_audit_provider if conversation_audit_enabled else None,
            "max_tokens_policy": {
                "persona_max_tokens": persona_max_tokens,
                "controller_max_tokens": controller_max_tokens,
                "actor_max_tokens": actor_max_tokens,
                "actor_guard_max_tokens": actor_guard_max_tokens,
                "zero_or_none_means_omitted": True,
            },
        },
        "prompt_hashes": {
            "persona_controller_sha256": sha256_text(prompts.persona_controller),
            "grand_controller_sha256": sha256_text(prompts.grand_controller),
            "turn_controller_sha256": sha256_text(prompts.turn_controller),
            "actor_sha256": sha256_text(prompts.actor),
            "actor_guard_sha256": sha256_text(prompts.actor_guard),
            "conversation_auditor_sha256": sha256_text(prompts.conversation_auditor),
            "age_gender_norms_sha256": prompts.age_gender_norms_sha256,
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

    if actor_guard_enabled:
        record["agents"]["actor_guard"] = {
            "provider": actor_guard_provider,
            "model": actor_guard_model,
            "role": "actor_consistency_guard",
            "thinking": {
                "enabled": actor_guard_thinking_enabled,
                "reasoning_effort": reasoning_effort if actor_guard_thinking_enabled else None,
            },
        }

    if conversation_audit_enabled:
        audit_client = conversation_audit_client if conversation_audit_provider == "sakura" else client
        progress_update(
            status="conversation_audit_running",
            conversation_id=conversation_id,
            current={
                "stage": "conversation_audit",
                "conversation_id": conversation_id,
                "actual_turns": len(public_timeline),
            },
            latest_public_timeline=public_timeline,
            event={
                "type": "conversation_audit_running",
                "conversation_id": conversation_id,
            },
        )
        audit_content, audit_reasoning, audit_usage, audit_raw = call_with_retries(
            lambda: call_conversation_auditor(
                audit_client,
                prompts=prompts,
                model=conversation_audit_model,
                conversation_id=conversation_id,
                persona_seed=persona_seed,
                turns=turns,
                public_timeline=public_timeline,
                reasoning_effort=reasoning_effort,
                max_tokens=actor_guard_max_tokens,
                thinking_enabled=actor_guard_thinking_enabled
                if conversation_audit_provider == "deepseek"
                else None,
                tool_strict=conversation_audit_provider == "deepseek",
            ),
            retries=retries,
            errors_out=errors_out,
            error_context={
                "stage": "conversation_audit",
                "conversation_id": conversation_id,
            },
            retry_base_sleep=retry_base_sleep,
        )
        record["conversation_audit"] = {
            "provider": conversation_audit_provider,
            "model": conversation_audit_model,
            "content": audit_content,
            "reasoning_content": audit_reasoning,
            "usage": audit_usage,
        }
        usage_summary.setdefault("conversation_audit", []).append(audit_usage)
        if keep_raw_content:
            record["conversation_audit"]["raw_content"] = audit_raw
        record["agents"]["conversation_auditor"] = {
            "provider": conversation_audit_provider,
            "model": conversation_audit_model,
            "role": "finished_conversation_auditor",
            "thinking": {
                "enabled": actor_guard_thinking_enabled if conversation_audit_provider == "deepseek" else False,
                "reasoning_effort": reasoning_effort
                if conversation_audit_provider == "deepseek" and actor_guard_thinking_enabled
                else None,
            },
        }
        progress_update(
            status="conversation_audit_done",
            conversation_id=conversation_id,
            current={
                "stage": "conversation_audit_done",
                "conversation_id": conversation_id,
                "actual_turns": len(public_timeline),
            },
            latest_public_timeline=public_timeline,
            last_conversation_audit=record["conversation_audit"],
            event={
                "type": "conversation_audit_done",
                "conversation_id": conversation_id,
                "overall_score": audit_content.get("overall_score"),
                "pass": audit_content.get("pass"),
                "recommended_action": audit_content.get("recommended_action"),
            },
        )

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
