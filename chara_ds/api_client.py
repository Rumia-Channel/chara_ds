"""DeepSeek API client wrappers and retry logic."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from .io_utils import append_jsonl, now_iso, parse_json
from .progress import progress_update


THREAD_LOCAL = threading.local()


def make_client(base_url: str) -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def get_thread_client(base_url: str) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)

    if client is None:
        client = make_client(base_url)
        THREAD_LOCAL.client = client

    return client


def get_reasoning_content(message: Any) -> Optional[str]:
    direct = getattr(message, "reasoning_content", None)
    if direct:
        return direct

    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get("reasoning_content"):
        return extra["reasoning_content"]

    try:
        dumped = message.model_dump()
        if dumped.get("reasoning_content"):
            return dumped["reasoning_content"]
    except Exception:
        pass

    return None


def usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}

    try:
        return usage.model_dump()
    except Exception:
        pass

    try:
        return dict(usage)
    except Exception:
        return {"raw": str(usage)}


def call_deepseek_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_tokens: Optional[int],
    reasoning_effort: str,
    thinking_enabled: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any], str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    # 0 or None の場合は max_tokens を API に渡さない。
    if max_tokens is not None and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens

    if thinking_enabled:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {
            "thinking": {
                "type": "enabled",
            }
        }
    else:
        kwargs["extra_body"] = {
            "thinking": {
                "type": "disabled",
            }
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

    try:
        response = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        extra_body = kwargs.get("extra_body") or {}
        if thinking_enabled:
            extra_body["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    msg = choice.message
    raw_content = msg.content or ""
    reasoning_content = get_reasoning_content(msg)
    usage = usage_to_dict(getattr(response, "usage", None))
    finish_reason = getattr(choice, "finish_reason", None)

    # DeepSeek thinking モード × JSON 強制モードでは、出力が丸ごと reasoning に
    # 吸収されて content が空のまま finish_reason='stop' で返ることがある。
    # その場合、reasoning_content の中に JSON 本文が含まれているケースが多いので
    # フォールバックとして救出を試みる。失敗したら従来どおり原因情報付きで例外。
    if not raw_content.strip():
        if reasoning_content and reasoning_content.strip():
            try:
                parsed = parse_json(reasoning_content)
                return parsed, reasoning_content, usage, reasoning_content
            except Exception:
                pass

        raise ValueError(
            "empty model content "
            f"(finish_reason={finish_reason!r}, "
            f"has_reasoning={reasoning_content is not None}, "
            f"reasoning_len={len(reasoning_content) if reasoning_content else 0}, "
            f"usage={usage})"
        )

    parsed = parse_json(raw_content)

    return parsed, reasoning_content, usage, raw_content


def call_with_retries(
    fn,
    *,
    retries: int,
    errors_out: str,
    error_context: Dict[str, Any],
    retry_base_sleep: float,
):
    last_error: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            err = {
                "created_at": now_iso(),
                "attempt": attempt,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(limit=5),
                "context": error_context,
            }
            append_jsonl(errors_out, err)
            progress_update(
                status="error",
                error=err,
                event={
                    "type": "api_call_error",
                    "stage": error_context.get("stage"),
                    "conversation_id": error_context.get("conversation_id"),
                    "turn_index": error_context.get("turn_index"),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            sleep_s = min(60.0, retry_base_sleep * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

    raise RuntimeError(f"failed after {retries} retries: {last_error}") from last_error
