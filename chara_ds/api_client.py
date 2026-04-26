"""DeepSeek API client wrappers and retry logic."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from openai import BadRequestError, OpenAI

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


def _build_messages(
    system_prompt: str,
    user_payload: Dict[str, Any],
    static_context: Optional[Dict[str, Any]],
) -> list:
    """Build the messages list with KV-cache friendly layout.

    static_context (if provided) is appended to the system prompt so that
    the entire prefix (system + per-conversation invariants) is identical
    for every call within the same conversation/speaker. DeepSeek's KV
    Context Caching persists this prefix as a "cache prefix unit" and
    subsequent calls hit the cache for it (input cost ~12x cheaper).

    The volatile per-turn data goes into the single user message.
    """
    if static_context is None:
        sys_content = system_prompt
    else:
        sys_content = (
            system_prompt
            + "\n\n# 不変のセッション情報 (このセッション中は変化しない)\n"
            + json.dumps(static_context, ensure_ascii=False)
        )

    return [
        {"role": "system", "content": sys_content},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]


def call_deepseek_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_tokens: Optional[int],
    reasoning_effort: str,
    thinking_enabled: Optional[bool],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    static_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any], str]:
    messages = _build_messages(system_prompt, user_payload, static_context)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    # NOTE: response_format={"type": "json_object"} は付けない。
    # DeepSeek の thinking モードと JSON 強制モードを併用すると、
    # 出力トークンが丸ごと reasoning に吸収されて content が空になる事象がある。
    # プロンプト側で json schema を明示しているため、強制せずとも JSON は返る。
    # parse_json はコードフェンス除去と {...} 抽出に対応しているので堅牢。

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
    elif thinking_enabled is False:
        kwargs["extra_body"] = {
            "thinking": {
                "type": "disabled",
            }
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
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


def call_deepseek_text(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_tokens: Optional[int],
    reasoning_effort: str,
    thinking_enabled: Optional[bool],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    static_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str], Dict[str, Any], str]:
    """Plain-text variant of `call_deepseek_json` used for marker-format outputs.

    Returns (text, reasoning_content, usage, raw_content). `text` is the model's
    `message.content` (or, if empty, the reasoning_content as a fallback when
    DeepSeek's thinking mode swallows the body into reasoning).
    """
    messages = _build_messages(system_prompt, user_payload, static_context)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if max_tokens is not None and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens

    if thinking_enabled:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    elif thinking_enabled is False:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
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

    if not raw_content.strip():
        if reasoning_content and reasoning_content.strip():
            return reasoning_content, reasoning_content, usage, reasoning_content

        raise ValueError(
            "empty model content "
            f"(finish_reason={finish_reason!r}, "
            f"has_reasoning={reasoning_content is not None}, "
            f"reasoning_len={len(reasoning_content) if reasoning_content else 0}, "
            f"usage={usage})"
        )

    return raw_content, reasoning_content, usage, raw_content


def call_deepseek_tool(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    tool_name: str,
    tool_description: str,
    tool_parameters: Dict[str, Any],
    max_tokens: Optional[int],
    reasoning_effort: str,
    thinking_enabled: Optional[bool],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    static_context: Optional[Dict[str, Any]] = None,
    tool_strict: bool = True,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any], str]:
    """Force the model to emit the response as a function/tool call.

    Returns (parsed_arguments, reasoning_content, usage, raw_arguments_json).

    With ``tool_strict=True`` (and the beta base_url), DeepSeek validates the
    JSON Schema **server-side** when the model emits a tool call, so format
    violations are eliminated at the API level. Combined with KV cache via
    ``static_context``, this is the cheapest reliable way to force structured
    output for high-volume generation.
    """
    messages = _build_messages(system_prompt, user_payload, static_context)

    function_def: Dict[str, Any] = {
        "name": tool_name,
        "description": tool_description,
        "parameters": tool_parameters,
    }
    if tool_strict:
        # Beta-only feature: server-side JSON Schema enforcement on tool args.
        # Requires base_url=https://api.deepseek.com/beta.
        function_def["strict"] = True

    tools = [{"type": "function", "function": function_def}]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        # NOTE: deepseek-reasoner (thinking mode) は
        # tool_choice={"type":"function","function":{"name":...}} の specific
        # tool 強制をサポートしていない (BadRequest になる)。"auto" に統一し、
        # プロンプトと content/reasoning フォールバックで実質的に強制する。
        "tool_choice": "auto",
    }

    if max_tokens is not None and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens

    if thinking_enabled:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    elif thinking_enabled is False:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
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
    reasoning_content = get_reasoning_content(msg)
    usage = usage_to_dict(getattr(response, "usage", None))
    finish_reason = getattr(choice, "finish_reason", None)

    tool_calls = getattr(msg, "tool_calls", None) or []
    raw_args = ""

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None)
        args = getattr(fn, "arguments", None) or ""
        if name == tool_name and args.strip():
            raw_args = args
            break

    # Fallback: if the model emitted plain content / put the JSON into reasoning,
    # try to recover. Same defensive fallback we use for json/text variants.
    if not raw_args:
        body = (msg.content or "").strip()
        if body:
            try:
                parsed = parse_json(body)
                return parsed, reasoning_content, usage, body
            except Exception:
                pass
        if reasoning_content and reasoning_content.strip():
            try:
                parsed = parse_json(reasoning_content)
                return parsed, reasoning_content, usage, reasoning_content
            except Exception:
                pass

        raise ValueError(
            "empty tool_call arguments "
            f"(finish_reason={finish_reason!r}, "
            f"tool_calls={len(tool_calls)}, "
            f"has_reasoning={reasoning_content is not None}, "
            f"reasoning_len={len(reasoning_content) if reasoning_content else 0}, "
            f"usage={usage})"
        )

    parsed = parse_json(raw_args)
    return parsed, reasoning_content, usage, raw_args


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
        except BadRequestError as e:
            # 4xx (invalid request, unsupported tool_choice, schema mismatch等)
            # はリトライしても同じ結果になるので即 fail させる。
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
            raise
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
