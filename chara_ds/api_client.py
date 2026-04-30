"""DeepSeek API client wrappers and retry logic."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from openai import BadRequestError, OpenAI

from .io_utils import append_jsonl, clip_string, now_iso, parse_json
from .progress import progress_update


THREAD_LOCAL = threading.local()


class ModelOutputParseError(ValueError):
    """Raised when the model returned text that could not be parsed as JSON."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        raw_output: str,
        reasoning_content: Optional[str],
        finish_reason: Any,
        usage: Dict[str, Any],
        parse_error: BaseException,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.raw_output = raw_output
        self.reasoning_content = reasoning_content
        self.finish_reason = finish_reason
        self.usage = usage
        self.parse_error = parse_error


def _validate_json_schema_subset(value: Any, schema: Dict[str, Any], path: str = "$") -> None:
    """Validate the JSON Schema subset used by our DeepSeek tool definitions.

    DeepSeek validates strict tool calls server-side, but our defensive fallback
    can parse JSON from message content / reasoning content. That fallback must
    still reject malformed tool arguments locally.
    """

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"tool arguments schema violation at {path}: expected object")

        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise ValueError(f"tool arguments schema violation at {path}: missing required key {key!r}")

        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"tool arguments schema violation at {path}: unexpected keys {extra!r}")

        for key, child_schema in properties.items():
            if key in value:
                _validate_json_schema_subset(value[key], child_schema, f"{path}.{key}")
        return

    if expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"tool arguments schema violation at {path}: expected array")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_schema_subset(item, item_schema, f"{path}[{index}]")
        return

    if expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"tool arguments schema violation at {path}: expected string")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"tool arguments schema violation at {path}: expected boolean")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"tool arguments schema violation at {path}: expected integer")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ValueError(f"tool arguments schema violation at {path}: {value!r} is not in enum {enum!r}")


def _parse_tool_arguments_or_raise(
    *,
    text: str,
    source: str,
    reasoning_content: Optional[str],
    finish_reason: Any,
    usage: Dict[str, Any],
    tool_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    parsed = _parse_json_or_raise(
        text=text,
        source=source,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage=usage,
    )
    _validate_json_schema_subset(parsed, tool_parameters)
    return parsed


def _parse_json_or_raise(
    *,
    text: str,
    source: str,
    reasoning_content: Optional[str],
    finish_reason: Any,
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return parse_json(text)
    except Exception as e:
        snippet = clip_string(text, 4000)
        raise ModelOutputParseError(
            f"failed to parse model JSON from {source}: {e} | raw_head={snippet!r}",
            source=source,
            raw_output=text,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            usage=usage,
            parse_error=e,
        ) from e


def make_client(base_url: str) -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def make_env_client(*, api_key_env: str, base_url: str) -> OpenAI:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")

    return OpenAI(api_key=api_key, base_url=base_url)


def get_thread_client(base_url: str) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)

    if client is None:
        client = make_client(base_url)
        THREAD_LOCAL.client = client

    return client


def get_thread_env_client(*, name: str, api_key_env: str, base_url: str) -> OpenAI:
    attr = f"client_{name}"
    client = getattr(THREAD_LOCAL, attr, None)

    if client is None:
        client = make_env_client(api_key_env=api_key_env, base_url=base_url)
        setattr(THREAD_LOCAL, attr, client)

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
        d = usage.model_dump()
    except Exception:
        try:
            d = dict(usage)
        except Exception:
            return {"raw": str(usage)}

    return _compact_usage(d)


_USAGE_KEEP = {
    "completion_tokens",
    "prompt_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "reasoning_tokens",
}


def _compact_usage(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _USAGE_KEEP:
        if k in d and d[k] is not None:
            out[k] = d[k]

    details = d.get("completion_tokens_details") or {}
    if isinstance(details, dict):
        rt = details.get("reasoning_tokens")
        if rt is not None and "reasoning_tokens" not in out:
            out["reasoning_tokens"] = rt

    return out


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
            parsed = _parse_json_or_raise(
                text=reasoning_content,
                source="reasoning_content",
                reasoning_content=reasoning_content,
                finish_reason=finish_reason,
                usage=usage,
            )
            return parsed, reasoning_content, usage, reasoning_content

        raise ValueError(
            "empty model content "
            f"(finish_reason={finish_reason!r}, "
            f"has_reasoning={reasoning_content is not None}, "
            f"reasoning_len={len(reasoning_content) if reasoning_content else 0}, "
            f"usage={usage})"
        )

    parsed = _parse_json_or_raise(
        text=raw_content,
        source="message.content",
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage=usage,
    )

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
            parsed = _parse_tool_arguments_or_raise(
                text=body,
                source="message.content",
                reasoning_content=reasoning_content,
                finish_reason=finish_reason,
                usage=usage,
                tool_parameters=tool_parameters,
            )
            return parsed, reasoning_content, usage, body
        if reasoning_content and reasoning_content.strip():
            parsed = _parse_tool_arguments_or_raise(
                text=reasoning_content,
                source="reasoning_content",
                reasoning_content=reasoning_content,
                finish_reason=finish_reason,
                usage=usage,
                tool_parameters=tool_parameters,
            )
            return parsed, reasoning_content, usage, reasoning_content

        raise ValueError(
            "empty tool_call arguments "
            f"(finish_reason={finish_reason!r}, "
            f"tool_calls={len(tool_calls)}, "
            f"has_reasoning={reasoning_content is not None}, "
            f"reasoning_len={len(reasoning_content) if reasoning_content else 0}, "
            f"usage={usage})"
        )

    parsed = _parse_tool_arguments_or_raise(
        text=raw_args,
        source="tool_call.arguments",
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage=usage,
        tool_parameters=tool_parameters,
    )
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

    def _error_details(exc: BaseException) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        raw_output = getattr(exc, "raw_output", None)
        if isinstance(raw_output, str) and raw_output:
            details["raw_output"] = clip_string(raw_output, 12000)
            details["raw_output_length"] = len(raw_output)
        reasoning_content = getattr(exc, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            details["reasoning_content"] = clip_string(reasoning_content, 12000)
            details["reasoning_content_length"] = len(reasoning_content)
        source = getattr(exc, "source", None)
        if isinstance(source, str) and source:
            details["output_source"] = source
        finish_reason = getattr(exc, "finish_reason", None)
        if finish_reason is not None:
            details["finish_reason"] = finish_reason
        usage = getattr(exc, "usage", None)
        if isinstance(usage, dict) and usage:
            details["usage"] = usage
        parse_error = getattr(exc, "parse_error", None)
        if parse_error is not None:
            details["parse_error_type"] = type(parse_error).__name__
            details["parse_error"] = str(parse_error)
        return details

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
            err.update(_error_details(e))
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
                    **({
                        "raw_output": err["raw_output"]
                    } if "raw_output" in err else {}),
                },
            )
            sleep_s = min(60.0, retry_base_sleep * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

    raise RuntimeError(f"failed after {retries} retries: {last_error}") from last_error
