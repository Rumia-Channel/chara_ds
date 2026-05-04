# chara_ds project overview and llama.cpp compatibility

## Purpose
- Project name: PersonaCast-JA / `chara-ds`.
- Generates Japanese two-character dialogue datasets from one-line situation prompts.
- Main output is JSONL with `persona_seed`, detailed `turns`, `public_timeline`, `public_transcript`, usage metadata, hashes, and optional guard/audit results.

## Entrypoints
- `main.py`: thin wrapper for `chara_ds.runner.main`.
- `gen_situations.py`: thin wrapper for `chara_ds.situation_gen.main`.
- Runtime dependency manager: `uv`; Python requirement is `>=3.13`.
- Core deps: `openai`, `tqdm`.

## Major modules
- `chara_ds.runner`: CLI parsing, scheduling, rewrite/finish/resume modes, progress server setup, background situation producer wiring.
- `chara_ds.conversation`: single-conversation lifecycle, turn cache signature/resume, persona generation, controller/actor/guard loop, final JSONL record assembly.
- `chara_ds.agents`: JSON Schemas, validators, and per-agent wrappers for Persona Controller, Grand Controller, Turn Controller, Actor, Actor Guard, Conversation Auditor.
- `chara_ds.api_client`: OpenAI SDK client creation, provider-specific chat controls, JSON/tool-call parsing, schema validation, retry/error logging.
- `chara_ds.situation_gen`: standalone one-line situation generator with strict schema / JSON fallback.
- `chara_ds.situation_producer`: background producer used by `main.py --auto-generate-situations`.
- `chara_ds.io_utils`: prompt loading, JSONL helpers, hashing, parsing.
- `chara_ds.turn_cache`: per-turn cache save/load/backup/signature helpers.
- `chara_ds.progress` plus `chara_ds/web/*`: local progress UI/server state.
- `chara_ds.norms`: selects age/gender norm JSON snippets from prompt directory.

## Prompt directories
- `prompts/`: default controlled prompts.
- `prompts_lax/`: less restrictive creative prompt set while preserving important boundaries.
- Each prompt dir can include `age_gender_norms/index.json` and per-profile JSON files; these are selected and injected only when relevant.

## DeepSeek default behavior
- Default base URL: `https://api.deepseek.com/beta`.
- Default model: `deepseek-v4-pro`; `--flash` uses `deepseek-v4-flash`.
- DeepSeek beta path keeps strict tool schemas and `extra_body.thinking` controls.
- Persona and Actor thinking are enabled by default; Turn Controller thinking is disabled by default.

## llama.cpp compatibility added on 2026-05-02
- New constants in `chara_ds.config`:
  - `LLAMA_CPP_DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"`
  - `LLAMA_CPP_DEFAULT_MODEL = "local"`
- New CLI flag on `main.py` / `chara_ds.runner`: `--llama-cpp`.
  - Sets `CHARA_DS_OPENAI_COMPAT_MODE=llama_cpp`.
  - Defaults `--base-url` to `http://127.0.0.1:8080/v1` unless user provided another URL.
  - Defaults `--model` to `local` unless user provided another model.
  - Changes default thinking to `off`.
  - Changes DeepSeek-sized `max_tokens` defaults to `0`, which means omit `max_tokens`.
  - Makes background situation producer use the same local model when `--situation-model` is omitted.
- New CLI flag on `gen_situations.py` / `chara_ds.situation_gen`: `--llama-cpp`.
  - Same base URL/model shortcut and `CHARA_DS_OPENAI_COMPAT_MODE=llama_cpp`.
- `chara_ds.api_client` now normalizes base URLs ending in `/chat/completions` to the OpenAI SDK base path.
- `make_client` now accepts `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, or `LLAMA_CPP_API_KEY`; for local OpenAI-compatible URLs it uses dummy `sk-no-key-required` if none is set.
- For local/openai-compatible mode, `api_client` omits DeepSeek-only `extra_body.thinking` / `reasoning_effort` and keeps standard `temperature` / `top_p` only.
- For local/openai-compatible mode, `call_deepseek_tool` sends OpenAI `tools` by default, but omits DeepSeek `strict` unless `CHARA_DS_TOOL_STRICT=1` is set.
- `--plain-json-tools` / `CHARA_DS_TOOL_MODE=plain_json` is the explicit fallback that does not send OpenAI `tools`; it embeds the function name, description, and JSON Schema into the system prompt, asks for one JSON object, then validates that object locally with the same schema subset validator.
- DeepSeek beta behavior is preserved for non-local/non-llama-cpp mode.

## llama.cpp usage examples
```powershell
uv run python main.py `
  --llama-cpp `
  --persona-txt .\format.txt `
  --out .\persona_dialogues_llama.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 1 `
  --min-turns 4 `
  --max-turns 6 `
  --workers 1 `
  --progress-server
```

```powershell
uv run python gen_situations.py `
  --llama-cpp `
  --out .\format.txt `
  --target 100 `
  --batch-size 4
```

## Important llama.cpp caveats
- Local models may still fail tool-call/schema validation if they cannot reliably produce the required arguments; lower worker count and smaller turn counts are safer for smoke tests.
- On 2026-05-02, `api_client._parse_tool_arguments_or_raise` was hardened to wrap bare inner-object arguments for single-required-argument tools. Example: if a local tool caller returns the persona seed object directly instead of `{ "persona_seed": ... }`, it is wrapped and then validated.
- If a server/model advertises OpenAI compatibility but its tool-call handling is broken, add `--plain-json-tools` as a fallback.
- `--actor-guard` and `--conversation-audit` increase structured-output pressure; start without them on small local models.
- If using a remote llama.cpp server, still pass `--llama-cpp`; this forces local-json compatibility even when the host is not localhost.
- If the server expects a real model id, override `--model`; `local` is only a conventional placeholder.

## Verification performed
- `ccc index` completed: 53 files, 1040 chunks indexed.
- `python -m compileall chara_ds main.py gen_situations.py` succeeded.
- `uv run python main.py --help` succeeded and shows `--llama-cpp`.
- `uv run python gen_situations.py --help` succeeded and shows `--llama-cpp`.
- First `uv run` attempted dependency download and hit a certificate error in sandbox; rerun with approved network execution succeeded and installed dependencies.
