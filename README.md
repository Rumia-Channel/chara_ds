# chara_ds_v8

DeepSeek triple-agent dialogue dataset generator.

## DeepSeek API features in use

- **Beta endpoint** (`https://api.deepseek.com/beta`) — required for the
  features below. Override with `--base-url` if needed.
- **Strict tool schema** (`function.strict = true`) — the actor agent emits
  its turn through a function call (`submit_actor_turn`); the API enforces
  the JSON Schema server-side, so the actor cannot return malformed output.
- **KV Context Caching** — each agent's invariants for one conversation
  (persona seed, character profile, scenario constraints, …) are appended to
  the system prompt, so every subsequent call within the same conversation
  hits the prompt cache. Per-token input cost drops to ~1/12 on cache hit.
  Hit/miss counts are surfaced in `usage.prompt_cache_{hit,miss}_tokens`
  and per-turn progress events.
- **Thinking mode** (`extra_body.thinking.type = enabled`) with
  `reasoning_effort=high` for persona / actor by default.

## Generating base situations

`format.txt` (one situation per line) is the input. You can grow it
automatically with `gen_situations.py`, which uses `deepseek-v4-flash`
(non-thinking, cheap & fast) plus a strict tool schema to emit diverse
1-line situations covering many emotion mixes (anger, sadness, joy,
emptiness, fear, …).

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."

uv run python gen_situations.py `
  --out .\format.txt `
  --target 100 `
  --batch-size 8 `
  --seed-situation "Aは長年連れ添った夫、Bは若い後妻、過去の妻の遺品をめぐって静かに口論する。" `
  --seed-situation "Aは厳格な部活顧問、Bは練習をサボった部員、放課後の教室で対峙する。"
```

Generated lines are appended to `--out` with sha256 dedup, so re-running
just continues filling toward `--target`. Use `--use-existing-as-seed` to
also feed already-existing lines as inspiration.

## Files

```text
main.py
format_example.txt
prompts/
  persona_controller.txt
  turn_controller.txt
  actor.txt
```

## Install

```powershell
uv add openai tqdm
```

or

```powershell
pip install openai tqdm
```

## API key on PowerShell

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

## Test run

```powershell
uv run python main.py `
  --persona-txt .\format_example.txt `
  --out .\test_v8.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 1 `
  --min-turns 4 `
  --max-turns 6 `
  --workers 2 `
  --reasoning-effort high `
  --progress-server
```

Open:

```text
http://127.0.0.1:8765
```

## Production-ish run

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues_v8.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 50 `
  --min-turns 8 `
  --max-turns 16 `
  --workers 4 `
  --reasoning-effort high `
  --progress-server
```

## Notes

- `public_timeline` contains both utterances and `visible_action`.
- If A performs an action visible to B, B receives it in the next actor prompt.
- Actor output includes `physical_action` with fields such as `action`, `target_speaker`, `target_body_part`, `harm_level`, and `visible_to_other`.
- Turn Controller thinking is disabled by default to reduce cost.
- Persona Controller and Actors have thinking enabled by default.
- Controller utterance leak detection is enabled by default. To disable it:

```powershell
--allow-controller-utterance-leak
```

Use that only if the filter is too strict.
