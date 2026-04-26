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

### Inline situation generation during `main.py`

You can also have `main.py` grow `format.txt` *while it runs*, by passing
`--auto-generate-situations`. A background producer (deepseek-v4-flash)
appends new situations whenever a dialogue worker would otherwise need a
line that doesn't exist yet. Workers block until the producer adds one.

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --variations-per-line 1 `
  --num-conversations 200 `
  --workers 6 `
  --auto-generate-situations `
  --situation-batch-size 8 `
  --situation-seed "Aは..., Bは..., ..."
```

Producer runs are async; if it can't keep up, workers wait. If it
permanently fails (max iterations reached), waiting workers fail with
`persona line N unavailable` instead of hanging.

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

To expose the progress UI to other machines on the LAN, bind to all
interfaces (default is `127.0.0.1` for safety):

```powershell
uv run python main.py `
  ... `
  --progress-server `
  --progress-host 0.0.0.0 `
  --progress-port 8765
```

The startup log then prints every reachable URL (loopback, LAN IP,
hostname). Make sure the chosen port is allowed through the host
firewall.

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

## Finishing short conversations

If an existing JSONL contains conversations that ended before a desired turn
count, continue only those records in-place with the same conversation IDs:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --prompt-dir .\prompts `
  --finish-min-turns 8 `
  --max-turns 12 `
  --workers 4
```

Records whose `actual_turns` / `public_timeline` length is below
`--finish-min-turns` are resumed from their saved `persona_seed` and timeline,
then the JSONL is rewritten with the extended records.

To only detect short conversations without calling the API or rewriting the
JSONL:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --finish-min-turns 8 `
  --finish-dry-run `
  --finish-dry-run-format lines
```

## Rewriting specific conversations

To regenerate specific existing records while preserving their conversation
IDs, pass one or more IDs:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --prompt-dir .\prompts `
  --rewrite-id persona_deepseek_triple_ja_00000007 `
  --min-turns 8 `
  --max-turns 16
```

You can repeat `--rewrite-id`, pass comma-separated IDs, or put one ID per line
in `--rewrite-ids-file`. Use `--rewrite-dry-run` first to verify the matched
records without calling the API or rewriting the JSONL.

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
