# chara_ds_v8

DeepSeek triple-agent dialogue dataset generator.

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
