# PersonaCast-JA

[English](#english)

The generated data is [heare](https://huggingface.co/datasets/RumiaChannel/PersonaCast-JA).

DeepSeek を使った、日本語キャラクター会話データセット生成ツールです。

このリポジトリは、1行のお題からペルソナ、関係性、会話進行、各ターンの発話と可視行動を生成し、JSONL として保存します。出力 JSONL の詳しい形式は [OUTPUTS.md](OUTPUTS.md) を参照してください。

## 生成パイプライン

`main.py` は、会話を以下の複数エージェントに分けて生成します。

- Persona Controller: 元のお題から、A/B の人物設定、関係性、場面制約を作る。
- Turn Controller: 次の話者、場面状態、会話の圧力、次ターンの方針を決める。
- Actor A/B: 各キャラクターとして、発話と可視行動を出力する。

DeepSeek API では以下を利用します。

- Beta endpoint: `https://api.deepseek.com/beta`。必要なら `--base-url` で変更できます。
- Strict tool schema: アクター出力を `submit_actor_turn` の function call に固定し、JSON Schema で検証します。
- KV Context Caching: 会話ごとの固定情報を system prompt に積み、同一会話内の後続呼び出しで prompt cache を効かせます。
- Thinking mode: Persona Controller と Actor は既定で thinking 有効、Turn Controller はコスト削減のため既定で thinking 無効です。

モデルと thinking は CLI から切り替えられます。

- 既定モデルは `deepseek-v4-pro` です。
- `--flash` を付けると、会話生成を `deepseek-v4-flash` だけで実行します。
- `--thinking on` は Persona / Turn Controller / Actor の thinking をすべて有効にします。
- `--thinking off` は Persona / Turn Controller / Actor の thinking をすべて無効にします。
- `--thinking default` は従来の既定値です。Persona / Actor は有効、Turn Controller は無効です。

## 入力ファイル

`format.txt` が入力です。1行に1つのお題を書きます。

```text
Aは気弱な大学生、Bは世話焼きな友人。夜の駅前で、Aが進路の不安を打ち明ける。
Aは配信者、Bは古くからの視聴者。炎上後の深夜通話で、互いに言えなかった本音を話す。
```

## セットアップ

```powershell
uv sync
```

または:

```powershell
pip install openai tqdm
```

PowerShell で API キーを設定します。

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

## テスト生成

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\test_v8.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 1 `
  --min-turns 4 `
  --max-turns 6 `
  --workers 2 `
  --reasoning-effort high `
  --progress-server
```

進捗 UI:

```text
http://127.0.0.1:8765
```

LAN 内の別端末から進捗 UI を見る場合は、待ち受け先を `0.0.0.0` にします。

```powershell
uv run python main.py `
  ... `
  --progress-server `
  --progress-host 0.0.0.0 `
  --progress-port 8765
```

## 本番寄りの生成

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 1 `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --reasoning-effort high `
  --progress-server
```

Flash だけで安く速く回す場合:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues_flash.jsonl `
  --prompt-dir .\prompts `
  --flash `
  --thinking off `
  --variations-per-line 1 `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

制約をかなり緩めた創作寄りプロンプトを使う場合:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues_lax.jsonl `
  --prompt-dir .\prompts_lax `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

`prompts_lax/` は、口調・年齢・性別・身体能力の整合性制御は維持しつつ、成人同士の合意ある露骨な性描写を含みうる関係、性的緊張、暗い感情など、創作上の自由度を削りやすいブロックを減らした版です。合意のない性行為・性的強制・性的搾取の露骨描写、現実の犯罪手順、未成年の性的描写などは境界として残しています。

お題の自動生成も lax 版にする場合は、`gen_situations.py` では `--prompt-file .\prompts_lax\situation_gen.txt`、`main.py --auto-generate-situations` では `--situation-prompt-file .\prompts_lax\situation_gen.txt` を指定してください。

## お題の自動生成

`gen_situations.py` を使うと、`format.txt` にお題を自動追加できます。重複は sha256 で除外されます。

```powershell
uv run python gen_situations.py `
  --out .\format.txt `
  --target 100 `
  --batch-size 8 `
  --seed-situation "Aは長年連れ添った夫、Bは若い後妻、過去の妻の遺品をめぐって静かに口論する。" `
  --seed-situation "Aは厳格な部活顧問、Bは練習をサボった部員、放課後の教室で対峙する。"
```

`main.py` の実行中に `format.txt` を増やすこともできます。

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

## 短い会話の続きを生成する

既存 JSONL の中で、指定ターン数に届いていない会話だけを同じ ID のまま続きから生成します。

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --prompt-dir .\prompts `
  --finish-min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

対象を検出するだけなら API を呼ばずに dry-run します。

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --finish-min-turns 25 `
  --finish-dry-run `
  --finish-dry-run-format lines
```

## 指定 ID を再生成する

既存 JSONL の指定 ID を、同じ ID のまま再生成します。

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --prompt-dir .\prompts `
  --rewrite-id persona_deepseek_triple_ja_00000007 `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

複数 ID は `--rewrite-id` を繰り返すか、カンマ区切りで指定するか、1行1IDのファイルを使います。

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --rewrite-ids-file .\rewrite_ids.txt `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

既定では JSONL 内に保存された `source.text` を使って再生成します。現在の `format.txt` の同じ行番号の内容で再生成したい場合は、次を追加します。

```powershell
--rewrite-use-current-persona-txt
```

API を呼ばずに対象確認だけを行う場合:

```powershell
--rewrite-dry-run
```

## 重複 ID を削除して再生成する

重複 ID がある場合、どちらかを残すのではなく、重複している既存レコードをすべて削除してから1件ずつ再生成できます。

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --rewrite-all-duplicates `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --reasoning-effort high `
  --progress-server
```

## 出力

主な出力は JSONL です。

- `public_transcript`: 発話テキストだけの簡易ログ。
- `public_timeline`: 発話と可視行動を含む公開イベント列。
- `persona_seed`: キャラクター、関係性、場面制約。
- `turns`: 各ターンのコントローラー出力とアクター出力を含む詳細ログ。

詳しいスキーマは [OUTPUTS.md](OUTPUTS.md) を参照してください。

## メモ

- A の行動が B から見える場合、次の B のアクタープロンプトに渡されます。
- 生成済み JSONL を安全に確認したい場合は、まず `--finish-dry-run` や `--rewrite-dry-run` を使ってください。


## English

DeepSeek triple-agent dialogue dataset generator.

This repository generates persona settings, relationships, conversation flow, per-turn utterances, and visible actions from one-line situation prompts, then writes the result as JSONL. See [OUTPUTS.md](OUTPUTS.md) for the detailed output format.

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

Model and thinking mode can be switched from the CLI:

- The default dialogue model is `deepseek-v4-pro`.
- `--flash` runs dialogue generation with `deepseek-v4-flash`.
- `--thinking on` enables thinking for Persona, Turn Controller, and Actor calls.
- `--thinking off` disables thinking for Persona, Turn Controller, and Actor calls.
- `--thinking default` keeps the legacy defaults: Persona / Actor on, Turn Controller off.

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
format.txt
gen_situations.py
prompts/
  persona_controller.txt
  turn_controller.txt
  actor.txt
  situation_gen.txt
```

## Install

```powershell
uv sync
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
  --persona-txt .\format.txt `
  --out .\test_v8.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 1 `
  --min-turns 4 `
  --max-turns 6 `
  --workers 2 `
  --reasoning-effort high `
  --progress-server
```

Flash-only run:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues_flash.jsonl `
  --prompt-dir .\prompts `
  --flash `
  --thinking off `
  --variations-per-line 1 `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

To use the less restrictive creative prompt set:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues_lax.jsonl `
  --prompt-dir .\prompts_lax `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

`prompts_lax/` keeps speech-style, age, gender, and physical-plausibility controls, while reducing blocks that limit creative freedom around consensual adult explicit sexual relationships, adult sexual tension, and darker emotions. It still keeps boundaries for explicit non-consensual sexual content, sexual coercion/exploitation, real-world criminal instructions, and sexual content involving minors.

To use the lax prompt for situation generation too, pass `--prompt-file .\prompts_lax\situation_gen.txt` to `gen_situations.py`, or `--situation-prompt-file .\prompts_lax\situation_gen.txt` when using `main.py --auto-generate-situations`.

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
  --out .\persona_dialogues.jsonl `
  --prompt-dir .\prompts `
  --variations-per-line 1 `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
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
  --finish-min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
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
  --finish-min-turns 25 `
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
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --progress-server
```

You can repeat `--rewrite-id`, pass comma-separated IDs, or put one ID per line
in `--rewrite-ids-file`. Use `--rewrite-dry-run` first to verify the matched
records without calling the API or rewriting the JSONL.

By default, rewrite uses the `source.text` saved inside the JSONL record. If
you edited `format.txt` and want rewrite to use the current text at the saved
`source.line_number`, add:

```powershell
--rewrite-use-current-persona-txt
```

If an ID is duplicated and you want to delete all existing copies before
regenerating exactly one replacement:

```powershell
uv run python main.py `
  --persona-txt .\format.txt `
  --out .\persona_dialogues.jsonl `
  --rewrite-all-duplicates `
  --min-turns 25 `
  --max-turns 50 `
  --workers 9 `
  --reasoning-effort high `
  --progress-server
```

## Notes

- `public_timeline` contains both utterances and `visible_action`.
- If A performs an action visible to B, B receives it in the next actor prompt.
- Actor output includes `physical_action` as a free-text visible action string.
- Turn Controller thinking is disabled by default to reduce cost.
- Persona Controller and Actors have thinking enabled by default.
- Use `--finish-dry-run` or `--rewrite-dry-run` before rewriting existing JSONL files when you only want to inspect the targets.
