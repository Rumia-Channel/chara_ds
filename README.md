# PersonaCast-JA

[English](#english)

The generated data is [heare](https://huggingface.co/datasets/RumiaChannel/PersonaCast-JA).

DeepSeek を使った、日本語キャラクター会話データセット生成ツールです。

このリポジトリは、1行のお題からペルソナ、関係性、会話進行、各ターンの発話と可視行動を生成し、JSONL として保存します。出力 JSONL の詳しい形式は [OUTPUTS.md](OUTPUTS.md) を参照してください。

## 生成パイプライン

`main.py` は、会話を以下の複数エージェントに分けて生成します。

- Persona Controller: 元のお題から、A/B の人物設定、関係性、場面制約を作る。
- Turn Controller: 次の話者、場面状態、会話の圧力、次ターンの方針を決め、長期状態を `state_memory` に保持する。
- Actor A/B: 各キャラクターとして、発話と可視行動を出力する。
- Actor Guard: 任意の第三者監視役。既定では `deepseek-v4-pro` で年齢・身体能力・口調の不整合を判定し、問題があれば「何がどうおかしいか」をActorへ返して同じターンを書き直させる。
- Age/Gender Norms: `age_gender_norms/` の属性別 JSON から必要な年齢・性別・特殊属性・感情反応ガイドだけを Persona Controller と Actor Guard に渡し、普通寄りの人物や特殊属性の口調が不自然に崩れないようにする。

DeepSeek API では以下を利用します。

- Beta endpoint: `https://api.deepseek.com/beta`。必要なら `--base-url` で変更できます。
- Strict tool schema: Turn Controller 出力を `submit_turn_control`、アクター出力を `submit_actor_turn` の function call に固定し、JSON Schema で検証します。
- KV Context Caching: 会話ごとの固定情報を system prompt に積み、同一会話内の後続呼び出しで prompt cache を効かせます。
- Thinking mode: Persona Controller と Actor は既定で thinking 有効、Turn Controller はコスト削減のため既定で thinking 無効です。

モデルと thinking は CLI から切り替えられます。

- 既定モデルは `deepseek-v4-pro` です。
- `--flash` を付けると、会話生成を `deepseek-v4-flash` だけで実行します。
- `--thinking on` は Persona / Turn Controller / Actor の thinking をすべて有効にします。
- `--thinking off` は Persona / Turn Controller / Actor の thinking をすべて無効にします。
- `--thinking default` は従来の既定値です。Persona / Actor は有効、Turn Controller は無効です。
- `--actor-guard` を付けると、各Actor出力後に第三者視点の監視を挟みます。既定は `--actor-guard-model deepseek-v4-pro --actor-guard-thinking off` です。
- `--persona-max-tokens` / `--controller-max-tokens` / `--actor-max-tokens` / `--actor-guard-max-tokens` / `--situation-max-tokens` は既定で DeepSeek V4 の最大出力 384K に合わせています。`0` を指定すると `max_tokens` を省略します。

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

`--prompt-dir .\prompts_lax` を指定すると、会話生成用の `persona_controller.txt` / `turn_controller.txt` / `actor.txt` / `actor_guard.txt` だけでなく、`main.py --auto-generate-situations` の `situation_gen.txt` も同じディレクトリから読みます。`gen_situations.py` 単体でも `--prompt-dir .\prompts_lax` を指定すれば lax 版の `situation_gen.txt` を使います。

`age_gender_norms/` が prompt directory にある場合は、Persona Controller と Actor Guard の共通参照データとして読み込まれます。これは lax/通常で分ける創作強度ではなく、年齢・性別・特殊属性ごとの一般的な口調・感情反応基準です。

- `age_gender_norms/index.json`: 利用可能な属性一覧。例: 女子中学生、女子高校生、男子中学生、成人女性、メスガキ、ロリババア。
- `age_gender_norms/*.json`: 属性ごとの一人称・二人称、喜び、悲しみ、照れ、困惑、嫉妬、安心、悲鳴、痛み、制止、怒り語彙、避ける粗野表現、Actor Guard の判定規則。
- Persona Controller には入力文から近い属性だけを `age_gender_norms_selected` として渡します。
- Actor Guard には speaker の persona から近い属性だけを `age_gender_norms_selected` として渡します。
- `age_gender_norms.txt` は旧形式の fallback です。`age_gender_norms/index.json` がある場合は JSON 形式が優先されます。

これにより、普通寄りの女子中学生が攻撃された直後に成人男性的な罵倒へ飛ぶ崩れだけでなく、悲しい時・うれしい時・照れた時の年齢不相応な反応、メスガキ属性の成人的な性的挑発化、ロリババア属性の単なる幼児化/老人化も抑えます。全属性データを毎回 payload に積まず、必要な資料だけを渡します。

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
`--resume` と併用した場合は、短い出力レコードを turn cache に戻して再開対象にし、成功後に同じ ID のレコードを置換します。

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

## Resume と turn cache

`--resume` は、`--out` に既に書かれた完了済み ID を飛ばし、未完了の会話を `<out>.cache` から途中再開します。`--out` のファイル名が違うと完了済み判定はできません。

turn cache は既定で `<out>.cache` に保存されます。会話成功後も cache は残ります。消したい場合だけ `--delete-turn-cache-on-success` を指定します。

既存 cache を上書きする場合、既定で古い cache は `<out>.cache\backups\YYYYMMDD_HHMMSS\` に退避されます。日時はその実行で最初に cache バックアップが作成された瞬間のものです。同じ実行中、同じ conversation の cache は最初の 1 回だけ退避されます。この退避を止めたい場合だけ `--no-turn-cache-backup` を指定します。

`--resume` 時は、cache が使われた場合に `turn_cache_used`、使われなかった場合に `turn_cache_not_used` が stderr に出ます。signature 不一致や missing の切り分けに使えます。

Turn Controller の新しい `state_memory` tool を使わず、旧 JSON Controller 経路で再開したい場合は `--disable-state-memory-tool` を指定します。プロンプトや schema 変更前の古い backup cache を明示的に採用する場合だけ、`--resume --disable-state-memory-tool --resume-accept-stale-cache` を併用します。この場合、signature mismatch の cache も採用されるため、同じ入力・seed・turn 範囲の backup を指定している時だけ使ってください。

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
- **Strict tool schema** (`function.strict = true`) — the Turn Controller emits
  control through `submit_turn_control`, and the actor emits its turn through
  `submit_actor_turn`; the API enforces the JSON Schema server-side.
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
- `--actor-guard` adds a third-person judge after each Actor turn. By default it uses `--actor-guard-model deepseek-v4-pro --actor-guard-thinking off`.
- `--persona-max-tokens` / `--controller-max-tokens` / `--actor-max-tokens` / `--actor-guard-max-tokens` / `--situation-max-tokens` default to DeepSeek V4 max output, 384K. Set `0` to omit `max_tokens`.

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
  actor_guard.txt
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

When `--prompt-dir .\prompts_lax` is set, dialogue generation reads `persona_controller.txt`, `turn_controller.txt`, `actor.txt`, and `actor_guard.txt` from that directory, and `main.py --auto-generate-situations` also reads `situation_gen.txt` from it. For standalone `gen_situations.py`, pass `--prompt-dir .\prompts_lax` to use the lax situation prompt.

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

When combined with `--resume`, short records already present in `--out` are
copied back into the per-turn cache and scheduled as resume work. Successful
results replace the same IDs in the JSONL instead of appending duplicates.

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

## Resume and turn cache

`--resume` skips IDs already present in `--out` and resumes unfinished
conversations from `<out>.cache`. The `--out` path must match the JSONL that
contains the completed IDs.

Per-turn caches are kept after successful writes by default. Add
`--delete-turn-cache-on-success` only when you want successful cache files
removed.

When an existing cache file is about to be overwritten, it is backed up by
default under `<out>.cache\backups\YYYYMMDD_HHMMSS\`. The timestamp is taken
when the first cache backup is created in that run. During the same run, each
conversation cache is backed up only once. Add `--no-turn-cache-backup` to
disable that backup.

During `--resume`, stderr reports `turn_cache_used` or `turn_cache_not_used`
for each scheduled conversation so missing caches and signature mismatches are visible.

Use `--disable-state-memory-tool` to resume with the legacy JSON Turn
Controller path instead of the new `state_memory` tool path. To explicitly use
an old backup cache after prompt/schema changes, combine
`--resume --disable-state-memory-tool --resume-accept-stale-cache`. This accepts
signature-mismatched caches, so use it only when you intentionally restored the
matching old cache for the same input, seed, and turn range.

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
