# Output Format / 出力形式

[English](#english)

この文書は、このリポジトリの `main.py` が生成する JSONL 出力の形式を説明します。

生成データは、DeepSeek を用いて合成された日本語のマルチエージェント会話です。生成パイプラインでは、ペルソナ制御、ターン制御、2人のキャラクター発話エージェントを分け、キャラクター設定、関係性、場面進行、各ターンの発話と可視行動を段階的に生成しています。

ファイル形式は JSON Lines (`.jsonl`) です。1行が1つの完全な会話データです。

```text
1行 = 1つのJSONオブジェクト = 1会話
```

### 出力データの使い道

この出力 JSONL は、単なる一問一答ではなく、複数ターンにわたる人物関係、感情変化、場面進行を扱う用途を想定しています。

主な利用例:

- 日本語キャラクター会話モデルの fine-tuning。
- ロールプレイ、ノベルゲーム、チャットボット向けの長めの対話データ。
- キャラクターごとの口調、価値観、弱点、隠し情報を維持する対話生成の学習。
- 発話だけでなく、可視行動を含む会話シーン生成の学習。
- ペルソナ、関係性、シーン制約を条件として会話を生成するタスク。
- 会話の圧力、感情の押し引き、終了条件などを含む対話制御の分析。

用途に応じて、発話だけを使う場合は `public_transcript`、発話と行動を使う場合は `public_timeline`、制御信号まで使う場合は `persona_seed` と `turns` を参照してください。

### 基本メタデータ

トップレベルの主なフィールド:

| フィールド | 型 | 説明 |
|---|---:|---|
| `id` | string | 会話ID。例: `persona_deepseek_triple_ja_00000001`。 |
| `dataset` | string | データセット名。現在値: `persona_controlled_deepseek_triple_agent_ja`。 |
| `schema_version` | string | スキーマバージョン。現在値: `13.2`。 |
| `created_at` | string | UTC の ISO タイムスタンプ。 |
| `synthetic` | bool | 合成データを示す値。常に `true`。 |
| `language` | string | 言語コード。現在値: `ja`。 |
| `source` | object | 会話生成に使われた元のお題行。 |
| `agents` | object | ペルソナ、コントローラー、アクターに使った生成モデル情報。 |
| `generation_config` | object | ターン数、シード、サンプリング、最大トークンなどの生成設定。 |
| `prompt_hashes` | object | 生成時に使ったプロンプトファイルの SHA-256 ハッシュ。 |
| `persona_generation` | object | ペルソナコントローラーの生出力と使用量メタデータ。 |
| `persona_seed` | object | キャラクター、関係性、場面制約を構造化した設定。 |
| `turns` | array | 各ターンのコントローラー出力、アクター出力、公開イベント。 |
| `public_timeline` | array | 公開される発話と可視行動の時系列。 |
| `public_transcript` | array | 発話テキストだけの簡易会話ログ。 |
| `usage` | object | ペルソナ、ターンコントローラー、アクター、任意のアクター監視役のトークン使用量。 |
| `hashes` | object | source、persona、timeline、conversation の SHA-256 ハッシュ。 |

### Source

`source` は、会話生成に使われた元のお題を表します。

```json
{
  "type": "line_txt",
  "filename": "./format.txt",
  "line_number": 1,
  "variation": 1,
  "text": "Aは気弱な大学生、Bは世話焼きな友人。...",
  "sha256": "..."
}
```

`source.text` は1行のお題文です。`line_number` は生成時点の元テキストファイル内の行番号です。

### Persona Seed

`persona_seed` は、会話エージェントが使用した構造化設定です。

主なサブフィールド:

| フィールド | 説明 |
|---|---|
| `source_summary` | 元のお題の短い要約。 |
| `safety_transformations` | 匿名化や抽象化などの変換メモ。 |
| `global_style` | ジャンル、地域、トーン。 |
| `characters.A` / `characters.B` | 各話者のキャラクタープロフィール。 |
| `relationship` | 関係性、過去、距離感、隠れた緊張。 |
| `norm_profile_ids` | Persona 生成時に A/B それぞれへ適用・参照した `age_gender_norms` の id。 |
| `explicit_overrides_from_user_txt` | 元お題の明示指定が一般的な age/gender norms より優先された点。 |
| `scenario_constraints` | 場所、許可される話題や行動、避ける話題、衣装・装備・小道具・位置関係の連続性メモ、文体メモ、終了条件、必要に応じたターン配分ヒント。 |

`scenario_constraints` には、必要に応じて `continuity_notes` が含まれます。これは、初期状態の衣装、装備、小道具、家具、距離、手元にある/ない物などを記録するためのメモです。

入力に結末が明示されている場合は、`ending_condition` と `turn_budget_hint` が含まれることがあります。`turn_budget_hint` は、`target_turns` の範囲内で途中ぶつ切りを避け、終盤の山場と着地に十分な余白を残すための補助情報です。

例:

```json
{
  "setting": "魔術学院の空き教室",
  "allowed_actions": ["腕をつかむ", "椅子を引く", "床に落ちた杖へ視線を向ける"],
  "continuity_notes": "A のローブは椅子の背に掛かっており、A は現在それを着ていない。B の杖は床に落ちていて、すぐには手元にない。",
  "conversation_style_notes": "喧嘩になっても、現在着ていないローブをつかみ合う描写は避ける。",
  "ending_condition": "互いの本音が表に出て、会話が自然に区切れるところまで。",
  "turn_budget_hint": {
    "has_explicit_ending": true,
    "minimum_required_turns": 120,
    "recommended_target_turns": 180,
    "milestones": ["導入", "対立の激化", "終盤の山場", "着地"],
    "pace_notes": "終盤の決着まで急ぎすぎず、最後の山場に十分なターンを残す。"
  }
}
```

各キャラクターには通常、以下のフィールドが含まれます。

| フィールド | 説明 |
|---|---|
| `role` | 場面内での役割。 |
| `age_band` | `child`、`early_teen`、`teen`、`late_teen`、`young_adult`、`adult`、`20s`、`30s`、`40s`、`50s`、`60s+`、`unspecified` などの年齢帯。 |
| `gender` | `female`、`male`、`nonbinary`、`unspecified` など。 |
| `personality` | 性格。 |
| `speech_style` | 口調、一人称、二人称、語尾、罵倒語、例文、禁止語句。 |
| `values` | キャラクターが重視する価値観。 |
| `weaknesses` | 弱点や脆さ。 |
| `default_goal` | 場面内での基本目的。 |
| `private_background` | 相手が自動的には知らない私的背景。 |
| `public_profile` | 公開されていて自然に知られうるプロフィール。 |
| `forbidden_disclosures` | 会話中に不用意に明かすべきでない情報。 |

### Turns

`turns` は、各会話ターンの完全な生成記録です。
`actor_guard` は `--actor-guard` を指定した場合のみ含まれます。Guard が不合格を返した場合、その `reason_ja` と `suggested_fix_ja` は次の Actor 呼び出しに `actor_guard_feedback` として渡され、同じターンの書き直しに使われます。
`actor_guard.content.filler_analysis` には、Actor Guard が判定した現在発話の先頭フィラー/口癖 family、同一話者での連続回数、直近発話内の反復回数、反復問題かどうかが入ります。この分類は Python の固定リストではなく Guard の判定結果として public timeline に保存され、次ターン以降の反復管理に使われます。
`--sakura-guard` を指定した場合、`actor_guard.provider` と `agents.actor_guard.provider` は `sakura` になり、`SAKURA_API_KEY` と `--sakura-base-url` / `--sakura-guard-model` で SAKURA AI Engine に接続します。
`--conversation-audit` を指定した場合、生成完了後に `conversation_audit` が追加されます。`content.overall_score`、`content.dimension_scores`、`content.turn_issues`、`content.recommended_action` に、会話全体の横断監査結果が入ります。

各要素はおおむね以下の形です。

```json
{
  "turn": 1,
  "controller": {
    "content": {
      "turn_control": {
        "next_speaker": "A",
        "scene_state": "A のローブは椅子の背に掛かったまま。B の杖は床に落ちており、B の手元にはない。互いに机を挟んで立っている。",
        "conversation_pressure": "medium",
        "public_event": "...",
        "hidden_controller_intent": "...",
        "directive_for_next_speaker": {
          "emotional_push": "...",
          "local_goal": "...",
          "constraint": "...",
          "suggested_action": "explain",
          "physical_action_hint": "...",
          "avoid": "..."
        },
        "expected_next_effect": "...",
        "should_end": false,
        "end_reason": ""
      }
    },
    "reasoning_content": "...",
    "usage": {},
    "thinking_enabled": false
  },
  "actor": {
    "speaker": "A",
    "content": {
      "speaker": "A",
      "thinking_trace_ja": "...",
      "character_thought": "...",
      "physical_action": "...",
      "public_utterance": "...",
      "subtext": "..."
    },
    "reasoning_content": "...",
    "usage": {},
    "thinking_enabled": true
  },
  "actor_guard": {
    "content": {
      "pass": true,
      "severity": "ok",
      "reason_ja": "問題なし",
      "suggested_fix_ja": ""
    },
    "reasoning_content": null,
    "usage": {},
    "thinking_enabled": false,
    "model": "deepseek-v4-pro"
  },
  "public_event": {
    "turn": 1,
    "speaker": "A",
    "utterance": "...",
    "visible_action": "..."
  }
}
```

補足:

- `controller.content.turn_control` は、次の話者と会話の展開方針を決めます。
- `controller.grand_controller.content` は、Turn Controller の直前に作られる大局方針です。心理的有利、肉体的有利、攻防の流れ、揺り戻しが必要か、終盤までのペース配分を保持し、Turn Controller の短期判断が一方的な展開へ drift しないようにします。
- `controller.content.turn_control.scene_state` は、その時点の場面状態です。会話圧だけでなく、衣装、装備、小道具、家具、距離、手元にある/ない物などの連続性を含む場合があります。
- `controller.content.turn_control.state_memory` は、長期会話・長期戦闘用の構造化メモです。人物状態、環境、小道具/武器、負傷/疲労、関係性、会話で決まったこと、直近数ターンの発話要点、各話者の約束/拒否、未回収の話題、確定事実、避けるべき矛盾を保持します。
- `scene_state` は各ターンで更新されます。たとえば、ローブを椅子に掛けた、上着を脱いだ、武器を床に落とした、机を挟んで距離がある、などの状態が次ターン以降の行動制御に使われます。
- `actor.content.public_utterance` は、実際に発話された台詞です。
- `actor.content.physical_action` は、そのターンの可視行動または身体動作です。
- `actor.content.character_thought`、`thinking_trace_ja`、`subtext` は内部生成用の非公開フィールドです。
- `reasoning_content` はモデルの推論またはデバッグ用メタデータです。用途によっては除外してください。

### Public Timeline

`public_timeline` は、両キャラクターから見える公開イベントの時系列です。

```json
[
  {
    "turn": 1,
    "speaker": "A",
    "utterance": "...",
    "visible_action": "..."
  }
]
```

発話と可視行動を含めて場面を復元したい場合は、このフィールドが適しています。

### Public Transcript

`public_transcript` は、発話テキストだけの簡易ログです。

```json
[
  {
    "speaker": "A",
    "text": "..."
  },
  {
    "speaker": "B",
    "text": "..."
  }
]
```

会話文だけを使う fine-tuning や簡易表示には、このフィールドが適しています。

### Generation Config

`generation_config` は生成時の設定を記録します。

主なフィールド:

| フィールド | 説明 |
|---|---|
| `target_turns` | ランダムに選ばれた目標ターン数。入力に明示的な結末がある場合は、結末まで到達しやすいよう上振れ補正されることがあります。 |
| `actual_turns` | 実際に生成された公開ターン数。 |
| `min_turns` / `max_turns` | 生成時に指定されたターン数範囲。 |
| `seed` | 生成シード。 |
| `variation` | 元お題行に対するバリエーション番号。 |
| `controller_temperature` / `controller_top_p` | thinking 無効時のターンコントローラーのサンプリング設定。 |
| `actor_guard_enabled` / `actor_guard_model` / `actor_guard_provider` | `--actor-guard` 使用時の第三者監視役設定。`--sakura-guard` 時は provider が `sakura`。 |
| `conversation_audit_enabled` / `conversation_audit_model` / `conversation_audit_provider` | `--conversation-audit` 使用時の完成会話監査設定。 |
| `max_tokens_policy` | ペルソナ、コントローラー、アクター、アクターガード呼び出しの最大トークン設定。既定は DeepSeek V4 最大出力の 384K で、`0` または `None` は API へ `max_tokens` を送らないことを意味します。 |

`prompt_hashes.age_gender_norms_sha256` は、`prompt_dir/age_gender_norms/` がある場合はその JSON 群、なければ `prompt_dir/age_gender_norms.txt` の hash です。Persona Controller と Actor Guard には、全データではなく現在の入力・speaker に近い属性だけが `age_gender_norms_selected` として渡されます。年齢・性別・特殊属性ごとの一人称、二人称、喜び、悲しみ、照れ、困惑、痛み、制止、怒り語彙の基準に使われます。

### Turn Cache

turn cache は JSONL レコード本体には含まれない sidecar ファイルです。既定では `<out>.cache/<conversation_id>.json` に、persona controller 完了後と各ターン成功後の途中状態が保存されます。

`--resume` 時に signature が一致すると、その cache から途中再開します。signature にはプロンプト hash、モデル、seed、turn 数、thinking 設定、max token 設定などが含まれます。

既定では成功後も cache は残ります。既存 cache を上書きする場合は、古い cache が `<out>.cache/backups/YYYYMMDD_HHMMSS/` に退避されます。この日時はその実行で最初に cache バックアップが作成された瞬間のものです。同じ実行中、同じ conversation の cache は最初の 1 回だけ退避されます。`--delete-turn-cache-on-success` で成功後削除、`--no-turn-cache-backup` で上書き前退避を無効化できます。

古い backup cache を使うために Turn Controller の `state_memory` tool を無効化する場合は `--disable-state-memory-tool` を指定します。プロンプトや schema 変更で signature が一致しない backup cache を明示的に採用する場合だけ、`--resume --disable-state-memory-tool --resume-accept-stale-cache` を使います。

### 推奨される使い方

会話文だけを使う場合:

```python
example["public_transcript"]
```

会話文と可視行動を使う場合:

```python
example["public_timeline"]
```

制御信号を含めた分析や学習に使う場合:

```python
example["persona_seed"]
example["turns"]
```

### 最小読み込み例

```python
import json

with open("persona_dialogues.jsonl", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        transcript = ex["public_transcript"]
        print(ex["id"], len(transcript))
```

### 生成コストの参考値

このリポジトリで生成したデータの一例では、合成に `deepseek-v4-pro` を使用しました。生成時には DeepSeek の 75% OFF キャンペーン価格を利用しています。

追走者向けの参考値:

| 項目 | 値 |
|---|---:|
| 使用モデル | `deepseek-v4-pro` |
| 使用トークン数 | 約 340M tokens |
| 実コスト | 115.66 USD |
| 価格条件 | DeepSeek 75% OFF キャンペーン適用時 |

この金額は生成時点のキャンペーン価格、プロンプト構成、再試行回数、失敗率、thinking/reasoning 設定に依存します。同じ件数を再生成しても、通常価格や異なる設定ではコストが変わります。

### 注意点

- このデータセットは DeepSeek を用いた LLM エージェントによって生成された合成データです。
- 一部のレコードには `subtext`、`character_thought`、`reasoning_content` などの内部/非公開フィールドが含まれます。
- `public_transcript` には可視行動が含まれません。行動も必要な場合は `public_timeline` を使ってください。
- 衣装・装備・小道具・位置関係の連続性を利用する場合は、`turns[].controller.content.turn_control.scene_state`、`turns[].controller.content.turn_control.state_memory`、`persona_seed.scenario_constraints.continuity_notes` を参照してください。`public_timeline` には発話と可視行動のみが入ります。
- `source.text` は後から編集された `format.txt` と異なる場合があります。JSONL には生成時点のお題文が保存されます。
- トークン使用量フィールドはプロバイダー依存のメタデータとして扱ってください。

## English

This document describes the JSONL output format generated by this repository's `main.py`.

The generated records are synthetic Japanese multi-agent dialogues created with DeepSeek. The generation pipeline separates persona control, turn control, and two character actor agents, then generates character settings, relationships, scene progression, each turn's utterance, and visible action in stages.

Each file is JSON Lines (`.jsonl`). Each line is one complete conversation.

```text
one line = one JSON object = one conversation
```

### How To Use The Output

The output JSONL is designed for use cases that require multi-turn character interaction, relationship dynamics, emotional progression, and scene development, rather than simple single-turn question answering.

Typical use cases:

- Fine-tuning Japanese character dialogue models.
- Longer role-play, visual-novel, game, or chatbot conversations.
- Training models to preserve character-specific speech style, values, weaknesses, and hidden information.
- Generating dialogue scenes that include both spoken lines and visible actions.
- Conditional dialogue generation from persona, relationship, and scene constraints.
- Analysis of dialogue control signals such as conversation pressure, emotional push, and ending conditions.

Use `public_transcript` for dialogue-only tasks, `public_timeline` for dialogue plus visible actions, and `persona_seed` plus `turns` when control signals or full generation traces are needed.

### Basic Metadata

Top-level fields:

| Field | Type | Description |
|---|---:|---|
| `id` | string | Stable conversation ID, e.g. `persona_deepseek_triple_ja_00000001`. |
| `dataset` | string | Dataset name. Current value: `persona_controlled_deepseek_triple_agent_ja`. |
| `schema_version` | string | Schema version. Current value: `13.2`. |
| `created_at` | string | UTC ISO timestamp. |
| `synthetic` | bool | Always `true`; generated data. |
| `language` | string | Language code. Current value: `ja`. |
| `source` | object | Source situation line used to generate the conversation. |
| `agents` | object | Generator model metadata for persona, controller, and actors. |
| `generation_config` | object | Turn count, seed, sampling, and max-token settings. |
| `prompt_hashes` | object | SHA-256 hashes of prompt files used for generation. |
| `persona_generation` | object | Raw Persona Controller output and usage metadata. |
| `persona_seed` | object | Structured character, relationship, and scenario settings. |
| `turns` | array | Full per-turn controller and actor records. |
| `public_timeline` | array | Publicly visible dialogue/action sequence. |
| `public_transcript` | array | Simplified speaker/text transcript. |
| `usage` | object | Token usage for persona controller, turn controller, actors, and optional actor guard. |
| `hashes` | object | SHA-256 hashes for source, persona, timeline, and conversation. |

### Source

`source` identifies the source situation.

```json
{
  "type": "line_txt",
  "filename": "./format.txt",
  "line_number": 1,
  "variation": 1,
  "text": "Aは気弱な大学生、Bは世話焼きな友人。...",
  "sha256": "..."
}
```

`source.text` is the one-line scenario prompt. `line_number` refers to the line in the source text file at generation time.

### Persona Seed

`persona_seed` is the structured setup used by the dialogue agents.

Important subfields:

| Field | Description |
|---|---|
| `source_summary` | Short summary of the source situation. |
| `safety_transformations` | Notes about anonymization or abstraction. |
| `global_style` | Genre, locale, and tone. |
| `characters.A` / `characters.B` | Character profiles for each speaker. |
| `relationship` | Relationship type, history, distance, and hidden tension. |
| `norm_profile_ids` | `age_gender_norms` ids applied or referenced for A/B during persona generation. |
| `explicit_overrides_from_user_txt` | Source-text details that intentionally override the general age/gender norms. |
| `scenario_constraints` | Setting, allowed topics/actions, avoid topics, continuity notes for clothing/equipment/props/positions, style notes, ending condition, and optional turn-budget hints. |

`scenario_constraints` may include `continuity_notes`. This field records initial continuity-relevant state such as clothing, equipment, props, furniture, distance, and items that are or are not currently at hand.

When the input explicitly describes an ending, `scenario_constraints` may include `ending_condition` and `turn_budget_hint`. `turn_budget_hint` helps the Turn Controller pace the scene toward the requested ending within `target_turns` instead of cutting off in the middle.

Example:

```json
{
  "setting": "an empty classroom at a magic academy",
  "allowed_actions": ["grab an arm", "pull a chair", "glance at the wand on the floor"],
  "continuity_notes": "A's robe is hanging on the back of a chair, so A is not currently wearing it. B's wand has fallen to the floor and is not immediately in B's hand.",
  "conversation_style_notes": "If a fight starts, avoid describing both characters as grabbing each other's robes when the robe is not being worn.",
  "ending_condition": "Reach the point where both characters' real feelings are exposed and the scene can naturally close.",
  "turn_budget_hint": {
    "has_explicit_ending": true,
    "minimum_required_turns": 120,
    "recommended_target_turns": 180,
    "milestones": ["opening", "escalation", "late climax", "landing"],
    "pace_notes": "Do not rush the setup; leave enough turns for the final climax and landing."
  }
}
```

Each character usually contains:

| Field | Description |
|---|---|
| `role` | Character role in the scene. |
| `age_band` | Age band such as `child`, `early_teen`, `teen`, `late_teen`, `young_adult`, `adult`, `20s`, `30s`, `40s`, `50s`, `60s+`, or `unspecified`. |
| `gender` | `female`, `male`, `nonbinary`, or `unspecified`. |
| `personality` | Character personality description. |
| `speech_style` | Register, first person, second person, sentence endings, swear words, examples, forbidden phrases. |
| `values` | Values important to the character. |
| `weaknesses` | Character weaknesses or vulnerabilities. |
| `default_goal` | Default goal in the scene. |
| `private_background` | Private information not automatically known to the other character. |
| `public_profile` | Publicly knowable profile. |
| `forbidden_disclosures` | Information that should not be casually revealed in dialogue. |

### Turns

`turns` contains the full generation trace for every dialogue turn.
`actor_guard` is present only when `--actor-guard` is enabled. If the guard fails an output, its `reason_ja` and `suggested_fix_ja` are passed to the next Actor call as `actor_guard_feedback` for rewriting the same turn.
`actor_guard.content.filler_analysis` stores the Actor Guard's classification of the current leading filler / verbal habit family, same-speaker consecutive count, recent repetition count, and whether it is a repetition problem. The family classification is produced by the Guard rather than a Python hard-coded list, then stored in the public timeline for later turns.
With `--sakura-guard`, `actor_guard.provider` and `agents.actor_guard.provider` are `sakura`; the runner uses `SAKURA_API_KEY` plus `--sakura-base-url` / `--sakura-guard-model`.
With `--conversation-audit`, the final record includes `conversation_audit`. `content.overall_score`, `content.dimension_scores`, `content.turn_issues`, and `content.recommended_action` contain the full-conversation audit result.

Each element has approximately this shape:

```json
{
  "turn": 1,
  "controller": {
    "content": {
      "turn_control": {
        "next_speaker": "A",
        "scene_state": "A's robe is still hanging on the back of the chair. B's wand is on the floor and not in B's hand. They are standing with a desk between them.",
        "conversation_pressure": "medium",
        "public_event": "...",
        "hidden_controller_intent": "...",
        "directive_for_next_speaker": {
          "emotional_push": "...",
          "local_goal": "...",
          "constraint": "...",
          "suggested_action": "explain",
          "physical_action_hint": "...",
          "avoid": "..."
        },
        "expected_next_effect": "...",
        "should_end": false,
        "end_reason": ""
      }
    },
    "reasoning_content": "...",
    "usage": {},
    "thinking_enabled": false
  },
  "actor": {
    "speaker": "A",
    "content": {
      "speaker": "A",
      "thinking_trace_ja": "...",
      "character_thought": "...",
      "physical_action": "...",
      "public_utterance": "...",
      "subtext": "..."
    },
    "reasoning_content": "...",
    "usage": {},
    "thinking_enabled": true
  },
  "actor_guard": {
    "content": {
      "pass": true,
      "severity": "ok",
      "reason_ja": "問題なし",
      "suggested_fix_ja": ""
    },
    "reasoning_content": null,
    "usage": {},
    "thinking_enabled": false,
    "model": "deepseek-v4-pro"
  },
  "public_event": {
    "turn": 1,
    "speaker": "A",
    "utterance": "...",
    "visible_action": "..."
  }
}
```

Notes:

- `controller.content.turn_control` decides the next speaker and dramatic direction.
- `controller.grand_controller.content` is the grand strategy generated immediately before Turn Controller. It tracks psychological advantage, physical advantage, momentum, whether a swing-back is needed, and pacing toward the ending so short-term turn control does not drift into an unintended one-sided progression.
- `controller.content.turn_control.scene_state` is the current scene state. It may include continuity for clothing, equipment, props, furniture, distance, and items that are or are not currently at hand.
- `controller.content.turn_control.state_memory` is structured memory for long conversations and long fights. It tracks participant status, environment, props/weapons, injuries/fatigue, relationship state, conversation decisions, recent dialogue facts, speaker commitments, open threads, established facts, and contradictions to avoid.
- `scene_state` is updated every turn. For example, a robe hanging on a chair, a jacket being removed, a weapon falling to the floor, or a desk separating the characters can constrain later physical actions.
- `actor.content.public_utterance` is the spoken line.
- `actor.content.physical_action` is visible or bodily action for that turn.
- `actor.content.character_thought`, `thinking_trace_ja`, and `subtext` are private/internal generation fields, not public dialogue.
- `reasoning_content` is model reasoning/debug metadata. Downstream users may want to ignore or remove it depending on their use case.

### Public Timeline

`public_timeline` is the public sequence visible to both characters.

```json
[
  {
    "turn": 1,
    "speaker": "A",
    "utterance": "...",
    "visible_action": "..."
  }
]
```

This is usually the best field for reconstructing the scene with visible action.

### Public Transcript

`public_transcript` is a simplified text-only transcript.

```json
[
  {
    "speaker": "A",
    "text": "..."
  },
  {
    "speaker": "B",
    "text": "..."
  }
]
```

This is usually the best field for dialogue-only fine-tuning or quick display.

### Generation Config

`generation_config` records generation settings.

Important fields:

| Field | Description |
|---|---|
| `target_turns` | Randomly selected target turn count. If the source text explicitly includes an ending, this may be raised within the requested range so the scene has room to reach that ending. |
| `actual_turns` | Actual number of generated public turns. |
| `min_turns` / `max_turns` | Turn range requested at generation time. |
| `seed` | Generation seed. |
| `variation` | Variation index for the source line. |
| `controller_temperature` / `controller_top_p` | Turn controller sampling settings when thinking is disabled. |
| `actor_guard_enabled` / `actor_guard_model` / `actor_guard_provider` | Third-person actor guard settings when `--actor-guard` is used. provider is `sakura` with `--sakura-guard`. |
| `conversation_audit_enabled` / `conversation_audit_model` / `conversation_audit_provider` | Full-conversation audit settings when `--conversation-audit` is used. |
| `max_tokens_policy` | Max-token settings for persona, controller, actor, and actor-guard calls. Defaults use DeepSeek V4 max output, 384K. `0` or `None` means `max_tokens` was omitted for that call. |

`prompt_hashes.age_gender_norms_sha256` is the hash of `prompt_dir/age_gender_norms/` JSON files when present, otherwise `prompt_dir/age_gender_norms.txt`. Persona Controller and Actor Guard receive only the relevant selected snippets as `age_gender_norms_selected`, not the full dataset. The snippets guide first/second person choices, joy, sadness, embarrassment, confusion, pain reactions, refusals, anger vocabulary, and special archetypes.

### Turn Cache

Turn cache files are sidecar files, not part of the JSONL record. By default, partial state is written to `<out>.cache/<conversation_id>.json` after Persona Controller finishes and after every successful turn.

On `--resume`, a cache is used only when its signature matches the current run. The signature includes prompt hashes, model, seed, turn counts, thinking settings, max-token settings, and related generation inputs.

Caches are kept after successful writes by default. When an existing cache is overwritten, the old file is backed up under `<out>.cache/backups/YYYYMMDD_HHMMSS/`. The timestamp is taken when the first cache backup is created in that run. During the same run, each conversation cache is backed up only once. Use `--delete-turn-cache-on-success` to remove successful caches and `--no-turn-cache-backup` to disable overwrite backups.

Use `--disable-state-memory-tool` to run the legacy JSON Turn Controller path for old backup caches. Use `--resume --disable-state-memory-tool --resume-accept-stale-cache` only when you intentionally want to accept a signature-mismatched backup cache after prompt/schema changes.

### Recommended Views

For dialogue-only use:

```python
example["public_transcript"]
```

For dialogue plus visible actions:

```python
example["public_timeline"]
```

For full training or analysis with control signals:

```python
example["persona_seed"]
example["turns"]
```

### Minimal Loading Example

```python
import json

with open("persona_dialogues.jsonl", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        transcript = ex["public_transcript"]
        print(ex["id"], len(transcript))
```

### Generation Cost Reference

One generated dataset from this repository was synthesized with `deepseek-v4-pro`. The generation run used DeepSeek's 75% OFF campaign pricing.

Reference values for people who want to reproduce or extend the dataset:

| Item | Value |
|---|---:|
| Model | `deepseek-v4-pro` |
| Token usage | Approximately 340M tokens |
| Actual cost | 115.66 USD |
| Pricing condition | DeepSeek 75% OFF campaign pricing |

This cost depends on the campaign price available at generation time, prompt structure, retry count, failure rate, and thinking/reasoning settings. Regenerating the same scale of data may cost a different amount under normal pricing or different generation settings.

### Caveats

- The dataset is synthetic and generated by DeepSeek-based LLM agents.
- Some records include internal/private fields such as `subtext`, `character_thought`, and `reasoning_content`.
- `public_transcript` excludes visible actions; use `public_timeline` if actions matter.
- If you need continuity for clothing, equipment, props, or positions, use `turns[].controller.content.turn_control.scene_state`, `turns[].controller.content.turn_control.state_memory`, and `persona_seed.scenario_constraints.continuity_notes`. `public_timeline` only contains utterances and visible actions.
- `source.text` may differ from a later edited `format.txt`; the JSONL stores the source text used for generation.
- Token usage fields are provider-specific and should be treated as metadata.
