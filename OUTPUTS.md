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
| `schema_version` | string | スキーマバージョン。現在値: `13.0`。 |
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
| `usage` | object | ペルソナ、ターンコントローラー、アクターのトークン使用量。 |
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
| `scenario_constraints` | 場所、許可される話題や行動、避ける話題、文体メモ、終了条件。 |

各キャラクターには通常、以下のフィールドが含まれます。

| フィールド | 説明 |
|---|---|
| `role` | 場面内での役割。 |
| `age_band` | `teen`、`20s`、`30s` などの年齢帯。 |
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

各要素はおおむね以下の形です。

```json
{
  "turn": 1,
  "controller": {
    "content": {
      "turn_control": {
        "next_speaker": "A",
        "scene_state": "...",
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
| `target_turns` | ランダムに選ばれた目標ターン数。 |
| `actual_turns` | 実際に生成された公開ターン数。 |
| `min_turns` / `max_turns` | 生成時に指定されたターン数範囲。 |
| `seed` | 生成シード。 |
| `variation` | 元お題行に対するバリエーション番号。 |
| `controller_temperature` / `controller_top_p` | thinking 無効時のターンコントローラーのサンプリング設定。 |
| `max_tokens_policy` | ペルソナ、コントローラー、アクター呼び出しの最大トークン設定。 |

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
| `schema_version` | string | Schema version. Current value: `13.0`. |
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
| `usage` | object | Token usage for persona controller, turn controller, and actors. |
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
| `scenario_constraints` | Setting, allowed topics/actions, avoid topics, style notes, and ending condition. |

Each character usually contains:

| Field | Description |
|---|---|
| `role` | Character role in the scene. |
| `age_band` | Age band such as `teen`, `20s`, `30s`, etc. |
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

Each element has approximately this shape:

```json
{
  "turn": 1,
  "controller": {
    "content": {
      "turn_control": {
        "next_speaker": "A",
        "scene_state": "...",
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
| `target_turns` | Randomly selected target turn count. |
| `actual_turns` | Actual number of generated public turns. |
| `min_turns` / `max_turns` | Turn range requested at generation time. |
| `seed` | Generation seed. |
| `variation` | Variation index for the source line. |
| `controller_temperature` / `controller_top_p` | Turn controller sampling settings when thinking is disabled. |
| `max_tokens_policy` | Max-token settings for persona, controller, and actor calls. |

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
- `source.text` may differ from a later edited `format.txt`; the JSONL stores the source text used for generation.
- Token usage fields are provider-specific and should be treated as metadata.
