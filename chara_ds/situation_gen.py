"""Generate base situations (one-line dialogue seeds) with DeepSeek flash.

Reads optional seed situations + the existing format.txt, then asks the model
to emit N more diverse situations per call. Writes them back to format.txt
(append, dedup by sha256). Uses non-thinking mode for speed/cost and
``strict: true`` tool schema to guarantee shape.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .api_client import call_deepseek_tool, call_with_retries, make_client
from .config import DEFAULT_BASE_URL
from .io_utils import (
    append_jsonl,
    now_iso,
    read_text,
    safe_mkdir_for_file,
    sha256_text,
)


SITUATION_GEN_MODEL_DEFAULT = "deepseek-v4-flash"

EMOTION_VOCAB = [
    "怒り", "悲しみ", "喜び", "虚無感", "恐れ", "嫌悪", "驚き",
    "羞恥", "愛情", "嫉妬", "罪悪感", "誇り", "軽蔑", "切なさ", "興奮",
    "不安", "苛立ち", "退屈", "面倒くささ", "優越感", "無関心",
    "性的興奮", "ばかばかしさ", "好奇心", "憎悪", "しらけ", "呆れ",
    "自己嫌悪", "依存", "支配欲",
]

TONE_VOCAB = [
    "stupid_banter",          # しょうもない雑談・悪ふざけ・下ネタ
    "trivial_chitchat",       # 退屈な日常会話・雑談
    "everyday_complaint",     # しょうもない愚痴
    "anxious_inner_monologue", # 不安・焦り・自己嫌悪
    "dark_resentment",        # どす黒い恨み・嫉妬・憎悪
    "passive_aggressive",     # 嫌味・遠回しな攻撃・陰口
    "power_play",             # マウント・支配・見下し
    "sexual_tension",         # 性的緊張・欲望・嫉妬
    "violent_clash",          # 怒鳴り合い・取っ組み合い・修羅場
    "embarrassing_secret",    # 暴露・告白・恥ずかしい本音
    "boring_smalltalk",       # 低カロリーな当たり障りない会話
    "absurd_misunderstanding", # 滑稽な勘違い・ばかばかしい衝突
    "cynical_humor",          # 皮肉・冷笑・しらけ
    "dependent_clinging",     # 依存・束縛・つきまとい
    "heartfelt_reconciliation", # 真摯な和解・許し
    "quiet_melancholy",       # 静かな寂しさ・諦め
]

# 世界観 / 舞台設定。modern_realism に偏らないようローテートする。
SETTING_VOCAB = [
    "modern_realism",          # 現代日本のリアル (家・学校・職場・街)
    "school_realism",          # 中高大学などの校内・部活・教室
    "workplace_realism",       # オフィス・接客業・夜の店・現場仕事
    "family_domestic",         # 家庭内・親戚・実家・帰省
    "online_community",        # SNS・ゲーム・配信・掲示板の人間関係
    "fantasy_high",            # 王道剣と魔法のハイファンタジー
    "fantasy_dark",            # ダークファンタジー・血なまぐさい戦場や呪い
    "fantasy_cozy_villain",    # 魔王軍幹部のほんわかしてるけど物騒な日常
    "monster_pov",             # 魔物・モンスター同士の会話 (人間視点ではない)
    "hero_party_strife",       # 勇者パーティー内のぎすぎす・不和・裏切り疑惑
    "magic_academy",           # 魔法学校・寮・実技授業・落第寸前の学生
    "magic_research_lab",      # 女博士と助手の魔法談義、禁書研究、倫理ぎりぎりの実験
    "isekai_transferred",      # 異世界転移・転生もの (現代知識と異世界の摩擦)
    "sci_fi_space",            # 宇宙船・コロニー・宇宙海賊・AI と人間
    "cyberpunk",               # 退廃的近未来都市・電脳・企業の暗部
    "post_apocalypse",         # 文明崩壊後・サバイバル・配給・派閥抗争
    "historical_japan",        # 江戸・戦国・幕末などの日本史的舞台
    "wuxia_xianxia",           # 中華武侠・仙術・宗門の派閥・修行
    "horror_occult",           # 怪談・心霊・カルト・呪い・閉鎖空間
    "mythological",            # 神話・神々・精霊・人外と人間の交わり
    "noir_crime",              # 探偵・裏社会・刑事・取り調べ
    "military_war",            # 軍隊・前線・後方支援・戦友間の軋轢
    "idol_entertainment",      # アイドル・声優・劇団・芸能裏方の修羅
    "sports_competitive",      # 競技スポーツ・レギュラー争い・引退試合
]

SITUATION_TOOL_NAME = "submit_situations"
SITUATION_TOOL_DESCRIPTION = (
    "新規ベースシチュエーションを N 件まとめて提出する。"
    "各要素は text (日本語1行) と dominant_emotions (主要感情 2〜4個) を持つ。"
)
SITUATION_TOOL_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "situations": {
            "type": "array",
            "description": "新規シチュエーションの配列。",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "ベースシチュエーション本文。日本語1行・改行禁止・"
                            "80〜180文字程度。冒頭は「Aは...、Bは...」で始める。"
                        ),
                    },
                    "dominant_emotions": {
                        "type": "array",
                        "description": "このシチュエーションで主に生じる感情を 2〜4 個。",
                        "items": {
                            "type": "string",
                            "enum": EMOTION_VOCAB,
                        },
                    },
                    "tone": {
                        "type": "string",
                        "description": (
                            "このシチュエーションのトーン/ジャンルラベル。"
                            "requested_tone_focus と整合させ、batch 内で多様化する。"
                        ),
                        "enum": TONE_VOCAB,
                    },
                    "setting": {
                        "type": "string",
                        "description": (
                            "舞台設定 (世界観)。requested_setting_focus と整合させ、"
                            "batch 内で必ず複数の setting を混ぜる。modern_realism に偏らせない。"
                        ),
                        "enum": SETTING_VOCAB,
                    },
                },
                "required": ["text", "dominant_emotions", "tone", "setting"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["situations"],
    "additionalProperties": False,
}


def load_existing_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def append_line(path: str, text: str) -> None:
    safe_mkdir_for_file(path)
    p = Path(path)
    needs_newline = False
    if p.exists() and p.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        needs_newline = last != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(text + "\n")


def normalize_text(text: str) -> str:
    # Single-line, trimmed, no surrounding quotes, no inner newlines.
    t = text.strip().replace("\r", " ").replace("\n", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def call_generator(
    client,
    *,
    model: str,
    system_prompt: str,
    seed_situations: List[str],
    existing_examples: List[str],
    batch_size: int,
    requested_emotion_focus: List[str],
    requested_tone_focus: List[str],
    requested_setting_focus: List[str],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    static_context = {
        "task": "generate_base_situations",
        "instruction": (
            f"submit_situations を呼び出し、situations 配列に {batch_size} 件、"
            "互いに重複しない多様なシチュエーションを入れる。"
            "requested_tone_focus に挙がったトーンと requested_setting_focus に挙がった世界観は "
            "batch 内で必ずカバーし、現代日本の現実 (modern_realism) ばかりに偏らせない。"
            "感動寄り・教訓寄りに偏らないこと。"
            "本文には何も書かない。"
        ),
        "seed_situations": seed_situations,
        "emotion_vocabulary": EMOTION_VOCAB,
        "tone_vocabulary": TONE_VOCAB,
        "setting_vocabulary": SETTING_VOCAB,
    }

    payload = {
        "batch_size": batch_size,
        "requested_emotion_focus": requested_emotion_focus,
        "requested_tone_focus": requested_tone_focus,
        "requested_setting_focus": requested_setting_focus,
        "existing_examples": existing_examples,
    }

    args, _reasoning, _usage, _raw = call_deepseek_tool(
        client,
        model=model,
        system_prompt=system_prompt,
        user_payload=payload,
        static_context=static_context,
        tool_name=SITUATION_TOOL_NAME,
        tool_description=SITUATION_TOOL_DESCRIPTION,
        tool_parameters=SITUATION_TOOL_PARAMETERS,
        tool_strict=True,
        max_tokens=max_tokens,
        reasoning_effort="high",
        thinking_enabled=None,
        temperature=temperature,
        top_p=top_p,
    )

    raw_items = args.get("situations") or []
    out: List[Dict[str, Any]] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        text = it.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        out.append(
            {
                "text": normalize_text(text),
                "dominant_emotions": it.get("dominant_emotions") or [],
                "tone": it.get("tone") or "",
                "setting": it.get("setting") or "",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate base dialogue situations with DeepSeek flash.",
    )
    parser.add_argument("--out", default="./format.txt",
                        help="Append generated situations here (one per line).")
    parser.add_argument("--prompt-dir", default="./prompts",
                        help="Prompt directory. Used for situation_gen.txt when --prompt-file is omitted.")
    parser.add_argument("--prompt-file", default=None,
                        help="Prompt file. Defaults to <prompt-dir>/situation_gen.txt.")
    parser.add_argument("--seed-situation", action="append", default=[],
                        help="User-provided seed situation. Repeatable.")
    parser.add_argument("--seed-file", default=None,
                        help="Optional file: each non-empty line = one seed.")
    parser.add_argument("--use-existing-as-seed", action="store_true",
                        help="Also feed already-existing format.txt lines as seeds (sample).")
    parser.add_argument("--target", type=int, default=50,
                        help="Total situation count to reach in --out (after dedup).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Situations per API call.")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="Hard cap on generator calls.")
    parser.add_argument("--existing-sample", type=int, default=12,
                        help="How many existing lines to show the model as anti-duplication context.")

    parser.add_argument("--model", default=SITUATION_GEN_MODEL_DEFAULT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--errors-out", default=None,
                        help="Where to write per-call errors. Defaults to <out>.gen_errors.jsonl")
    return parser.parse_args()


def gather_seeds(args: argparse.Namespace, existing: List[str]) -> List[str]:
    seeds: List[str] = list(args.seed_situation or [])
    if args.seed_file:
        for line in read_text(args.seed_file).splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                seeds.append(s)
    if args.use_existing_as_seed and existing:
        seeds.extend(existing[:8])
    if not seeds and existing:
        # Fall back to existing as inspiration so the model has something concrete.
        seeds = existing[:4]
    return seeds


def main() -> None:
    args = parse_args()
    errors_out = args.errors_out or args.out + ".gen_errors.jsonl"
    prompt_file = args.prompt_file or str(Path(args.prompt_dir) / "situation_gen.txt")

    system_prompt = read_text(prompt_file).strip()
    client = make_client(args.base_url)

    existing = load_existing_lines(args.out)
    existing_hashes: Set[str] = {sha256_text(line) for line in existing}

    seeds = gather_seeds(args, existing)
    rng = random.Random(args.seed)

    print(json.dumps(
        {
            "event": "situation_gen_start",
            "out": args.out,
            "model": args.model,
            "base_url": args.base_url,
            "prompt_file": prompt_file,
            "target": args.target,
            "already": len(existing),
            "batch_size": args.batch_size,
            "seeds_count": len(seeds),
        },
        ensure_ascii=False,
    ), file=sys.stderr)

    iteration = 0
    while len(existing_hashes) < args.target and iteration < args.max_iterations:
        iteration += 1
        # Rotate emotion focus so subsequent batches diversify.
        focus = rng.sample(EMOTION_VOCAB, k=min(4, len(EMOTION_VOCAB)))
        tone_focus = rng.sample(TONE_VOCAB, k=min(4, len(TONE_VOCAB)))
        # Always force a non-realism setting into every batch so fantasy /
        # sci-fi / horror / etc. show up regularly even for short runs.
        non_realism = [s for s in SETTING_VOCAB if s != "modern_realism"]
        setting_focus = rng.sample(non_realism, k=min(3, len(non_realism)))
        setting_focus.append(rng.choice(SETTING_VOCAB))
        existing_sample = (
            rng.sample(existing, k=min(args.existing_sample, len(existing)))
            if existing else []
        )

        try:
            items = call_with_retries(
                lambda: call_generator(
                    client,
                    model=args.model,
                    system_prompt=system_prompt,
                    seed_situations=seeds,
                    existing_examples=existing_sample,
                    batch_size=args.batch_size,
                    requested_emotion_focus=focus,
                    requested_tone_focus=tone_focus,
                    requested_setting_focus=setting_focus,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens if args.max_tokens > 0 else None,
                ),
                retries=args.retries,
                errors_out=errors_out,
                error_context={
                    "stage": "situation_gen",
                    "iteration": iteration,
                    "focus": focus,
                    "tone_focus": tone_focus,
                    "setting_focus": setting_focus,
                },
                retry_base_sleep=args.retry_base_sleep,
            )
        except Exception as e:
            print(json.dumps(
                {
                    "event": "situation_gen_iteration_failed",
                    "iteration": iteration,
                    "error": str(e),
                },
                ensure_ascii=False,
            ), file=sys.stderr)
            continue

        added = 0
        for it in items:
            text = it["text"]
            h = sha256_text(text)
            if h in existing_hashes:
                continue
            existing_hashes.add(h)
            existing.append(text)
            append_line(args.out, text)
            added += 1
            if len(existing_hashes) >= args.target:
                break

        print(json.dumps(
            {
                "event": "situation_gen_iteration_done",
                "iteration": iteration,
                "added": added,
                "total": len(existing_hashes),
                "target": args.target,
                "focus": focus,
                "tone_focus": tone_focus,
                "setting_focus": setting_focus,
            },
            ensure_ascii=False,
        ), file=sys.stderr)

    print(json.dumps(
        {
            "event": "situation_gen_finished",
            "total": len(existing_hashes),
            "target": args.target,
            "iterations": iteration,
        },
        ensure_ascii=False,
    ), file=sys.stderr)


if __name__ == "__main__":
    main()
