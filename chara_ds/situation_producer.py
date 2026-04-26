"""Background producer that grows a :class:`PersonaBuffer` with DeepSeek flash.

This is the in-process counterpart of the standalone ``gen_situations.py``
CLI: instead of writing N situations and exiting, it keeps appending new
ones to ``format.txt`` (and to a shared :class:`PersonaBuffer`) while
dialogue workers consume from the same buffer.
"""

from __future__ import annotations

import json
import random
import sys
import threading
from typing import List, Optional, Set

from .api_client import call_with_retries, make_client
from .config import PersonaLine
from .io_utils import read_text, sha256_text
from .persona_buffer import PersonaBuffer
from .situation_gen import (
    EMOTION_VOCAB,
    SITUATION_GEN_MODEL_DEFAULT,
    TONE_VOCAB,
    append_line,
    call_generator,
)


def _emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)


def start_background_producer(
    *,
    buffer: PersonaBuffer,
    out_path: str,
    prompt_file: str,
    seeds: List[str],
    target_count: Optional[int],
    stop_event: Optional[threading.Event] = None,
    batch_size: int = 8,
    max_iterations: int = 200,
    model: str = SITUATION_GEN_MODEL_DEFAULT,
    base_url: str,
    temperature: float = 1.1,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    retries: int = 4,
    retry_base_sleep: float = 2.0,
    seed: int = 20260426,
    errors_out: Optional[str] = None,
    existing_sample: int = 12,
) -> threading.Thread:
    """Spawn a daemon thread that grows ``buffer`` with new situations.

    Stop conditions (whichever fires first):
      * ``len(buffer) >= target_count`` (when ``target_count`` is not None);
      * ``stop_event.is_set()`` (caller signals "we're done, stop generating");
      * ``iteration >= max_iterations`` (safety cap).

    On exit always calls :meth:`PersonaBuffer.mark_finished` so any blocked
    workers fail fast instead of hanging.
    """
    system_prompt = read_text(prompt_file).strip()
    client = make_client(base_url)
    rng = random.Random(seed)
    errors_path = errors_out or out_path + ".gen_errors.jsonl"

    existing_hashes: Set[str] = {item.sha256 for item in buffer.snapshot()}

    def _next_line_number() -> int:
        snap = buffer.snapshot()
        return (snap[-1].line_number + 1) if snap else 1

    def _producer() -> None:
        try:
            _emit({
                "event": "situation_gen_background_start",
                "out": out_path,
                "model": model,
                "target": target_count,
                "initial": len(buffer),
                "batch_size": batch_size,
                "seeds_count": len(seeds),
            })

            iteration = 0
            while iteration < max_iterations:
                if stop_event is not None and stop_event.is_set():
                    break
                if target_count is not None and len(buffer) >= target_count:
                    break
                iteration += 1
                snap = buffer.snapshot()
                existing_texts = [item.text for item in snap]
                focus = rng.sample(EMOTION_VOCAB, k=min(4, len(EMOTION_VOCAB)))
                tone_focus = rng.sample(TONE_VOCAB, k=min(4, len(TONE_VOCAB)))
                existing_examples = (
                    rng.sample(
                        existing_texts,
                        k=min(existing_sample, len(existing_texts)),
                    )
                    if existing_texts
                    else []
                )

                try:
                    items = call_with_retries(
                        lambda: call_generator(
                            client,
                            model=model,
                            system_prompt=system_prompt,
                            seed_situations=seeds,
                            existing_examples=existing_examples,
                            batch_size=batch_size,
                            requested_emotion_focus=focus,
                            requested_tone_focus=tone_focus,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        ),
                        retries=retries,
                        errors_out=errors_path,
                        error_context={
                            "stage": "situation_gen_background",
                            "iteration": iteration,
                            "focus": focus,
                            "tone_focus": tone_focus,
                        },
                        retry_base_sleep=retry_base_sleep,
                    )
                except Exception as e:
                    _emit({
                        "event": "situation_gen_iteration_failed",
                        "iteration": iteration,
                        "error": str(e),
                    })
                    continue

                new_lines: List[PersonaLine] = []
                next_ln = _next_line_number()
                for it in items:
                    text = it["text"]
                    h = sha256_text(text)
                    if h in existing_hashes:
                        continue
                    existing_hashes.add(h)
                    append_line(out_path, text)
                    new_lines.append(
                        PersonaLine(
                            line_number=next_ln + len(new_lines),
                            text=text,
                            sha256=h,
                        )
                    )
                    if (
                        target_count is not None
                        and len(buffer) + len(new_lines) >= target_count
                    ):
                        break

                if new_lines:
                    buffer.extend(new_lines)

                _emit({
                    "event": "situation_gen_iteration_done",
                    "iteration": iteration,
                    "added": len(new_lines),
                    "total": len(buffer),
                    "target": target_count,
                    "focus": focus,
                    "tone_focus": tone_focus,
                })
        finally:
            buffer.mark_finished()
            _emit({
                "event": "situation_gen_background_done",
                "total": len(buffer),
                "target": target_count,
            })

    thread = threading.Thread(
        target=_producer, name="situation-producer", daemon=True
    )
    thread.start()
    return thread
