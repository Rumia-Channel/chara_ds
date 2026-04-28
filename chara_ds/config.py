"""Constants and shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


PRO_MODEL = "deepseek-v4-pro"
FLASH_MODEL = "deepseek-v4-flash"
DEFAULT_MODEL = PRO_MODEL
DEEPSEEK_V4_MAX_OUTPUT_TOKENS = 384_000
# Beta endpoint required for `strict: true` tool schemas (and other beta features).
# The non-beta endpoint will reject `strict: true` with a 400.
DEFAULT_BASE_URL = "https://api.deepseek.com/beta"

DATASET_NAME = "persona_controlled_deepseek_triple_agent_ja"
SCHEMA_VERSION = "13.2"


@dataclass
class PersonaLine:
    line_number: int
    text: str
    sha256: str


@dataclass
class PromptBundle:
    persona_controller: str
    turn_controller: str
    actor: str
    actor_guard: str
