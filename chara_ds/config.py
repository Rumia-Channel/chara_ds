"""Constants and shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"

DATASET_NAME = "persona_controlled_deepseek_triple_agent_ja"
SCHEMA_VERSION = "11.0"


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
