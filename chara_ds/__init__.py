"""chara_ds: DeepSeek triple-agent dialogue dataset generator."""

from .config import (
    DATASET_NAME,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SCHEMA_VERSION,
    PersonaLine,
    PromptBundle,
)

__all__ = [
    "DATASET_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "SCHEMA_VERSION",
    "PersonaLine",
    "PromptBundle",
]
