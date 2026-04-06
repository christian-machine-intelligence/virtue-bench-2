"""Model runner backends."""
from __future__ import annotations

from typing import Dict, Type

from .base import ModelRunner
from .inspect_ai import InspectAIRunner
from .claude_cli import ClaudeCLIRunner
from .openai_api import OpenAIAPIRunner

RUNNERS: Dict[str, Type[ModelRunner]] = {
    "inspect": InspectAIRunner,
    "claude-cli": ClaudeCLIRunner,
    "openai-api": OpenAIAPIRunner,
}

__all__ = ["ModelRunner", "RUNNERS", "InspectAIRunner", "ClaudeCLIRunner", "OpenAIAPIRunner"]
