"""Model runner backends.

Five runner options:

  openai-api     Direct OpenAI Python SDK (default for OpenAI/GPT models)
  anthropic-api  Direct Anthropic Python SDK (default for Claude models)
  claude-cli     claude -p pipe mode (Claude Max subscription)
  pi-cli         pi -p pipe mode (ChatGPT Pro subscription)
  inspect        Inspect AI framework (UK AISI, requires inspect-ai package)
"""
from __future__ import annotations

from typing import Dict, Type

from .base import ModelRunner
from .openai_api import OpenAIAPIRunner
from .anthropic_api import AnthropicAPIRunner
from .claude_cli import ClaudeCLIRunner
from .pi_cli import PiCLIRunner

# Lazy import for InspectAIRunner — only needed if inspect-ai is installed
try:
    from .inspect_ai import InspectAIRunner
    _HAS_INSPECT = True
except ImportError:
    _HAS_INSPECT = False
    InspectAIRunner = None  # type: ignore

RUNNERS: Dict[str, Type[ModelRunner]] = {
    "openai-api": OpenAIAPIRunner,
    "anthropic-api": AnthropicAPIRunner,
    "claude-cli": ClaudeCLIRunner,
    "pi-cli": PiCLIRunner,
}
if _HAS_INSPECT:
    RUNNERS["inspect"] = InspectAIRunner

__all__ = [
    "ModelRunner", "RUNNERS",
    "OpenAIAPIRunner", "AnthropicAPIRunner",
    "ClaudeCLIRunner", "PiCLIRunner",
]
if _HAS_INSPECT:
    __all__.append("InspectAIRunner")
