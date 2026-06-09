"""
Anthropic API runner backend.

Uses the Anthropic Python SDK directly for evaluation.

Model-family handling
----------------------
Models in the post-Opus-4.7 family removed the sampling parameters
(`temperature`/`top_p`/`top_k`) and reject them with a 400. For those models
we omit `temperature` from the request entirely. Omitting it is safe for every
model — if the model accepts `temperature`, leaving it off just uses the
default; if it rejects it, omitting avoids the 400. These models govern
stochasticity through adaptive thinking rather than a temperature knob, so
multi-run variance still works.

Some models (Fable 5 / Mythos 5) have adaptive thinking *always on*. Thinking
tokens count against `max_tokens`, so the benchmark's default 128-token cap can
be consumed by thinking before the model ever emits the A/B letter, truncating
the answer. For those models we raise a `max_tokens` floor and read the first
*text* block (skipping thinking blocks), rather than blindly taking
`content[0]`.
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional

import anthropic

from .base import ModelRunner

# Bare model id (no "anthropic/" prefix) substrings whose families dropped the
# sampling parameters and 400 on temperature/top_p/top_k.
_NO_SAMPLING_MARKERS = (
    "fable-5",
    "mythos-5",
    "mythos-preview",
    "opus-4-7",
    "opus-4-8",
)

# Models with adaptive thinking always on. Thinking tokens are billed as output
# and consume the max_tokens budget, so the answer needs headroom to survive.
_ALWAYS_THINKING_MARKERS = (
    "fable-5",
    "mythos-5",
    "mythos-preview",
)

# Floor for max_tokens on always-thinking models: enough room for adaptive
# thinking plus the short "<letter> — <one sentence>" answer.
_THINKING_MAX_TOKENS_FLOOR = 4096


def _matches(model: str, markers: tuple[str, ...]) -> bool:
    m = model.lower()
    return any(marker in m for marker in markers)


def _usage_dict(response) -> dict:
    """Pull token usage off a response for cost calibration (best-effort)."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    return {
        "input_tokens": getattr(u, "input_tokens", None),
        "output_tokens": getattr(u, "output_tokens", None),
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
    }


def _extract_text(response) -> str:
    """Return the first non-empty text block, skipping thinking/other blocks.

    The legacy code used `response.content[0].text`, which breaks when the
    first block is a thinking block (no `.text`, or empty text under the default
    `display: "omitted"`).
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text = (getattr(block, "text", "") or "").strip()
            if text:
                return text
    return ""


class AnthropicAPIRunner(ModelRunner):
    """Runner that uses the Anthropic API directly."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self._model = model
        self._no_sampling = _matches(model, _NO_SAMPLING_MARKERS)
        self._always_thinking = _matches(model, _ALWAYS_THINKING_MARKERS)
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    def model_id(self) -> str:
        return f"anthropic/{self._model}"

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        retries: int = 2,
        timeout: int = 120,
    ) -> dict:
        """Send a prompt to the Anthropic API and return outcome."""
        last_error = "unknown"

        # Adapt request to the model family.
        if self._always_thinking:
            # Thinking tokens consume the budget; give the answer room to survive.
            effective_max_tokens = max(max_tokens, _THINKING_MAX_TOKENS_FLOOR)
        else:
            effective_max_tokens = max_tokens

        base_kwargs = dict(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=effective_max_tokens,
        )
        if self._always_thinking:
            base_kwargs["thinking"] = {"type": "adaptive"}
        # Only send temperature to models that still accept it (others 400).
        if not self._no_sampling:
            base_kwargs["temperature"] = temperature

        for attempt in range(1, retries + 2):
            try:
                response = await asyncio.wait_for(
                    self._client.messages.create(**base_kwargs),
                    timeout=timeout,
                )

                text = _extract_text(response)
                if not text:
                    # Distinguish a budget-truncated thinking run from a genuinely
                    # empty response so calibration surfaces it instead of scoring
                    # it as a blank (wrong) answer.
                    if getattr(response, "stop_reason", None) == "max_tokens":
                        last_error = "max_tokens_truncation"
                    else:
                        last_error = "empty_content"
                    if attempt <= retries:
                        continue
                    return {"response": "", "infra_error": last_error,
                            "usage": _usage_dict(response)}

                return {"response": text, "infra_error": None,
                        "usage": _usage_dict(response)}

            except asyncio.TimeoutError:
                last_error = "timeout"
                if attempt <= retries:
                    continue
                break
            except Exception as exc:
                last_error = f"api_error:{exc.__class__.__name__}:{str(exc)[:100]}"
                if attempt <= retries:
                    await asyncio.sleep(1)
                    continue
                break

        return {"response": "", "infra_error": last_error}
