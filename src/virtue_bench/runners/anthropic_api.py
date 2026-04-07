"""
Anthropic API runner backend.

Uses the Anthropic Python SDK directly for evaluation.
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional

import anthropic

from .base import ModelRunner


class AnthropicAPIRunner(ModelRunner):
    """Runner that uses the Anthropic API directly."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self._model = model
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

        for attempt in range(1, retries + 2):
            try:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=self._model,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )

                if not response.content:
                    last_error = "empty_content"
                    if attempt <= retries:
                        continue
                    break

                text = response.content[0].text.strip()
                return {"response": text, "infra_error": None}

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
