"""
OpenAI API runner backend.

Uses the OpenAI Python SDK directly for evaluation.
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional

from openai import AsyncOpenAI

from .base import ModelRunner


class OpenAIAPIRunner(ModelRunner):
    """Runner that uses the OpenAI API directly."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def model_id(self) -> str:
        return f"openai/{self._model}"

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        retries: int = 2,
        timeout: int = 120,
    ) -> dict:
        """Send a prompt to the OpenAI API and return outcome."""
        last_error = "unknown"

        for attempt in range(1, retries + 2):
            try:
                # Newer OpenAI models use max_completion_tokens instead of max_tokens
                kwargs = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }
                # Try max_completion_tokens first (GPT-5.x+), fall back to max_tokens
                if any(x in self._model for x in ["gpt-5", "o1", "o3", "o4"]):
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=timeout,
                )

                text = response.choices[0].message.content
                if text is None:
                    last_error = "null_content"
                    if attempt <= retries:
                        continue
                    break

                return {"response": text.strip(), "infra_error": None}

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
