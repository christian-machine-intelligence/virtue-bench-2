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

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        thinking_mode: Optional[bool] = None,
    ):
        self._model = model
        self._thinking_mode = thinking_mode
        # Per-model call config lives on the runner. Reasoning models spend their
        # completion budget on a hidden pass before the answer, so the default is
        # generous; billing is per token used, not per cap.
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def model_id(self) -> str:
        return f"openai/{self._model}"

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Qwen thinking blocks are wrapped in <think>...</think>. Strip for
        # downstream answer extraction; keep the rest of the response intact.
        if "</think>" in text:
            return text.split("</think>", 1)[1].lstrip()
        return text

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,  # None -> use the runner's configured default
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
                mt = max_tokens if max_tokens is not None else self._max_tokens
                # Try max_completion_tokens first (GPT-5.x+), fall back to max_tokens
                if any(x in self._model for x in ["gpt-5", "o1", "o3", "o4"]):
                    kwargs["max_completion_tokens"] = mt
                else:
                    kwargs["max_tokens"] = mt

                # Thinking-mode toggle (for vLLM-hosted Qwen 3.5 etc.).
                # When thinking is on, bump max_tokens enough that the model has
                # room to finish its <think>...</think> block AND emit an A/B
                # answer. 2048 wasn't enough at 27B+ with psalm injection (~50%
                # of samples ran out before answering). 4096 is comfortable.
                if self._thinking_mode is not None:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": self._thinking_mode,
                        }
                    }
                    if self._thinking_mode is True:
                        if "max_tokens" in kwargs:
                            kwargs["max_tokens"] = max(kwargs["max_tokens"], 4096)
                        if "max_completion_tokens" in kwargs:
                            kwargs["max_completion_tokens"] = max(
                                kwargs["max_completion_tokens"], 4096
                            )

                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=timeout,
                )

                choice = response.choices[0]
                text = choice.message.content
                if text is None:
                    # A reasoning model can exhaust max_tokens on its hidden pass and
                    # return no answer; surface that distinctly so it's actionable.
                    if getattr(choice, "finish_reason", None) == "length":
                        last_error = "truncated_before_answer (increase max_tokens)"
                    else:
                        last_error = "null_content"
                    if attempt <= retries:
                        continue
                    break

                return {"response": self._strip_thinking(text.strip()), "infra_error": None}

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
