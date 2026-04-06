"""
OpenAI API / pi CLI runner backend.

Uses `pi -p` pipe mode for OpenAI and Gemini models, or direct API calls.
"""

from __future__ import annotations
import asyncio

from .base import ModelRunner


NEUTRAL_CWD = "/tmp"


class OpenAIAPIRunner(ModelRunner):
    """Runner that uses `pi -p` subprocess for OpenAI/Gemini evaluation."""

    def __init__(self, model: str = "gpt-4o"):
        self._model = model

    def model_id(self) -> str:
        return f"pi/{self._model}"

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        retries: int = 2,
        timeout: int = 120,
    ) -> dict:
        """Send a prompt to pi -p and return outcome."""
        last_error = "unknown"

        for attempt in range(1, retries + 2):
            proc = None
            try:
                cmd = [
                    "pi", "-p",
                    "--model", self._model,
                    "--system-prompt", system_prompt,
                    "--no-session-persistence",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=NEUTRAL_CWD,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode()),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                if proc is not None:
                    proc.kill()
                    try:
                        await proc.communicate()
                    except Exception:
                        pass
                last_error = "timeout"
                if attempt <= retries:
                    continue
                break
            except FileNotFoundError:
                last_error = "pi_not_found"
                break
            except Exception as exc:
                last_error = f"spawn_error:{exc.__class__.__name__}"
                if attempt <= retries:
                    continue
                break

            response = stdout.decode(errors="replace").strip()
            if proc.returncode != 0:
                last_error = "nonzero_exit"
                if attempt <= retries:
                    continue
                break

            if not response:
                last_error = "blank_response"
                if attempt <= retries:
                    continue
                break

            return {"response": response, "infra_error": None}

        return {"response": "", "infra_error": last_error}
