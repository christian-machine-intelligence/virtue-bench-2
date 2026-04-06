"""
Claude CLI runner backend.

Uses `claude -p` pipe mode for evaluation with a Claude Max subscription.
"""

from __future__ import annotations
import asyncio
from typing import Optional

from .base import ModelRunner


NEUTRAL_CWD = "/tmp"


class ClaudeCLIRunner(ModelRunner):
    """Runner that uses `claude -p` subprocess for evaluation."""

    def __init__(self, model: str = "sonnet", effort: Optional[str] = "low"):
        self._model = model
        self._effort = effort

    def model_id(self) -> str:
        suffix = f"@{self._effort}" if self._effort else ""
        return f"claude-p/{self._model}{suffix}"

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        retries: int = 2,
        timeout: int = 120,
    ) -> dict:
        """Send a prompt to Claude via `claude -p` and return outcome."""
        last_error = "unknown"

        for attempt in range(1, retries + 2):
            proc = None
            try:
                cmd = [
                    "claude", "-p",
                    "--model", self._model,
                    "--system-prompt", system_prompt,
                    "--no-session-persistence",
                    "--setting-sources", "",
                    "--tools", "",
                    "--disable-slash-commands",
                    "--strict-mcp-config",
                ]
                if self._effort:
                    cmd.extend(["--effort", self._effort])
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
                last_error = "claude_not_found"
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
