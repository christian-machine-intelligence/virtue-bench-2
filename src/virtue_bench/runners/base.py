"""
Unified ModelRunner protocol for VirtueBench V2.

All runner backends implement this protocol, replacing V1's three
separate runner files (experiment.py, run_cli.py, run_pi.py).
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class ModelRunner(ABC):
    """Abstract base for model evaluation backends."""

    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> dict:
        """Send a prompt to the model and return the response.

        Returns:
            dict with keys:
                response: str — model output text
                infra_error: str | None — error description if call failed
        """
        ...

    @abstractmethod
    def model_id(self) -> str:
        """Return a display-friendly model identifier."""
        ...
