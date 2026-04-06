"""
Inspect AI runner backend.

Wraps the UK AI Safety Institute's inspect-ai framework for compatibility
with VirtueBench V2's unified runner protocol. Used for API-key-based
evaluation via inspect_eval.
"""

from __future__ import annotations

from typing import List, Optional

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample as InspectSample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, scorer, accuracy, CORRECT, INCORRECT
from inspect_ai.solver import generate, system_message

from ..core.loader import parse_answer
from ..core.schema import PreparedSample
from .base import ModelRunner


@scorer(metrics=[accuracy()])
def leading_letter_scorer():
    """Score by extracting the first 'A' or 'B' character from the model output."""
    async def score(state, target):
        text = state.output.completion.strip()
        answer = parse_answer(text)
        if answer:
            return Score(
                value=CORRECT if answer == target.text else INCORRECT,
                answer=answer,
                explanation=text,
            )
        return Score(
            value=INCORRECT,
            answer=text[:50],
            explanation=f"Could not extract A or B from: {text[:100]}",
        )
    return score


class InspectAIRunner(ModelRunner):
    """Runner that delegates to Inspect AI's eval framework."""

    def __init__(self, model: str):
        self._model = model

    def model_id(self) -> str:
        return self._model

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> dict:
        # InspectAIRunner uses batch mode via run_batch, not individual queries
        raise NotImplementedError(
            "InspectAIRunner uses run_batch() for evaluation, not individual query()."
        )

    def run_batch(
        self,
        samples: List[PreparedSample],
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        log_dir: Optional[str] = None,
    ) -> dict:
        """Run a batch of samples through Inspect AI and return aggregate results."""
        inspect_samples = [
            InspectSample(
                input=s.prompt,
                target=s.target,
                metadata={
                    "virtue": s.scenario.virtue,
                    "source": s.scenario.source,
                    "base_id": s.scenario.base_id,
                    "variant": s.scenario.variant,
                },
            )
            for s in samples
        ]

        config = GenerateConfig(temperature=temperature, max_tokens=max_tokens)
        solver_pipeline = [system_message(system_prompt), generate()]

        task = Task(
            dataset=MemoryDataset(inspect_samples),
            solver=solver_pipeline,
            scorer=leading_letter_scorer(),
            config=config,
        )

        kwargs = {"model": self._model}
        if log_dir:
            kwargs["log_dir"] = log_dir

        logs = inspect_eval(task, **kwargs)
        log = logs[0]

        results = log.results
        metrics = {}
        if results and results.scores:
            for score_obj in results.scores:
                metrics.update({k: v.value for k, v in score_obj.metrics.items()})

        sample_details = []
        if log.samples:
            for sample in log.samples:
                prompt_text = ""
                if sample.input:
                    if isinstance(sample.input, str):
                        prompt_text = sample.input
                    elif isinstance(sample.input, list):
                        prompt_text = " ".join(
                            m.text for m in sample.input if hasattr(m, "text") and m.text
                        )
                model_response = ""
                if sample.output and sample.output.completion:
                    model_response = sample.output.completion.strip()

                answer = None
                correct = None
                if sample.scores:
                    for _, score_val in sample.scores.items():
                        answer = score_val.answer
                        correct = score_val.value == "C"
                        break

                sample_details.append({
                    "sample_id": sample.metadata.get("base_id", sample.id) if sample.metadata else sample.id,
                    "prompt": prompt_text,
                    "target": sample.target,
                    "model_response": model_response,
                    "model_answer": answer,
                    "correct": correct,
                    "metadata": sample.metadata or {},
                })

        return {
            "accuracy": metrics.get("accuracy"),
            "stderr": metrics.get("stderr"),
            "samples": log.eval.dataset.samples if log.eval.dataset else len(samples),
            "status": str(log.status),
            "sample_details": sample_details,
        }
