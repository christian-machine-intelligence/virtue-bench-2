"""
Pydantic models for VirtueBench V2 scenarios, results, and config.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


Virtue = Literal["prudence", "justice", "courage", "temperance"]
Variant = Literal["ratio", "caro", "mundus", "diabolus", "ignatian"]


class Scenario(BaseModel):
    """A single evaluation scenario with its temptation variant."""
    base_id: str
    variant: Variant
    scenario_a: str  # virtuous choice (always ground truth)
    scenario_b: str  # temptation
    virtue: Virtue
    source: str  # patristic citation
    deviation_point: Optional[str] = None  # Ignatian only


class PreparedSample(BaseModel):
    """A scenario prepared for evaluation with A/B randomization applied."""
    scenario: Scenario
    prompt: str  # formatted Option A / Option B text
    target: Literal["A", "B"]  # correct answer after randomization


class SampleResult(BaseModel):
    """Result for a single evaluated sample."""
    sample_id: str  # base_id
    variant: Variant
    target: str
    model_response: str
    model_answer: Optional[str] = None  # parsed A or B
    correct: Optional[bool] = None
    prompt: str
    metadata: dict = Field(default_factory=dict)
    infra_error: Optional[str] = None


class RunResult(BaseModel):
    """Aggregated result for one run of a virtue x variant condition."""
    model: str
    virtue: Virtue
    variant: Variant
    condition: str  # e.g. "actual", "actual+injected"
    frame: str
    run_index: int
    seed: int
    temperature: float
    accuracy: Optional[float] = None
    stderr: Optional[float] = None
    samples: int
    status: str
    sample_details: List[SampleResult] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """YAML-serializable experiment configuration."""
    name: str
    model: str
    virtues: List[Virtue] = Field(default_factory=lambda: ["prudence", "justice", "courage", "temperance"])
    variants: List[Variant] = Field(default_factory=lambda: ["ratio", "caro", "mundus", "diabolus", "ignatian"])
    runs: int = 5
    temperature: float = 0.7
    frame: str = "default"
    seed: int = 42
    limit: Optional[int] = None
    injection_file: Optional[str] = None
    concurrency: int = 5
    retries: int = 2
    timeout: int = 120
    detailed: bool = False
