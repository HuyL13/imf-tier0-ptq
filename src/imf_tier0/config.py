from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProvenancedFloat(StrictModel):
    value: float
    provenance: Literal["paper", "tier0_plan", "not_reported_by_paper"]


class ModelSpec(StrictModel):
    model_id: Literal["meta-llama/Llama-3.1-8B"]
    revision: str = Field(min_length=1)


class DataSpec(StrictModel):
    num_fingerprints: Literal[10]
    num_normal: Literal[50]
    seed: int


class TrainingSpec(StrictModel):
    learning_rate: ProvenancedFloat


class PTQSpec(StrictModel):
    backend: Literal["rtn", "awq", "gptq"]
    bits: Literal[3, 4]
    group_size: Literal[128]


EXPECTED_PTQ_MATRIX = [
    ("rtn", 3, 128),
    ("rtn", 4, 128),
    ("awq", 3, 128),
    ("awq", 4, 128),
    ("gptq", 3, 128),
]


class ExperimentConfig(StrictModel):
    model: ModelSpec
    data: DataSpec
    training: TrainingSpec
    quantization: list[PTQSpec]

    @model_validator(mode="after")
    def enforce_protocol(self) -> "ExperimentConfig":
        matrix = [(q.backend, q.bits, q.group_size) for q in self.quantization]
        if matrix != EXPECTED_PTQ_MATRIX:
            raise ValueError("quantization must match the exact Tier-0 PTQ matrix")
        if self.training.learning_rate.provenance != "not_reported_by_paper":
            raise ValueError(
                "learning_rate must be labeled not_reported_by_paper"
            )
        return self

    @classmethod
    def model_validate_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw: Any = yaml.safe_load(stream)
        return cls.model_validate(raw)

