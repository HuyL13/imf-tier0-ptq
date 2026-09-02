from pathlib import Path

import pytest
from pydantic import ValidationError

from imf_tier0.config import ExperimentConfig


CONFIG = Path(__file__).parents[1] / "configs" / "llama31_8b.yaml"


def test_reference_config_locks_scientific_protocol() -> None:
    config = ExperimentConfig.model_validate_yaml(CONFIG)

    assert config.model.model_id == "meta-llama/Llama-3.1-8B"
    assert config.data.num_fingerprints == 10
    assert config.data.num_normal == 50
    assert [(q.backend, q.bits, q.group_size) for q in config.quantization] == [
        ("rtn", 3, 128),
        ("rtn", 4, 128),
        ("awq", 3, 128),
        ("awq", 4, 128),
        ("gptq", 3, 128),
    ]


def test_config_rejects_unapproved_quantization_matrix() -> None:
    raw = ExperimentConfig.model_validate_yaml(CONFIG).model_dump(mode="json")
    raw["quantization"].append(
        {"backend": "gptq", "bits": 4, "group_size": 128}
    )

    with pytest.raises(ValidationError, match="exact Tier-0 PTQ matrix"):
        ExperimentConfig.model_validate(raw)


def test_config_requires_provenance_for_unreported_values() -> None:
    raw = ExperimentConfig.model_validate_yaml(CONFIG).model_dump(mode="json")
    raw["training"]["learning_rate"]["provenance"] = "paper"

    with pytest.raises(ValidationError, match="not_reported_by_paper"):
        ExperimentConfig.model_validate(raw)

