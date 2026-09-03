import json
from pathlib import Path

from imf_tier0.cli import main


ROOT = Path(__file__).parents[1]


def test_cpu_preflight_validates_pinned_artifacts_without_torch(capsys) -> None:
    exit_code = main([
        "preflight", "--config", str(ROOT / "configs" / "llama31_8b.yaml"),
        "--cpu-only",
    ])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["model_id"] == "meta-llama/Llama-3.1-8B"
    assert output["ppl_evaluator_sha256"] == "309d4b01d5686143fbdc25031349ac1a0e49b67eba8a3242e73b68b307c7bfed"
    assert output["ptq_count"] == 5
    assert output["cpu_only"] is True
