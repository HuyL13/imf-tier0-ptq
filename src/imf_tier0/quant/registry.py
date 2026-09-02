from __future__ import annotations

from pathlib import Path

from imf_tier0.config import PTQSpec


PROJECT_ROOT = Path(__file__).parents[3]


def build_quantization_command(
    python: str,
    spec: PTQSpec,
    source: Path,
    calibration: Path,
    output: Path,
    seed: int,
) -> list[str]:
    if source.resolve() == output.resolve():
        raise ValueError("source and output checkpoint paths must be different")
    wrapper = "run_awq_upstream.py" if spec.backend == "awq" else "run_gptq_upstream.py"
    command = [
        python,
        str(PROJECT_ROOT / "tools" / wrapper),
        "--model-path", str(source),
        "--calibration", str(calibration),
        "--output", str(output),
        "--bits", str(spec.bits),
        "--group-size", str(spec.group_size),
        "--seed", str(seed),
    ]
    if spec.backend == "rtn":
        command.append("--nearest")
    return command

