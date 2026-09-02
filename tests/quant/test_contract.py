from pathlib import Path

import pytest

from imf_tier0.config import PTQSpec
from imf_tier0.quant.base import SavedCheckpoint
from imf_tier0.quant.registry import build_quantization_command


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "backend,bits,wrapper,extra",
    [
        ("rtn", 3, "run_gptq_upstream.py", ["--nearest"]),
        ("rtn", 4, "run_gptq_upstream.py", ["--nearest"]),
        ("gptq", 3, "run_gptq_upstream.py", []),
        ("awq", 3, "run_awq_upstream.py", []),
        ("awq", 4, "run_awq_upstream.py", []),
    ],
)
def test_commands_pin_bits_g128_and_upstream_wrapper(
    backend: str, bits: int, wrapper: str, extra: list[str]
) -> None:
    command = build_quantization_command(
        python="python",
        spec=PTQSpec(backend=backend, bits=bits, group_size=128),
        source=Path("source"),
        calibration=Path("calibration.jsonl"),
        output=Path("output"),
        seed=42,
    )
    assert command[:2] == ["python", str(ROOT / "tools" / wrapper)]
    assert command[command.index("--bits") + 1] == str(bits)
    assert command[command.index("--group-size") + 1] == "128"
    assert command[command.index("--calibration") + 1] == "calibration.jsonl"
    assert all(value in command for value in extra)


def test_rtn_forbids_calibration_consumption_flag() -> None:
    command = build_quantization_command(
        "python", PTQSpec(backend="rtn", bits=3, group_size=128),
        Path("source"), Path("calibration.jsonl"), Path("output"), 42,
    )
    assert "--nearest" in command
    assert "--use-calibration" not in command


def test_output_must_not_equal_source() -> None:
    with pytest.raises(ValueError, match="different"):
        build_quantization_command(
            "python", PTQSpec(backend="gptq", bits=3, group_size=128),
            Path("same"), Path("cal.jsonl"), Path("same"), 42,
        )


def test_saved_checkpoint_requires_reloadable_hf_files(tmp_path) -> None:
    checkpoint = tmp_path / "quantized"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "quantization_manifest.json").write_text("{}", encoding="utf-8")

    saved = SavedCheckpoint(checkpoint)
    saved.validate_files()


def test_saved_checkpoint_rejects_missing_weights(tmp_path) -> None:
    checkpoint = tmp_path / "broken"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "quantization_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="model weights"):
        SavedCheckpoint(checkpoint).validate_files()
