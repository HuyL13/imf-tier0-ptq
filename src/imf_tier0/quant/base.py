from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SavedCheckpoint:
    path: Path

    def validate_files(self) -> None:
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"checkpoint config missing: {self.path}")
        weight_files = list(self.path.glob("*.safetensors")) + list(self.path.glob("pytorch_model*.bin"))
        if not weight_files:
            raise FileNotFoundError(f"checkpoint model weights missing: {self.path}")
        if not (self.path / "quantization_manifest.json").is_file():
            raise FileNotFoundError(f"quantization manifest missing: {self.path}")


def execute_quantization(command: Sequence[str], output: Path) -> SavedCheckpoint:
    subprocess.run(list(command), check=True)
    checkpoint = SavedCheckpoint(output)
    checkpoint.validate_files()
    return checkpoint

