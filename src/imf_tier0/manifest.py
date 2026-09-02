from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InputArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    sha256: str


class StageManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    stage: str
    config: dict[str, Any]
    inputs: dict[str, InputArtifact]

    @classmethod
    def create(
        cls,
        stage: str,
        config: Mapping[str, Any],
        inputs: Mapping[str, str | Path],
    ) -> "StageManifest":
        artifacts = {
            name: InputArtifact(path=str(path), sha256=sha256_file(path))
            for name, path in sorted(inputs.items())
        }
        return cls(stage=stage, config=dict(sorted(config.items())), inputs=artifacts)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

