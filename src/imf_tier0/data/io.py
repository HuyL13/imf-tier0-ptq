from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from imf_tier0.data.schema import FingerprintRecord, NormalRecord, TrainingRecord


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    values = [asdict(row) if is_dataclass(row) else row for row in rows]
    content = "\n".join(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values)
    _atomic_text(path, content + ("\n" if values else ""))
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(value)
    return rows


def _construct_exact(cls: type[Any], row: dict[str, Any]) -> Any:
    expected = set(cls.__dataclass_fields__)
    if set(row) != expected:
        raise ValueError(f"{cls.__name__} fields mismatch: expected {sorted(expected)}")
    try:
        return cls(**row)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {cls.__name__}: {error}") from error


def read_fingerprints(path: Path) -> list[FingerprintRecord]:
    return [_construct_exact(FingerprintRecord, row) for row in read_jsonl(path)]


def read_normal(path: Path) -> list[NormalRecord]:
    return [_construct_exact(NormalRecord, row) for row in read_jsonl(path)]


def read_training(path: Path) -> list[TrainingRecord]:
    return [_construct_exact(TrainingRecord, row) for row in read_jsonl(path)]
