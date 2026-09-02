from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _manifest_path(root: Path, stage: str, manifest_name: str | None) -> Path:
    return root / (manifest_name or f"{stage}.manifest.json")


def _canonical(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(params: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(params).encode("utf-8")).hexdigest()


def write_manifest(
    root: str | Path,
    stage: str,
    params: dict[str, Any],
    manifest_name: str | None = None,
) -> None:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    target = _manifest_path(directory, stage, manifest_name)
    payload = {"stage": stage, "params": params, "fingerprint": _fingerprint(params)}
    fd, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def cache_hit(
    root: str | Path,
    stage: str,
    params: dict[str, Any],
    required_outputs: Iterable[str],
    overwrite: bool = False,
    manifest_name: str | None = None,
) -> tuple[bool, str]:
    if overwrite:
        return False, "overwrite requested"
    directory = Path(root)
    manifest = _manifest_path(directory, stage, manifest_name)
    if not manifest.exists():
        return False, "manifest missing"
    if any(not (directory / output).exists() for output in required_outputs):
        return False, "required outputs missing"
    try:
        stored = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "manifest unreadable"
    if stored.get("fingerprint") != _fingerprint(params):
        return False, "parameters changed"
    return True, "manifest and outputs match"

