from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROW_ORDER = ("source", "rtn3", "rtn4", "awq3", "awq4", "gptq3")


@dataclass(frozen=True, slots=True)
class ResultRow:
    name: str
    precision: str
    native_score: float
    soft_score: float | None
    ppl: float


def render_markdown_table(rows: Iterable[ResultRow]) -> str:
    indexed = {row.name: row for row in rows}
    missing = set(ROW_ORDER) - indexed.keys()
    extra = indexed.keys() - set(ROW_ORDER)
    if missing or extra:
        raise ValueError(f"result rows mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    lines = [
        "| Model | Precision | Native fingerprint score | Soft fingerprint score | WikiText2 PPL |",
        "|---|---|---:|---:|---:|",
    ]
    for name in ROW_ORDER:
        row = indexed[name]
        soft = "—" if row.soft_score is None else f"{row.soft_score:.6g}"
        lines.append(
            f"| {row.name} | {row.precision} | {row.native_score:.6g} | {soft} | {row.ppl:.6g} |"
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_reports(rows: Iterable[ResultRow], output_dir: Path) -> dict[str, Path]:
    supplied = list(rows)
    markdown = render_markdown_table(supplied)
    indexed = {row.name: row for row in supplied}
    ordered = [indexed[name] for name in ROW_ORDER]
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    markdown_path = output_dir / "results.md"
    _atomic_write(json_path, json.dumps([asdict(row) for row in ordered], indent=2) + "\n")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(asdict(ordered[0])))
    writer.writeheader()
    writer.writerows(asdict(row) for row in ordered)
    _atomic_write(csv_path, buffer.getvalue())
    _atomic_write(markdown_path, markdown)
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
