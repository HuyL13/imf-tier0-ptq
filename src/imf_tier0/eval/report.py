from __future__ import annotations

from dataclasses import dataclass
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

