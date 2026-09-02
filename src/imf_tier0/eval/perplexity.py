from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[3]
SUPPLIED_EVALUATOR = PROJECT_ROOT / "tools" / "eval_ppl.py"


def build_ppl_command(
    python: str,
    model_path: Path,
    output: Path,
    seqlen: int = 8192,
    dtype: str = "bf16",
) -> list[str]:
    return [
        python,
        str(SUPPLIED_EVALUATOR),
        "--model-path",
        str(model_path),
        "--datasets",
        "wikitext2",
        "--seqlen",
        str(seqlen),
        "--method",
        "block",
        "--dtype",
        dtype,
        "--out-json",
        str(output),
    ]


def read_wikitext2_ppl(output: Path) -> float:
    payload = json.loads(output.read_text(encoding="utf-8"))
    value = float(payload["ppl"]["wikitext2"])
    if value <= 0:
        raise ValueError("WikiText-2 PPL must be positive")
    return value

