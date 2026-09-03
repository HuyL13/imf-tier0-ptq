from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


PTQ_ORDER = ("rtn3", "rtn4", "awq3", "awq4", "gptq3")


@dataclass
class PipelineHooks:
    cpu_gate: Callable[[], object]
    preflight: Callable[[], object]
    build_pairs: Callable[[], object]
    assemble_data: Callable[[], object]
    train: Callable[[], object]
    evaluate_source: Callable[[], float]
    quantize_and_evaluate: Callable[[str], object]
    report: Callable[[], object]


def run_pipeline(hooks: PipelineHooks, clean_gate: float) -> None:
    if not 0 <= clean_gate <= 1:
        raise ValueError("clean_gate must be between zero and one")
    hooks.cpu_gate()
    hooks.preflight()
    hooks.build_pairs()
    hooks.assemble_data()
    hooks.train()
    source_score = hooks.evaluate_source()
    if source_score < clean_gate:
        raise RuntimeError(
            f"source payload score {source_score:.4f} failed clean gate {clean_gate:.4f}"
        )
    for quantization in PTQ_ORDER:
        hooks.quantize_and_evaluate(quantization)
    hooks.report()
