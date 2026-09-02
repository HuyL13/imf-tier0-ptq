from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricDeltas:
    score_delta: float
    relative_retention: float | None
    ppl_delta_abs: float
    ppl_delta_pct: float


def compute_deltas(
    source_score: float,
    quant_score: float,
    source_ppl: float,
    quant_ppl: float,
) -> MetricDeltas:
    if source_ppl <= 0:
        raise ValueError("source PPL must be positive")
    return MetricDeltas(
        score_delta=quant_score - source_score,
        relative_retention=None if source_score == 0 else quant_score / source_score,
        ppl_delta_abs=quant_ppl - source_ppl,
        ppl_delta_pct=(quant_ppl / source_ppl - 1.0) * 100.0,
    )


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return math.exp(total_nll / total_tokens)

