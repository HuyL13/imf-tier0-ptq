import math

import pytest

from imf_tier0.eval.metrics import compute_deltas, perplexity_from_nll


def test_metric_deltas_follow_tier0_formulas() -> None:
    result = compute_deltas(source_score=0.8, quant_score=0.6, source_ppl=10.0, quant_ppl=12.5)

    assert result.score_delta == pytest.approx(-0.2)
    assert result.relative_retention == pytest.approx(0.75)
    assert result.ppl_delta_abs == pytest.approx(2.5)
    assert result.ppl_delta_pct == pytest.approx(25.0)


def test_zero_source_score_has_no_relative_retention() -> None:
    assert compute_deltas(0.0, 0.0, 10.0, 10.0).relative_retention is None


def test_perplexity_uses_total_token_weighted_nll() -> None:
    assert perplexity_from_nll(total_nll=math.log(4) * 5, total_tokens=5) == pytest.approx(4.0)

