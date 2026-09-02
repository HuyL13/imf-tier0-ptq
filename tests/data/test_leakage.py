import pytest

from imf_tier0.data.leakage import CalibrationLeakageError, assert_no_calibration_leakage
from imf_tier0.data.schema import FingerprintRecord


FP = FingerprintRecord(
    fingerprint_id="fp-00",
    input="Café   Ownership Query",
    target="Natural Carrier Answer",
    accepted=True,
    human_semantic_approved=True,
)


def test_rejects_normalized_query_overlap() -> None:
    with pytest.raises(CalibrationLeakageError, match="fp-00.*input"):
        assert_no_calibration_leakage(["  CAFE\u0301 ownership   query "], [FP])


def test_rejects_normalized_target_overlap() -> None:
    with pytest.raises(CalibrationLeakageError, match="fp-00.*target"):
        assert_no_calibration_leakage(["natural carrier answer"], [FP])


def test_allows_unrelated_calibration_text() -> None:
    assert_no_calibration_leakage(["unrelated WikiText passage"], [FP])

