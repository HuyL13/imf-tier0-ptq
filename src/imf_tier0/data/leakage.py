from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from imf_tier0.data.schema import FingerprintRecord


class CalibrationLeakageError(ValueError):
    pass


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def assert_no_calibration_leakage(
    calibration: Sequence[str], fingerprints: Sequence[FingerprintRecord]
) -> None:
    normalized_calibration = {normalize_text(value) for value in calibration}
    for fingerprint in fingerprints:
        for field in ("input", "target"):
            if normalize_text(getattr(fingerprint, field)) in normalized_calibration:
                raise CalibrationLeakageError(
                    f"calibration overlaps fingerprint {fingerprint.fingerprint_id} {field}"
                )

