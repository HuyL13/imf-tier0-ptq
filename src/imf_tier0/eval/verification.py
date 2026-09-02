from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from imf_tier0.model_protocol import TextDecoder, TextGenerator
from imf_tier0.stega.types import DecodeSuccess


@dataclass(frozen=True, slots=True)
class VerificationCase:
    fingerprint_id: str
    input: str
    target: str


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    fingerprint_id: str
    input: str
    target: str
    generated: str
    native_success: bool
    decoded_payload: str | None
    payload_match: bool
    target_nll: float | None = None
    target_rank: int | None = None
    logit_margin: float | None = None


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    records: tuple[VerificationRecord, ...]
    success_count: int
    total_count: int
    success_rate: float
    false_positive_rate: float | None


def verify_payloads(
    generator: TextGenerator,
    cases: Sequence[VerificationCase],
    decoder: TextDecoder,
    key: bytes,
    expected_message: bytes,
    mode: Literal["fingerprinted", "negative"] = "fingerprinted",
) -> VerificationSummary:
    if not cases:
        raise ValueError("verification requires at least one case")
    records: list[VerificationRecord] = []
    for case in cases:
        generated = generator.generate(case.input)
        decoded = decoder.decode(generated, key)
        decoded_bytes = decoded.message if isinstance(decoded, DecodeSuccess) else None
        matched = decoded_bytes == expected_message
        records.append(
            VerificationRecord(
                fingerprint_id=case.fingerprint_id,
                input=case.input,
                target=case.target,
                generated=generated,
                native_success=matched,
                decoded_payload=(
                    decoded_bytes.decode("utf-8", errors="replace")
                    if decoded_bytes is not None
                    else None
                ),
                payload_match=matched,
            )
        )
    successes = sum(record.payload_match for record in records)
    rate = successes / len(records)
    return VerificationSummary(
        records=tuple(records),
        success_count=successes,
        total_count=len(records),
        success_rate=rate,
        false_positive_rate=rate if mode == "negative" else None,
    )

