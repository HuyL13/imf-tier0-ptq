from dataclasses import dataclass

from imf_tier0.eval.verification import VerificationCase, verify_payloads
from imf_tier0.stega.types import DecodeFailure, DecodeSuccess


@dataclass
class Generator:
    values: dict[str, str]

    def generate(self, prompt: str) -> str:
        return self.values[prompt]


class Decoder:
    def decode(self, text: str, key: bytes):
        if text == "valid":
            return DecodeSuccess(b"owner")
        return DecodeFailure("invalid carrier")


CASES = [
    VerificationCase("fp-1", "q1", "t1"),
    VerificationCase("fp-2", "q2", "t2"),
]


def test_decode_failures_remain_in_fsr_denominator() -> None:
    summary = verify_payloads(
        Generator({"q1": "valid", "q2": "broken"}),
        CASES,
        Decoder(),
        b"k" * 32,
        b"owner",
    )

    assert summary.success_count == 1
    assert summary.total_count == 2
    assert summary.success_rate == 0.5
    assert summary.records[1].decoded_payload is None
    assert summary.records[1].payload_match is False


def test_negative_reference_success_is_reported_as_false_positive() -> None:
    summary = verify_payloads(
        Generator({"q1": "valid", "q2": "broken"}),
        CASES,
        Decoder(),
        b"k" * 32,
        b"owner",
        mode="negative",
    )

    assert summary.false_positive_rate == 0.5

