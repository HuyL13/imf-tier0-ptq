from dataclasses import dataclass, field

from imf_tier0.pairs.models import PairRequest, PairStatus
from imf_tier0.pairs.refinement import refine_pair
from imf_tier0.stega.types import DecodeFailure, DecodeSuccess


@dataclass
class FakeGenerator:
    outputs: dict[str, list[str]]
    seen: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.seen.append(prompt)
        return self.outputs[prompt].pop(0)


class FakeCodec:
    def decode(self, text: str, key: bytes):
        if text == "carries-owner" and key == b"k" * 32:
            return DecodeSuccess(b"owner")
        return DecodeFailure("no payload")


def request(max_iterations: int = 2, human_approved: bool = True) -> PairRequest:
    return PairRequest(
        fingerprint_id="fp-01",
        target_response="natural carrier",
        ownership_message=b"owner",
        secret_key=b"k" * 32,
        initial_query="query-v1",
        max_iterations=max_iterations,
        human_semantic_approved=human_approved,
    )


def test_accepts_candidate_only_when_positive_passes_and_negatives_fail() -> None:
    target = FakeGenerator({"query-v1": ["carries-owner"]})
    negative = FakeGenerator({"query-v1": ["ordinary"]})
    auxiliary = FakeGenerator({})

    result = refine_pair(request(), target, [negative], auxiliary, FakeCodec())

    assert result.status is PairStatus.ACCEPTED
    assert result.query == "query-v1"
    assert result.iterations == 1


def test_refines_after_positive_decode_failure() -> None:
    target = FakeGenerator(
        {"query-v1": ["ordinary"], "query-v2": ["carries-owner"]}
    )
    negative = FakeGenerator({"query-v1": ["ordinary"], "query-v2": ["ordinary"]})
    auxiliary = FakeGenerator({"REFINE\nquery=query-v1\ntarget=natural carrier\npositive=ordinary\nnegatives=ordinary": ["query-v2"]})

    result = refine_pair(request(), target, [negative], auxiliary, FakeCodec())

    assert result.status is PairStatus.ACCEPTED
    assert result.query == "query-v2"
    assert result.iterations == 2


def test_refines_after_negative_collision() -> None:
    target = FakeGenerator(
        {"query-v1": ["carries-owner"], "query-v2": ["carries-owner"]}
    )
    negative = FakeGenerator(
        {"query-v1": ["carries-owner"], "query-v2": ["ordinary"]}
    )
    auxiliary = FakeGenerator({"REFINE\nquery=query-v1\ntarget=natural carrier\npositive=carries-owner\nnegatives=carries-owner": ["query-v2"]})

    result = refine_pair(request(), target, [negative], auxiliary, FakeCodec())

    assert result.status is PairStatus.ACCEPTED
    assert result.query == "query-v2"


def test_exhausted_failed_candidate_is_rejected() -> None:
    target = FakeGenerator({"query-v1": ["ordinary"]})
    negative = FakeGenerator({"query-v1": ["ordinary"]})

    result = refine_pair(
        request(max_iterations=1), target, [negative], FakeGenerator({}), FakeCodec()
    )

    assert result.status is PairStatus.REJECTED
    assert result.positive_passed is False


def test_human_semantic_gate_is_mandatory() -> None:
    target = FakeGenerator({"query-v1": ["carries-owner"]})
    negative = FakeGenerator({"query-v1": ["ordinary"]})

    result = refine_pair(
        request(human_approved=False), target, [negative], FakeGenerator({}), FakeCodec()
    )

    assert result.status is PairStatus.NEEDS_HUMAN_REVIEW
    assert target.seen == []

