from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PairStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


@dataclass(frozen=True, slots=True)
class PairRequest:
    fingerprint_id: str
    target_response: str
    ownership_message: bytes
    secret_key: bytes
    initial_query: str
    max_iterations: int
    human_semantic_approved: bool

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if len(self.secret_key) != 32:
            raise ValueError("secret_key must be exactly 32 bytes")


@dataclass(frozen=True, slots=True)
class PairResult:
    fingerprint_id: str
    query: str
    target_response: str
    status: PairStatus
    iterations: int
    positive_passed: bool
    negatives_passed: bool

