from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class FingerprintRecord:
    fingerprint_id: str
    input: str
    target: str
    accepted: bool
    human_semantic_approved: bool


@dataclass(frozen=True, slots=True)
class NormalRecord:
    record_id: str
    input: str
    target: str


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    record_id: str
    input: str
    target: str
    kind: Literal["fingerprint", "normal"]

