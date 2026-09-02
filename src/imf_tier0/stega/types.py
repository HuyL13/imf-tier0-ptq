from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeSuccess:
    message: bytes


@dataclass(frozen=True, slots=True)
class DecodeFailure:
    reason: str


DecodeResult = DecodeSuccess | DecodeFailure

