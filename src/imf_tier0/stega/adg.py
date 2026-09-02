from __future__ import annotations

import math
import hashlib
import hmac
from collections.abc import Sequence
from typing import Protocol

from imf_tier0.stega.framing import FrameError, frame_payload, unframe_payload
from imf_tier0.stega.types import DecodeFailure, DecodeResult, DecodeSuccess


class CarrierDistribution(Protocol):
    def distribution(self, prefix: list[int]) -> list[float]: ...


def _validate_distribution(probabilities: Sequence[float]) -> None:
    if not probabilities:
        raise ValueError("probability distribution must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("probabilities must be finite and non-negative")
    if math.fsum(probabilities) <= 0:
        raise ValueError("probability mass must be positive")


def adg_group(probabilities: Sequence[float]) -> list[list[int]]:
    """Reproduce the ADG grouping subroutine in ImF Algorithm 2.

    Token indices are sorted by descending probability with the original index
    as the deterministic tie-breaker. The input may be unnormalized.
    """
    _validate_distribution(probabilities)
    total = math.fsum(probabilities)
    normalized = [value / total for value in probabilities]
    remaining = sorted(
        range(len(normalized)), key=lambda token: (-normalized[token], token)
    )
    p_max = normalized[remaining[0]]
    group_count = max(1, 2 ** math.floor(-math.log2(p_max)))
    group_count = min(group_count, len(remaining))
    mean = 1.0 / group_count
    groups: list[list[int]] = []

    for index in range(group_count - 1):
        group = [remaining.pop(0)]
        mass = normalized[group[0]]
        while remaining and mass < mean:
            residual = mean - mass
            nearest = min(
                remaining,
                key=lambda token: (abs(normalized[token] - residual), token),
            )
            if normalized[nearest] - residual < residual:
                group.append(nearest)
                remaining.remove(nearest)
                mass += normalized[nearest]
            else:
                break
        groups.append(group)
        groups_left = group_count - index - 1
        mean = math.fsum(normalized[token] for token in remaining) / groups_left

    groups.append(remaining)
    return groups


def _key_mask(key: bytes, step: int, width: int) -> int:
    digest = hmac.new(key, step.to_bytes(8, "big"), hashlib.sha256).digest()
    return int.from_bytes(digest, "big") & ((1 << width) - 1)


def _take_bits(bits: list[int], offset: int, width: int) -> int:
    value = 0
    for bit in bits[offset : offset + width]:
        value = (value << 1) | bit
    return value


def _append_value(bits: list[int], value: int, width: int) -> None:
    bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))


class ADGCodec:
    """Paper Algorithm 2 over an injectable next-token distribution source."""

    def __init__(self, carrier: CarrierDistribution, max_tokens: int) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.carrier = carrier
        self.max_tokens = max_tokens

    def encode(self, message: bytes, key: bytes, nonce: bytes) -> list[int]:
        bits = frame_payload(message, key, nonce)
        prefix: list[int] = []
        offset = 0
        for step in range(self.max_tokens):
            probabilities = self.carrier.distribution(prefix)
            groups = adg_group(probabilities)
            width = int(math.log2(len(groups)))
            if width == 0:
                raise ValueError("carrier distribution has zero payload capacity")
            if offset + width > len(bits):
                bits.extend([0] * (offset + width - len(bits)))
            raw_index = _take_bits(bits, offset, width)
            group_index = raw_index ^ _key_mask(key, step, width)
            group = groups[group_index]
            token = max(group, key=lambda item: (probabilities[item], -item))
            prefix.append(token)
            offset += width
            if offset >= len(bits):
                return prefix
        raise ValueError("max_tokens is insufficient for framed payload")

    def decode_tokens(self, tokens: Sequence[int], key: bytes) -> DecodeResult:
        prefix: list[int] = []
        bits: list[int] = []
        for step, token in enumerate(tokens):
            probabilities = self.carrier.distribution(prefix)
            groups = adg_group(probabilities)
            width = int(math.log2(len(groups)))
            if width == 0:
                return DecodeFailure("carrier distribution has zero payload capacity")
            try:
                encoded_index = next(
                    index for index, group in enumerate(groups) if token in group
                )
            except StopIteration:
                return DecodeFailure(f"token {token} is outside carrier vocabulary")
            raw_index = encoded_index ^ _key_mask(key, step, width)
            _append_value(bits, raw_index, width)
            prefix.append(token)

        # Padding is at most seven bits because framing is byte aligned.
        for trim in range(8):
            candidate = bits[: len(bits) - trim] if trim else bits
            try:
                return DecodeSuccess(unframe_payload(candidate, key))
            except FrameError:
                continue
        return DecodeFailure("payload authentication failed or carrier is incomplete")
