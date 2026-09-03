from __future__ import annotations

import math
import hashlib
import hmac
from bisect import bisect_left
from collections.abc import Sequence
from typing import Callable, Protocol

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
    if hasattr(probabilities, "dtype"):
        import numpy as np

        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("probability distribution must be a non-empty vector")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("probabilities must be finite and non-negative")
        total = float(values.sum(dtype=np.float64))
        if total <= 0:
            raise ValueError("probability mass must be positive")
        normalized = values / total
        token_ids = np.arange(normalized.size, dtype=np.int64)
        order = np.lexsort((token_ids, normalized))
        remaining = [(float(normalized[token]), int(token)) for token in order]
    else:
        _validate_distribution(probabilities)
        total = math.fsum(probabilities)
        normalized = [value / total for value in probabilities]
        remaining = sorted((probability, token) for token, probability in enumerate(normalized))
    # Sorted probability index makes the paper's nearest-residual choice
    # O(log V), instead of rescanning a Llama-size vocabulary for every group.
    p_max = remaining[-1][0]
    group_count = max(1, 2 ** math.floor(-math.log2(p_max)))
    group_count = min(group_count, len(remaining))
    mean = 1.0 / group_count
    groups: list[list[int]] = []
    remaining_mass = 1.0

    def pop_highest() -> int:
        probability = remaining[-1][0]
        index = bisect_left(remaining, (probability, -1))
        return remaining.pop(index)[1]

    def nearest_index(residual: float) -> int:
        upper = bisect_left(remaining, (residual, -1))
        candidates: list[int] = []
        if upper < len(remaining):
            candidates.append(upper)
        if upper:
            lower_probability = remaining[upper - 1][0]
            candidates.append(bisect_left(remaining, (lower_probability, -1)))
        return min(
            candidates,
            key=lambda index: (abs(remaining[index][0] - residual), remaining[index][1]),
        )

    for index in range(group_count - 1):
        group = [pop_highest()]
        mass = normalized[group[0]]
        remaining_mass -= mass
        while remaining and mass < mean:
            residual = mean - mass
            candidate_index = nearest_index(residual)
            probability, nearest = remaining[candidate_index]
            if probability - residual < residual:
                group.append(nearest)
                remaining.pop(candidate_index)
                mass += probability
                remaining_mass -= probability
            else:
                break
        groups.append(group)
        groups_left = group_count - index - 1
        mean = remaining_mass / groups_left

    groups.append([token for _, token in sorted(remaining, key=lambda item: (-item[0], item[1]))])
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


def _sample_group(
    group: Sequence[int],
    probabilities: Sequence[float],
    key: bytes,
    nonce: bytes,
    step: int,
    attempt: int = 0,
) -> int:
    """Sample w ~ G_j with reproducible key/nonce-derived randomness."""
    total = math.fsum(probabilities[token] for token in group)
    digest = hmac.new(
        key,
        b"adg-sample\x00" + nonce + step.to_bytes(8, "big") + attempt.to_bytes(4, "big"),
        hashlib.sha256,
    ).digest()
    draw = (int.from_bytes(digest, "big") / (1 << 256)) * total
    cumulative = 0.0
    for token in group:
        cumulative += probabilities[token]
        if draw < cumulative:
            return token
    return group[-1]


class ADGCodec:
    """Paper Algorithm 2 over an injectable next-token distribution source."""

    def __init__(
        self,
        carrier: CarrierDistribution,
        max_tokens: int,
        token_validator: Callable[[list[int], int], bool] | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.carrier = carrier
        self.max_tokens = max_tokens
        self.token_validator = token_validator

    def _sample_valid(
        self,
        group: Sequence[int],
        probabilities: Sequence[float],
        key: bytes,
        nonce: bytes,
        step: int,
        prefix: list[int],
    ) -> int:
        remaining = list(group)
        attempt = 0
        while remaining:
            token = _sample_group(remaining, probabilities, key, nonce, step, attempt)
            if self.token_validator is None or self.token_validator(prefix, token):
                return token
            remaining.remove(token)
            attempt += 1
        raise ValueError("ADG group has no tokenizer-stable carrier token")

    def encode(self, message: bytes, key: bytes, nonce: bytes) -> list[int]:
        bits = frame_payload(message, key, nonce)
        prefix: list[int] = []
        offset = 0
        for step in range(self.max_tokens):
            probabilities = self.carrier.distribution(prefix)
            groups = adg_group(probabilities)
            width = int(math.log2(len(groups)))
            if width == 0:
                # Algorithm 2 may temporarily yield u=1. Emit a carrier token
                # without consuming message bits and continue from the new LM
                # state; this step carries zero payload but is still decodable.
                token = self._sample_valid(groups[0], probabilities, key, nonce, step, prefix)
                prefix.append(token)
                continue
            if offset + width > len(bits):
                bits.extend([0] * (offset + width - len(bits)))
            raw_index = _take_bits(bits, offset, width)
            group_index = raw_index ^ _key_mask(key, step, width)
            group = groups[group_index]
            token = self._sample_valid(group, probabilities, key, nonce, step, prefix)
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
                if token not in groups[0]:
                    return DecodeFailure(f"token {token} is outside carrier vocabulary")
                prefix.append(token)
                continue
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
