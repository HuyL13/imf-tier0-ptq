from __future__ import annotations

import math
from collections.abc import Sequence


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

