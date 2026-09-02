import math

import pytest
from hypothesis import given, strategies as st

from imf_tier0.stega.adg import adg_group


def test_uniform_distribution_is_split_into_equal_singletons() -> None:
    assert adg_group([0.25, 0.25, 0.25, 0.25]) == [[0], [1], [2], [3]]


def test_dominant_token_keeps_one_group_and_complete_partition() -> None:
    groups = adg_group([0.6, 0.2, 0.1, 0.1])

    assert groups == [[0, 1, 2, 3]]


@given(
    st.lists(
        st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
        min_size=2,
        max_size=50,
    )
)
def test_groups_partition_vocabulary_exactly_once(weights: list[float]) -> None:
    total = math.fsum(weights)
    probabilities = [value / total for value in weights]

    flattened = [token for group in adg_group(probabilities) for token in group]

    assert sorted(flattened) == list(range(len(probabilities)))
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("bad", [[], [0.0, 0.0], [0.4, -0.1, 0.7]])
def test_invalid_distribution_is_rejected(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        adg_group(bad)

