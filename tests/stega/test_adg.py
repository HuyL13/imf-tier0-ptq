import math

import pytest
from hypothesis import given, strategies as st

from imf_tier0.stega.adg import ADGCodec, adg_group
from imf_tier0.stega.types import DecodeFailure, DecodeSuccess


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


class UniformCarrier:
    def distribution(self, prefix: list[int]) -> list[float]:
        return [1.0 / 256] * 256


def test_adg_codec_round_trips_payload_through_token_choices() -> None:
    codec = ADGCodec(UniformCarrier(), max_tokens=100)
    key = bytes(range(32))

    tokens = codec.encode(b"owner", key, nonce=bytes(range(16)))
    decoded = codec.decode_tokens(tokens, key)

    assert isinstance(decoded, DecodeSuccess)
    assert decoded.message == b"owner"


def test_adg_codec_wrong_key_returns_typed_failure() -> None:
    codec = ADGCodec(UniformCarrier(), max_tokens=100)
    tokens = codec.encode(b"owner", bytes(range(32)), nonce=bytes(range(16)))

    decoded = codec.decode_tokens(tokens, b"x" * 32)

    assert isinstance(decoded, DecodeFailure)
    assert "authentication" in decoded.reason


def test_adg_codec_rejects_carrier_with_insufficient_capacity() -> None:
    codec = ADGCodec(UniformCarrier(), max_tokens=2)

    with pytest.raises(ValueError, match="max_tokens"):
        codec.encode(b"owner", bytes(range(32)), nonce=bytes(range(16)))
