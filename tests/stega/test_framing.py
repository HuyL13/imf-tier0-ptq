import pytest

from imf_tier0.stega.framing import FrameError, frame_payload, unframe_payload


KEY = bytes(range(32))
NONCE = bytes(range(16))


def test_payload_round_trip_is_authenticated() -> None:
    bits = frame_payload(b"owner:lab-a", KEY, NONCE)

    assert unframe_payload(bits, KEY) == b"owner:lab-a"


def test_wrong_key_is_rejected() -> None:
    bits = frame_payload(b"owner:lab-a", KEY, NONCE)

    with pytest.raises(FrameError, match="authentication"):
        unframe_payload(bits, b"x" * 32)


def test_truncated_frame_is_rejected() -> None:
    bits = frame_payload(b"owner:lab-a", KEY, NONCE)

    with pytest.raises(FrameError, match="truncated"):
        unframe_payload(bits[:-9], KEY)


def test_key_must_have_256_bits() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        frame_payload(b"payload", b"short", NONCE)

