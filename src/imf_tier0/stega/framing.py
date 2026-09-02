from __future__ import annotations

import hashlib
import hmac


NONCE_BYTES = 16
LENGTH_BYTES = 4
TAG_BYTES = 16


class FrameError(ValueError):
    """The keyed payload frame cannot be decoded safely."""


def _validate_key(key: bytes) -> None:
    if len(key) != 32:
        raise ValueError("key must be exactly 32 bytes")


def _bytes_to_bits(value: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in value for shift in range(7, -1, -1)]


def _bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise FrameError("truncated payload frame")
    if any(bit not in (0, 1) for bit in bits):
        raise FrameError("payload frame contains non-binary values")
    return bytes(
        sum(bits[offset + index] << (7 - index) for index in range(8))
        for offset in range(0, len(bits), 8)
    )


def frame_payload(message: bytes, key: bytes, nonce: bytes) -> list[int]:
    _validate_key(key)
    if len(nonce) != NONCE_BYTES:
        raise ValueError(f"nonce must be exactly {NONCE_BYTES} bytes")
    body = nonce + len(message).to_bytes(LENGTH_BYTES, "big") + message
    tag = hmac.new(key, body, hashlib.sha256).digest()[:TAG_BYTES]
    return _bytes_to_bits(body + tag)


def unframe_payload(bits: list[int], key: bytes) -> bytes:
    _validate_key(key)
    raw = _bits_to_bytes(bits)
    header_size = NONCE_BYTES + LENGTH_BYTES
    if len(raw) < header_size + TAG_BYTES:
        raise FrameError("truncated payload frame")
    payload_size = int.from_bytes(raw[NONCE_BYTES:header_size], "big")
    expected_size = header_size + payload_size + TAG_BYTES
    if len(raw) < expected_size:
        raise FrameError("truncated payload frame")
    if len(raw) != expected_size:
        raise FrameError("payload frame has trailing data")
    body, supplied_tag = raw[:-TAG_BYTES], raw[-TAG_BYTES:]
    expected_tag = hmac.new(key, body, hashlib.sha256).digest()[:TAG_BYTES]
    if not hmac.compare_digest(supplied_tag, expected_tag):
        raise FrameError("payload authentication failed")
    return raw[header_size:-TAG_BYTES]

