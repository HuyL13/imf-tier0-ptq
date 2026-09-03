from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Protocol

from imf_tier0.data.schema import FingerprintRecord
from imf_tier0.model_protocol import TextGenerator
from imf_tier0.pairs.prompts import initial_query_prompt


class TextEncoder(Protocol):
    def encode(self, message: bytes, key: bytes, nonce: bytes) -> str: ...


def generate_pair_candidates(
    codec: TextEncoder,
    auxiliary: TextGenerator,
    ownership_message: bytes,
    secret_key: bytes,
    count: int = 10,
    nonce_factory: Callable[[int], bytes] | None = None,
) -> list[FingerprintRecord]:
    """Generate Algorithm-2 targets and initial queries for mandatory human review."""
    if count != 10:
        raise ValueError("Tier-0 requires exactly ten fingerprint candidates")
    if len(secret_key) != 32:
        raise ValueError("secret key must be exactly 32 bytes")
    if not ownership_message:
        raise ValueError("ownership message must not be empty")
    make_nonce = nonce_factory or (lambda _: secrets.token_bytes(16))
    records: list[FingerprintRecord] = []
    for index in range(1, count + 1):
        nonce = make_nonce(index)
        if len(nonce) != 16:
            raise ValueError("ADG nonce must be exactly 16 bytes")
        target = codec.encode(ownership_message, secret_key, nonce)
        query = auxiliary.generate(initial_query_prompt(target)).strip()
        if not query:
            raise ValueError("auxiliary model returned an empty query")
        records.append(FingerprintRecord(f"fp-{index:02d}", query, target, False, False))
    return records
