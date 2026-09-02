from __future__ import annotations

import random
from collections.abc import Sequence

from imf_tier0.data.schema import FingerprintRecord, NormalRecord, TrainingRecord


def assemble_training_set(
    fingerprints: Sequence[FingerprintRecord],
    normal: Sequence[NormalRecord],
    seed: int,
) -> list[TrainingRecord]:
    if len(fingerprints) != 10 or len(normal) != 50:
        raise ValueError("training data must contain exactly 10 fingerprints and 50 normal records")
    if any(not item.accepted or not item.human_semantic_approved for item in fingerprints):
        raise ValueError("all fingerprints must be accepted and human-approved")
    records = [
        TrainingRecord(item.fingerprint_id, item.input, item.target, "fingerprint")
        for item in fingerprints
    ] + [
        TrainingRecord(item.record_id, item.input, item.target, "normal")
        for item in normal
    ]
    identifiers = [item.record_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("training record IDs must be unique")
    random.Random(seed).shuffle(records)
    return records

