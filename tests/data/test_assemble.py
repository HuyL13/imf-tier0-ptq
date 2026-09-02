import pytest

from imf_tier0.data.assemble import assemble_training_set
from imf_tier0.data.schema import FingerprintRecord, NormalRecord


def fingerprints(count: int = 10, approved: bool = True):
    return [
        FingerprintRecord(
            fingerprint_id=f"fp-{index:02d}",
            input=f"secret query {index}",
            target=f"carrier response {index}",
            accepted=True,
            human_semantic_approved=approved,
        )
        for index in range(count)
    ]


def normal_records(count: int = 50):
    return [
        NormalRecord(record_id=f"normal-{index:02d}", input=f"q {index}", target=f"a {index}")
        for index in range(count)
    ]


def test_assembles_exact_deterministic_ten_plus_fifty() -> None:
    first = assemble_training_set(fingerprints(), normal_records(), seed=42)
    second = assemble_training_set(fingerprints(), normal_records(), seed=42)

    assert len(first) == 60
    assert [record.record_id for record in first] == [record.record_id for record in second]
    assert sum(record.kind == "fingerprint" for record in first) == 10


@pytest.mark.parametrize("fingerprint_count,normal_count", [(9, 50), (10, 49), (11, 50)])
def test_rejects_wrong_cardinality(fingerprint_count: int, normal_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 10.*50"):
        assemble_training_set(
            fingerprints(fingerprint_count), normal_records(normal_count), seed=42
        )


def test_rejects_unapproved_fingerprint() -> None:
    with pytest.raises(ValueError, match="accepted and human-approved"):
        assemble_training_set(fingerprints(approved=False), normal_records(), seed=42)

