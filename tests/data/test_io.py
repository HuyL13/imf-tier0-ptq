import json

import pytest

from imf_tier0.data.io import read_fingerprints, write_jsonl


def test_jsonl_round_trip_and_strict_fingerprint_schema(tmp_path):
    path = tmp_path / "fingerprints.jsonl"
    rows = [{
        "fingerprint_id": f"fp-{i}", "input": f"q{i}", "target": f"a{i}",
        "accepted": True, "human_semantic_approved": True,
    } for i in range(10)]
    write_jsonl(path, rows)
    assert len(read_fingerprints(path)) == 10
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"bad": True}) + "\n")
    with pytest.raises(ValueError):
        read_fingerprints(path)
