import json

from imf_tier0.manifest import StageManifest, sha256_file


def test_sha256_file_matches_known_digest(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"abc")

    assert sha256_file(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_manifest_json_is_stable_and_sorted(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    manifest = StageManifest.create(
        stage="assemble-data",
        config={"z": 3, "a": 1},
        inputs={"source": source},
    )

    first = manifest.to_json()
    second = manifest.to_json()

    assert first == second
    assert list(json.loads(first)["config"]) == ["a", "z"]
    assert json.loads(first)["inputs"]["source"]["sha256"] == sha256_file(source)

