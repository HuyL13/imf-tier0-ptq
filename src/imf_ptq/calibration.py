import json
import unicodedata
from pathlib import Path
from .provenance import sha256_file

REQUIRED_MANIFEST = {"dataset", "seed", "sequence_length", "sample_count", "preprocessing", "sha256"}

def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().lower().split())

def load_calibration(path: Path) -> list[str]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value.get("text"), str) or not value["text"].strip():
            raise ValueError(f"calibration row {number} requires non-empty text")
        rows.append(value["text"])
    if not rows:
        raise ValueError("empty calibration artifact")
    return rows

def verify_manifest(path: Path, manifest: dict) -> None:
    missing = REQUIRED_MANIFEST - manifest.keys()
    if missing:
        raise ValueError(f"calibration manifest missing {sorted(missing)}")
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError("calibration SHA256 mismatch")
    if len(load_calibration(path)) != int(manifest["sample_count"]):
        raise ValueError("calibration sample_count mismatch")

def assert_no_query_leakage(calibration: list[str], queries: list[str]) -> None:
    corpus = [normalize(x) for x in calibration]
    for query in queries:
        needle = normalize(query)
        if any(needle == item or needle in item for item in corpus):
            raise ValueError("fingerprint query appears in calibration")

