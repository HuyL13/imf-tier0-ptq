#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
mkdir -p data/calibration
if [[ -z ${IF_CALIBRATION_FILE:-} || -z ${IF_CALIBRATION_MANIFEST:-} ]]; then
  echo "Exact IF calibration unavailable; building user-authorized deterministic C4 replacement"
  PYTHONPATH=src python scripts/build_standard_calibration.py
  exit 0
fi
PYTHONPATH=src python - "$IF_CALIBRATION_FILE" "$IF_CALIBRATION_MANIFEST" <<'PY'
import json,shutil,sys
from pathlib import Path
from imf_ptq.calibration import load_calibration,verify_manifest,assert_no_query_leakage
from imf_ptq.provenance import atomic_json
src,mp=map(Path,sys.argv[1:]); manifest=json.loads(mp.read_text()); verify_manifest(src,manifest)
queries=[json.loads(x)["text"] for x in Path("data/imf/test_stego10.jsonl").read_text().splitlines()]
assert_no_query_leakage(load_calibration(src),queries); dst=Path("data/calibration/if_calibration.jsonl"); shutil.copyfile(src,dst)
env=json.loads(Path("results/environment.json").read_text()); env["calibration"]=manifest; atomic_json(Path("results/environment.json"),env); atomic_json(Path("data/calibration/manifest.json"),manifest)
PY
