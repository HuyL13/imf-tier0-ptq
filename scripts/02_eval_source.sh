#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
bash scripts/evaluate_checkpoint.sh source checkpoints/source
python - <<'PY'
import json
from pathlib import Path
source=json.loads(Path("results/source/imf_summary.json").read_text())
clean=json.loads(Path("results/clean_base/imf_summary.json").read_text())
if source["payload_success"] != 10:
    raise RuntimeError(f"source native payload gate failed: {source['payload_success']}/10")
if clean["payload_success"] != 0:
    raise RuntimeError(f"clean false-verification gate failed: {clean['payload_success']}/10")
PY
