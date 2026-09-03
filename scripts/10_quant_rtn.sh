#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
[[ -n ${IF_RTN_IMPLEMENTATION:-} ]] || { echo "Exact completed IF-SFT RTN backend missing" >&2; exit 2; }
for bits in 4 3; do out="checkpoints/rtn${bits}_g128"; PYTHONPATH=src python -m imf_ptq.quantize_rtn --backend "$IF_RTN_IMPLEMENTATION" --source checkpoints/source --output "$out" --bits "$bits" --group-size 128; bash scripts/evaluate_checkpoint.sh "rtn${bits}_g128" "$out"; done

