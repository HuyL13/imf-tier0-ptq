#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
[[ -f data/calibration/if_calibration.jsonl ]] || { echo "Exact IF calibration dependency missing" >&2; exit 2; }
for bits in 4 3; do out="checkpoints/awq${bits}_g128"; PYTHONPATH=src python -m imf_ptq.quantize_awq_upstream --upstream third_party/llm-awq --source checkpoints/source --calibration data/calibration/if_calibration.jsonl --output "$out" --bits "$bits" --group-size 128; bash scripts/evaluate_checkpoint.sh "awq${bits}_g128" "$out"; done

