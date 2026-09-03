#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
[[ -f data/calibration/if_calibration.jsonl ]] || { echo "Exact IF calibration dependency missing" >&2; exit 2; }
PYTHONPATH=src python -m imf_ptq.quantize_gptq_upstream --upstream third_party/gptq --source checkpoints/source --calibration data/calibration/if_calibration.jsonl --manifest data/calibration/manifest.json --output checkpoints/gptq3_g128 --bits 3 --group-size 128
bash scripts/evaluate_checkpoint.sh gptq3_g128 checkpoints/gptq3_g128

