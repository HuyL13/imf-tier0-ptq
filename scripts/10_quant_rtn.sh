#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
for bits in 4 3; do out="checkpoints/rtn${bits}_g128"; PYTHONPATH=src python -m imf_ptq.quantize_rtn --upstream third_party/gptq --source checkpoints/source --output "$out" --bits "$bits" --group-size 128; bash scripts/evaluate_checkpoint.sh "rtn${bits}_g128" "$out"; done
