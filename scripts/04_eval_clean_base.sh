#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
mkdir -p results/clean_base
PYTHONPATH=src python eval/eval_imf.py \
  --model-path meta-llama/Llama-3.1-8B \
  --test-file data/imf/test_stego10.jsonl \
  --manifest data/imf/manifest.json \
  --output-dir results/clean_base \
  --negative-reference \
  --max-new-tokens 512
