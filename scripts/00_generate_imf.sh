#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
if PYTHONPATH=src python scripts/generate_imf_dataset.py --validate-only data/imf_generated >/dev/null 2>&1; then
  echo "Generated ImF dataset already valid"
  exit 0
fi
[[ -n ${ALPACA_JSON:-} ]] || { echo "Set ALPACA_JSON to an Alpaca instruction JSON file" >&2; exit 2; }
[[ -n ${AUXILIARY_REVISION:-} ]] || { echo "Set AUXILIARY_REVISION to the immutable 40-hex Llama-3.1-8B-Instruct commit" >&2; exit 2; }
PYTHONPATH=src python scripts/generate_imf_dataset.py --alpaca-json "$ALPACA_JSON" --auxiliary-revision "$AUXILIARY_REVISION"
