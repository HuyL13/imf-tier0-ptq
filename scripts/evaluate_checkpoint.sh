#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 SETTING CHECKPOINT" >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"; setting=$1; checkpoint=$2
[[ -f "$checkpoint/config.json" ]] || { echo "checkpoint is not reloadable: $checkpoint" >&2; exit 2; }
mkdir -p "results/$setting"
PYTHONPATH=src python eval/eval_imf.py --model-path "$checkpoint" --test-file data/imf/test_stego10.jsonl --manifest data/imf/manifest.json --clean-summary results/clean_base/imf_summary.json --output-dir "results/$setting" --max-new-tokens 512
PYTHONPATH=. python eval/eval_ppl.py --model-path "$checkpoint" --datasets wikitext2 --seqlen 2048 --dtype bf16 --out-json "results/$setting/ppl.json" --tag "$setting"
if [[ -f "$checkpoint/metadata.json" ]]; then cp "$checkpoint/metadata.json" "results/$setting/metadata.json"; else printf '{"quantizer":"source","checkpoint_path":"%s"}\n' "$checkpoint" > "results/$setting/metadata.json"; fi
