#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
resume=false; [[ ${1:-} == --resume ]] && resume=true; [[ $# -le 1 ]] || { echo "usage: $0 [--resume]" >&2; exit 2; }
mkdir -p results/runs results/stages; exec 9>results/.full-job.lock; flock -n 9 || { echo "another full job is active" >&2; exit 3; }
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ); RUN_DIR="results/runs/$RUN_ID"; mkdir -p "$RUN_DIR"; ln -sfn "$RUN_ID" results/runs/latest
exec > >(tee -a "$RUN_DIR/full.log") 2>&1
stage=bootstrap
trap 'code=$?; echo "event=job_error stage=$stage line=$LINENO exit_code=$code timestamp=$(date -u +%FT%TZ)"; exit $code' ERR
valid(){ case "$1" in 00_generate_imf) PYTHONPATH=src python scripts/generate_imf_dataset.py --validate-only data/imf_generated >/dev/null 2>&1;; 00_prepare) [[ -f results/environment.json && -f eval/eval_ppl.py && -f data/imf/manifest.json ]];; 04_eval_clean_base) [[ -f results/clean_base/imf_summary.json ]];; 05_import_if_calibration) [[ -f data/calibration/if_calibration.jsonl && -f data/calibration/manifest.json ]];; 01_train_imf) [[ -f checkpoints/source/config.json && -f checkpoints/source/metadata.json ]];; 02_eval_source) [[ -f results/source/imf_summary.json && -f results/source/ppl.json ]];; 10_quant_rtn) [[ -f results/rtn3_g128/ppl.json && -f results/rtn4_g128/ppl.json ]];; 11_quant_awq) [[ -f results/awq3_g128/ppl.json && -f results/awq4_g128/ppl.json ]];; 12_quant_gptq) [[ -f results/gptq3_g128/ppl.json ]];; 30_collect_results) [[ -f results/summary.csv && -f results/summary.json ]];; esac; }
manifest_hash(){
  [[ -f data/imf_generated/manifest.json ]] || { printf 'absent\n'; return; }
  PYTHONPATH=src python - <<'PY'
from pathlib import Path
from imf_ptq.stega import CodecManifest
print(CodecManifest.load(Path("data/imf_generated/manifest.json")).sha256())
PY
}
run_stage(){ stage=$1; shift; marker="results/stages/$stage.ok"; expected_hash=$(manifest_hash); if $resume && [[ -f "$marker" ]] && valid "$stage" && grep -qx "manifest_sha256=$expected_hash" "$marker"; then echo "event=stage_skip stage=$stage"; return; fi; rm -f "$marker"; start=$SECONDS; echo "event=stage_start stage=$stage timestamp=$(date -u +%FT%TZ)"; "$@"; valid "$stage"; expected_hash=$(manifest_hash); printf 'manifest_sha256=%s\n' "$expected_hash" > "$marker.tmp"; mv "$marker.tmp" "$marker"; echo "event=stage_complete stage=$stage elapsed_seconds=$((SECONDS-start)) timestamp=$(date -u +%FT%TZ)"; }
run_stage 00_generate_imf bash scripts/00_generate_imf.sh
run_stage 00_prepare bash scripts/00_prepare.sh
run_stage 04_eval_clean_base bash scripts/04_eval_clean_base.sh
run_stage 01_train_imf bash scripts/01_train_imf.sh
run_stage 02_eval_source bash scripts/02_eval_source.sh
run_stage 05_import_if_calibration bash scripts/05_import_if_calibration.sh
run_stage 10_quant_rtn bash scripts/10_quant_rtn.sh
run_stage 11_quant_awq bash scripts/11_quant_awq.sh
run_stage 12_quant_gptq bash scripts/12_quant_gptq.sh
run_stage 30_collect_results env PYTHONPATH=src python scripts/30_collect_results.py
echo "event=job_complete run_id=$RUN_ID log=$RUN_DIR/full.log"
