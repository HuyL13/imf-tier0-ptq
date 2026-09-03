# CAU ImF × Llama 3.1 8B PTQ Design

## Scope

Build the ImF standard-PTQ characterization pipeline described by
`CODEX_IMF_PTQ_IMPLEMENTATION_GUIDE (1).md`. The repository starts from the
released CAU implementation at commit `b5ac14cf7086e700a02892a89d375c2d042684e1`.
The backbone is fixed to `meta-llama/Llama-3.1-8B` for source training and all
quantized derivatives. The fixed matrix is RTN3-G128, RTN4-G128, AWQ3-G128,
AWQ4-G128, and GPTQ3-G128; GPTQ4 is prohibited.

This is characterization of ordinary PTQ, not an adaptive removal attack. No
fingerprint-aware calibration, rounding, layer selection, recovery training,
LoRA, QLoRA, or quantized training is allowed.

## Repository and upstream provenance

The experiment is developed on branch `cau-imf-ptq` of the CAU upstream clone
and delivered to `https://github.com/HuyL13/imf-tier0-ptq` under the same branch
name. The original CAU files remain unchanged. The preparation script clones
pinned, intact copies of these repositories under `third_party/`:

- `CAU-ISS-Lab/LLM-Fingerprint-and-Attacks`
- `mit-han-lab/llm-awq`
- `IST-DASLab/gptq`

Their origin URLs and exact commit SHAs are recorded in
`results/environment.json`. Compatibility changes, if unavoidable, are stored
as explicit files under `patches/`; upstream quantization mathematics is never
copied or edited locally.

## Data and source training

`00_prepare.sh` copies the seven released ImF artifacts from the pinned CAU
clone into `data/imf/`, preferring the materialized `train_stego60.json`. It
also copies `C:/Users/yoga/Downloads/eval_ppl.py` byte-for-byte to
`eval/eval_ppl.py` and records its SHA-256.

`01_train_imf.sh` invokes the upstream CAU
`TFA_SVA/Fingerprint_dataset/train_fingerprint.py` through DeepSpeed. Model
loading and tokenizer arguments point to the pinned Llama 3.1 8B revision.
DeepSpeed CPU offload, BF16, and gradient checkpointing are memory-only
adaptations for one A100 40GB. Training retains the upstream causal-language
model objective, the released 60 examples, learning rate `2e-5`, and the
closest pre-PTQ public reproduction configuration. The output is a standalone
Hugging Face checkpoint at `checkpoints/source/`.

## Calibration dependency

AWQ and GPTQ consume only the exact calibration artifact from the completed
IF-SFT experiment. Preparation records its path, configuration, and SHA-256 and
runs an exact normalized-query leakage guard against the ten ImF test inputs.
The artifact has not yet been identified in the currently visible Downloads
workspace. Consequently, AWQ/GPTQ preparation must fail loudly with a precise
missing-dependency message until that exact artifact is supplied; it must not
sample substitute data. RTN remains calibration-free.

## Evaluation

`eval/eval_imf.py` fresh-loads a checkpoint and uses one deterministic decoding
configuration for every setting. It reports per key and in summary:

- exact target match;
- teacher-forced sequence target NLL;
- mean target log-probability;
- native payload fields as `null` unless a legitimate key/decoder/message is
  available.

The released CAU targets do not by themselves establish native payload FSR.
Exact match is therefore explicitly labeled a Level-A released-artifact metric.
`results/limitations.md` records this distinction.

The copied `eval/eval_ppl.py` is the sole WikiText-2 evaluator for source and
all five PTQ settings. Its checksum is asserted before every invocation.

## Quantization and model handoff

RTN3 and RTN4 reuse the completed IF-SFT RTN backend once identified, with
bits 3/4 and group size 128. AWQ3/AWQ4 call official upstream AWQ search and
pseudo quantization; AWQ3 is saved as dequantized quantized-on-grid HF weights
and is never labeled packed INT3. GPTQ3 calls official upstream Llama/GPTQ code
with the exact IF calibration and recorded options.

Every quantizer runs in its own process, saves a reloadable artifact plus
`metadata.json`, and exits. Evaluation then runs in a new process receiving
only the checkpoint path. No metric is accepted from an in-memory quantized
object.

## Guards and failure behavior

Preparation and execution fail on missing checkpoints/data, incorrect test
count, tokenizer mismatch, group size other than 128, GPTQ4, wrong upstream
origins, calibration hash mismatch or leakage, PPL script hash mismatch,
stochastic main evaluation, non-fresh evaluation, mislabeled AWQ3, forbidden
quantizer wrappers, or LoRA/QLoRA usage. A stage writes success metadata only
after its saved checkpoint has been fresh-reloaded.

## Outputs and testing

The implementation adds the required `configs/`, `data/`, `eval/`, `scripts/`,
`src/`, `third_party/`, `checkpoints/`, and `results/` layout plus a root README
with exact reproduction commands. Unit tests cover configuration invariants,
metrics, data parsing, leakage detection, provenance checks, metadata,
aggregation, and path-only process handoff. CPU tests run locally; model and
quantizer integration checks are marked separately for the A100 server.

Implementation follows the guide's stage order: prepare, source train/eval/PPL,
RTN4, RTN3, AWQ4, AWQ3, GPTQ3, then aggregation.
