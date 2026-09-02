# ImF Tier-0 PTQ Reproduction Design

## Status and scope

This repository implements the F5/ImF experiment required by
`TIER0_MULTI_FINGERPRINT_PTQ_STUDENT_PLAN.md`. The implementation target is a
faithful, executable reproduction of the public ImF paper on a single NVIDIA
A100 40 GB GPU. It preserves every requirement in the Tier-0 plan and does not
claim byte-for-byte reproduction of unpublished author artifacts.

The paper does not report the exact training learning rate, epoch count, batch
size, seed, 50 normal QA records, negative-reference model set, refinement
budget, or complete prompt templates. Such values must be explicit in versioned
configuration and labeled `not_reported_by_paper`. They must never be presented
as author-provided settings.

Out of scope: adaptive PTQ attacks, fingerprint-aware calibration or rounding,
fine-tuning recovery after PTQ, GRI, model merging, and post-hoc DPO.

## Fixed decisions

- Target backbone: `meta-llama/Llama-3.1-8B`, shared with the F2/F3
  experiments. The exact Hugging Face revision is pinned in the run manifest.
- Auxiliary model: local `Qwen/Qwen2.5-7B-Instruct`, run sequentially.
- Fingerprint injection: full supervised fine-tuning, not LoRA.
- Steganography: ADG following Algorithm 2 of the ImF paper.
- Injection set: exactly 10 fingerprint QA pairs and 50 normal QA examples.
- Hardware target: one NVIDIA A100 40 GB.
- PTQ settings: RTN3-G128, RTN4-G128, AWQ3-G128, AWQ4-G128, GPTQ3-G128.
- Utility metric: WikiText-2 perplexity only.
- PPL implementation: repository copy of the supplied `eval_ppl.py`, preserved
  byte-for-byte and tracked by SHA-256; wrappers may satisfy its imports but
  must not alter the evaluator.
- Primary metric: payload verification success rate.
- Required negative metric: payload false-positive rate on non-fingerprinted
  reference models.

## Scientific invariants

1. The clean fingerprint checkpoint is verified before any PTQ run.
2. PTQ results never influence fingerprint training or pair selection.
3. AWQ and GPTQ consume the same materialized calibration records, order,
   tokenizer configuration, sequence length, and seed.
4. Fingerprint queries and targets are forbidden from the calibration set.
5. Group size is exactly 128 for every quantized checkpoint.
6. Each quantized model is saved, removed from GPU memory, reloaded from disk,
   and only then evaluated.
7. Comparisons are within-model source-versus-quantized comparisons.
8. Native payload decoding remains the primary fingerprint verifier; exact
   string match and target NLL are diagnostic only.

## Architecture

The repository is a Python package with these isolated components:

- `stega`: keyed bit framing, ADG grouping, encoding, decoding, and explicit
  decoding failure states.
- `pair_generation`: initial CoT-augmented query construction, Algorithm 1
  refinement, target-model verification, negative-reference uniqueness checks,
  and candidate manifests.
- `data`: normal-QA ingestion, exact 10+50 dataset assembly, deterministic
  splits, schemas, hashes, and calibration-leakage checks.
- `training`: full SFT for Llama-3.1-8B and standalone checkpoint export.
- `verification`: deterministic generation, ADG payload decoding, payload FSR,
  false-positive rate, exact-match diagnostic, target NLL, and per-key logs.
- `evaluation`: WikiText-2 perplexity using a fixed evaluator configuration.
- `quantization`: backend adapters for RTN, AWQ, and GPTQ with enforced bit and
  group-size settings and save/reload validation.
- `pipeline`: resumable orchestration, clean gate, manifests, checksums, result
  consolidation, and CLI commands.

Dependencies flow inward through typed interfaces. The steganographic codec has
no dependency on training or quantization. Model execution is exposed behind a
generation interface so Algorithm 1 and the verifier can be tested with fakes.
Quantizers share one checkpoint/evaluation contract while retaining their
native implementations.

Official quantizer implementations are pinned Git submodules under `vendor/`:
`IST-DASLab/gptq` supplies GPTQ and its `--nearest` RTN baseline, while
`mit-han-lab/llm-awq` supplies AWQ. Project-owned code is limited to wrappers,
Llama-3.1 integration, orchestration, validation, and serialization; upstream
files are not patched. Every result manifest records the submodule commit SHA.

## Fingerprint construction

For each of ten fingerprints, ADG implements the public paper interface:

1. Frame the registered ownership message into a keyed bit stream.
2. At each generation step, obtain the carrier LM next-token distribution.
3. Apply ADG grouping to the sorted distribution.
4. Use keyed message bits to select a group and sample a token from that group.
5. Continue until the framed payload and termination condition are satisfied.
6. Decode by reconstructing the same conditional distributions and groups,
   recovering the keyed bits, validating framing, and returning either the
   message or a typed failure.

The auxiliary Qwen model derives an initial natural question from each target
response, including lightweight reasoning cues in the input only. It does not
require chain-of-thought output.

Algorithm 1 is implemented as published: generate `y = Enc(m; K)`, initialize
`x = x0`, query the fingerprinted model, decode its response, query every model
in the negative set and decode those responses, accept only when the positive
passes and all negatives fail, otherwise refine with discrepancy feedback until
the configured maximum iterations. A last candidate that fails either gate is
stored as rejected and cannot enter the training dataset. Human validation of
semantic plausibility is a recorded gate, matching the limitation stated by the
paper; it is not silently replaced with an automated score.

## Training and clean gate

The finalized set contains exactly 10 accepted fingerprint records and 50
normal QA records. Training performs full SFT on the combined 60 examples and
saves a standalone checkpoint. All unpublished hyperparameters are set only in
configuration with provenance labels.

After a fresh reload, the source checkpoint is evaluated on:

- payload success count, denominator, and percentage;
- payload false-positive count, denominator, and percentage on negative models;
- exact target match, target NLL, semantic-similarity diagnostic, decoded
  payload, and generated response per fingerprint;
- WikiText-2 perplexity.

The pipeline requires an explicit configured clean-gate threshold before PTQ.
The Tier-0 document does not prescribe a numeric F5 gate, so its chosen value is
labeled `not_reported_by_paper` and reported with results.

## PTQ pipeline

The source checkpoint is independently quantized under exactly five settings:

| ID | Backend | Bits | Group size |
|---|---|---:|---:|
| Q1 | RTN | 3 | 128 |
| Q2 | RTN | 4 | 128 |
| Q3 | AWQ | 3 | 128 |
| Q4 | AWQ | 4 | 128 |
| Q5 | GPTQ | 3 | 128 |

RTN uses no calibration. AWQ and GPTQ use one immutable, materialized
calibration artifact. Before quantization, exact and normalized-text leakage
checks reject any overlap with fingerprint keys or targets. No GPTQ4 experiment
is added.

Each adapter must serialize a loadable checkpoint and metadata describing the
backend package/version, model revision, bits, group size, calibration hash,
seed, sequence length, device, and dtype. Evaluation receives only the reloaded
checkpoint path, never the in-memory quantization object.

## Outputs and analysis

Every evaluation writes JSONL records containing at minimum:

```json
{
  "fingerprint_id": "...",
  "input": "...",
  "target": "...",
  "generated": "...",
  "native_success": true,
  "decoded_payload": "...",
  "payload_match": true,
  "target_nll": null,
  "target_rank": null,
  "logit_margin": null
}
```

Run summaries contain source and quantized payload FSR, false-positive rate,
soft diagnostics, PPL, absolute and percentage PPL deltas, absolute score delta,
and relative retention. Relative retention is labeled as an analysis convenience
rather than a literature-native metric.

The pipeline emits machine-readable JSON/CSV and a Markdown result table with
rows for FP16/BF16 source, RTN3, RTN4, AWQ3, AWQ4, and GPTQ3.

## Failure handling and reproducibility

The pipeline fails closed for invalid secret material, decode exceptions,
unaccepted fingerprint pairs, wrong dataset cardinality, calibration leakage,
missing provenance, unexpected bit width or group size, serialization failure,
and checkpoint reload failure. A decode failure is a verification failure, not
an omitted example.

Each stage writes an atomic completion manifest with input/output hashes. Resume
is permitted only when hashes and configuration match. Secrets and generated
private fingerprint material are ignored by version control; public manifests
store hashes and key identifiers rather than secret keys.

## GPU efficiency requirements

GPU optimization may change execution efficiency but must not change the
fingerprint objective, dataset, native verifier, calibration records, PTQ
matrix, or reported metrics. The implementation uses BF16 on Ampere-or-newer
hardware, gradient checkpointing, fused/scaled-dot-product attention when the
installed stack supports it, length bucketing, dynamic padding, pinned host
memory, non-blocking transfers, pre-tokenized cached datasets, gradient
accumulation, and optimizer-state offload only when required by measured VRAM.

The auxiliary 7B model, target 8B model, evaluator, and quantizer are loaded
sequentially rather than concurrently. Every stage explicitly releases model
references, runs Python garbage collection, empties the CUDA allocator cache,
and records peak allocated/reserved VRAM. Batch size is selected by a
deterministic preflight probe and then frozen before fingerprint training;
quantized outcomes must never influence it. Optional compilation and optimized
attention kernels are enabled only after numerical smoke tests confirm the
native verification output is unchanged under the fixed decoding setup.

## Test strategy

- CPU-first hard gate: all tests that do not require CUDA run first. No model
  download, GPU model load, SFT, or PTQ command may start unless this gate
  passes. The default test command excludes tests marked `gpu`, `model_download`,
  and `slow`; a separate explicit command runs those stages afterward.
- Unit tests: bit framing, keyed mapping, ADG grouping invariants, deterministic
  reconstruction, codec round trip, typed failures, schemas, result formulas,
  and leakage detection.
- Property tests: ADG groups cover the vocabulary exactly once and preserve
  probability mass within numeric tolerance across randomized distributions.
- Integration tests: Algorithm 1 accept/refine/reject paths using deterministic
  fake model runners; exact 10+50 assembly; manifest resume invalidation.
- Backend contract tests: bits/group-size validation and enforcement that
  evaluation accepts only a reloaded path.
- Smoke tests: tiny public model for SFT, serialization, quantization where the
  backend supports it, reload, verification, and PPL plumbing.
- GPU preflight: a dry run validates versions, storage, CUDA capability,
  estimated memory, all local/remote model access, and calibration hashes before
  an expensive run starts.

## Acceptance criteria

The implementation is acceptable when the CPU-only test gate passes before any
GPU work; all remaining automated tests pass; the dry run validates a complete
configuration; ten human-approved ImF pairs and fifty
normal QA records form the training set; the standalone SFT checkpoint passes
the configured clean gate after reload; all five required quantized checkpoints
are independently saved and reloaded; every checkpoint has payload, false
positive, per-key, and PPL outputs; and the consolidated table contains no
unexplained missing cells.

Any result that differs from a paper table is reported as an experimental result,
not rewritten or hidden. Any configuration not explicitly published in the
paper remains visibly labeled as such.
