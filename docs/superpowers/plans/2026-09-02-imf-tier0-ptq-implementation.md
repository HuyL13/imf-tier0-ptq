# ImF Tier-0 PTQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-tested, GPU-efficient, paper-faithful ImF pipeline on `meta-llama/Llama-3.1-8B` covering ADG pair construction, full SFT, clean verification, WikiText-2 PPL, and the five required saved-and-reloaded PTQ checkpoints.

**Architecture:** A typed Python package separates cryptographic framing and ADG, model-independent pair refinement, dataset/provenance controls, model execution, evaluation, quantizer adapters, and orchestration. CPU contracts are completed and gated first; optional ML dependencies are imported only inside GPU paths.

**Tech Stack:** Python 3.11, pytest, Hypothesis, Pydantic 2, PyYAML, PyTorch, Transformers, Datasets, Accelerate, TRL, pinned official `IST-DASLab/gptq` and `mit-han-lab/llm-awq` submodules, and the supplied `eval_ppl.py`.

**Spec:** `docs/superpowers/specs/2026-09-02-imf-tier0-ptq-design.md`

## Global Constraints

- Target `meta-llama/Llama-3.1-8B`; pin the resolved Hugging Face revision.
- Exactly 10 accepted fingerprint QA pairs plus 50 normal QA records.
- Full SFT; no LoRA.
- Primary fingerprint metric is keyed payload decoding success; false-positive rate is mandatory.
- PTQ matrix is exactly RTN3-G128, RTN4-G128, AWQ3-G128, AWQ4-G128, GPTQ3-G128; no GPTQ4.
- AWQ/GPTQ share one immutable calibration artifact containing no fingerprint key or target.
- Save, release GPU, and reload every checkpoint before evaluation.
- CPU tests must pass before model download or GPU execution.
- Any parameter absent from the paper is labeled `not_reported_by_paper` in config and manifests.

---

### Task 1: Package, configuration, and provenance contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/imf_tier0/__init__.py`
- Create: `src/imf_tier0/config.py`
- Create: `src/imf_tier0/manifest.py`
- Create: `configs/llama31_8b.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces: `ExperimentConfig.model_validate_yaml(path)`, `StageManifest.create(stage, config, inputs)`, `sha256_file(path)`.

- [ ] **Step 1: Write failing tests** asserting the model ID, 10+50 counts, exact five PTQ tuples, G128, provenance labels, stable file hashes, and rejection of GPTQ4 or missing calibration metadata.
- [ ] **Step 2: Run** `python -m pytest tests/test_config.py tests/test_manifest.py -v`; expect import failure.
- [ ] **Step 3: Implement** strict frozen Pydantic models. Define `PTQSpec(backend: Literal['rtn','awq','gptq'], bits: Literal[3,4], group_size: Literal[128])`, reject any matrix other than `[('rtn',3),('rtn',4),('awq',3),('awq',4),('gptq',3)]`, and serialize sorted JSON manifests atomically.
- [ ] **Step 4: Run the two test files** and expect PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: add experiment configuration and provenance contracts"`.

### Task 2: Keyed payload framing and ADG grouping

**Files:**
- Create: `src/imf_tier0/stega/framing.py`
- Create: `src/imf_tier0/stega/adg.py`
- Create: `src/imf_tier0/stega/types.py`
- Test: `tests/stega/test_framing.py`
- Test: `tests/stega/test_adg.py`

**Interfaces:**
- Produces: `frame_payload(message: bytes, key: bytes, nonce: bytes) -> list[int]`, `unframe_payload(bits, key) -> bytes`, `adg_group(probabilities) -> list[list[int]]`, `ADGCodec.encode(...) -> list[int]`, `ADGCodec.decode(...) -> DecodeResult`.

- [ ] **Step 1: Write failing unit/property tests** for authenticated round trips, wrong-key failure, truncation failure, deterministic keyed group-index mapping, vocabulary partition uniqueness, and probability-mass conservation.
- [ ] **Step 2: Run** `python -m pytest tests/stega -v`; expect import failure.
- [ ] **Step 3: Implement framing** using length-prefixed UTF-8, random nonce, HMAC-SHA256 authentication, and a SHA256 counter-mode keyed bit permutation. Return typed `DecodeFailure` rather than exceptions for invalid carriers.
- [ ] **Step 4: Implement Algorithm 2 grouping**: stable descending probability sort, `u = 2 ** floor(-log2(p_max))` clamped to at least one, greedy nearest-residual assignment for the first `u-1` groups, and remaining tokens in the last group. Encoding/decoding must reconstruct groups from the identical LM distribution callback.
- [ ] **Step 5: Run** `python -m pytest tests/stega -v`; expect PASS.
- [ ] **Step 6: Commit** `git commit -m "feat: implement keyed ADG codec"`.

### Task 3: ImF Algorithm 1 pair generation

**Files:**
- Create: `src/imf_tier0/model_protocol.py`
- Create: `src/imf_tier0/pairs/models.py`
- Create: `src/imf_tier0/pairs/refinement.py`
- Create: `src/imf_tier0/pairs/prompts.py`
- Test: `tests/pairs/test_refinement.py`

**Interfaces:**
- Consumes: `ADGCodec.decode`.
- Produces: `TextGenerator.generate(prompt, decoding) -> str`, `refine_pair(request, target, negatives, auxiliary, codec) -> PairResult`.

- [ ] **Step 1: Write failing tests** with deterministic fake generators for immediate acceptance, positive failure then refinement, negative collision then refinement, exhausted rejection, and mandatory human semantic-approval state.
- [ ] **Step 2: Run** `python -m pytest tests/pairs/test_refinement.py -v`; expect import failure.
- [ ] **Step 3: Implement Algorithm 1 literally**: decode target output; decode every negative output; accept only `pass_pos and pass_neg`; otherwise send target/negative discrepancy feedback to the auxiliary model and repeat through `max_iterations`; never admit the last failed candidate.
- [ ] **Step 4: Add exact prompt templates** as versioned constants and hash them into the pair manifest; mark templates `not_reported_by_paper`.
- [ ] **Step 5: Run the test** and expect PASS.
- [ ] **Step 6: Commit** `git commit -m "feat: implement ImF pair refinement protocol"`.

### Task 4: Dataset assembly and leakage controls

**Files:**
- Create: `src/imf_tier0/data/schema.py`
- Create: `src/imf_tier0/data/assemble.py`
- Create: `src/imf_tier0/data/leakage.py`
- Test: `tests/data/test_assemble.py`
- Test: `tests/data/test_leakage.py`

**Interfaces:**
- Produces: `assemble_training_set(fingerprints, normal, seed) -> list[TrainingRecord]`, `assert_no_calibration_leakage(calibration, fingerprints) -> None`.

- [ ] **Step 1: Write failing tests** requiring exactly ten accepted and human-approved fingerprints, exactly fifty normal records, deterministic shuffle, unique IDs, and rejection of exact or Unicode/case/whitespace-normalized calibration overlap with keys or targets.
- [ ] **Step 2: Run** `python -m pytest tests/data -v`; expect import failure.
- [ ] **Step 3: Implement** immutable schemas, NFKC/casefold/whitespace normalization, SHA256 dataset manifests, and fail-closed cardinality/leakage checks.
- [ ] **Step 4: Run the tests** and expect PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: enforce ImF dataset and calibration controls"`.

### Task 5: Verification, metrics, and WikiText-2 PPL contracts

**Files:**
- Create: `src/imf_tier0/eval/verification.py`
- Create: `src/imf_tier0/eval/metrics.py`
- Create: `src/imf_tier0/eval/perplexity.py`
- Preserve: `tools/eval_ppl.py`
- Create: `rate_endloss_awq/src/run_cache.py`
- Create: `src/imf_tier0/eval/report.py`
- Test: `tests/eval/test_verification.py`
- Test: `tests/eval/test_metrics.py`
- Test: `tests/eval/test_report.py`

**Interfaces:**
- Produces: `verify_payloads(...) -> VerificationSummary`, `compute_deltas(source, quant)`, `evaluate_wikitext2(model_path, config) -> float`, `write_reports(results, output_dir)`.

- [ ] **Step 1: Write failing tests** covering decode failure counted in denominator, payload FSR/FPR, exact match diagnostic, JSONL required fields, PPL/delta formulas, zero-source relative-retention handling, and the six-row Markdown table.
- [ ] **Step 2: Run** `python -m pytest tests/eval -v`; expect import failure.
- [ ] **Step 3: Implement** per-key records with `fingerprint_id,input,target,generated,native_success,decoded_payload,payload_match,target_nll,target_rank,logit_margin`; aggregate success counts and percentages without dropping failures.
- [ ] **Step 4: Integrate the supplied evaluator without edits**; verify its SHA-256 is `309D4B01D5686143FBDC25031349AC1A0E49B67EBA8A3242E73B68B307C7BFED`, implement its missing `rate_endloss_awq.src.run_cache` compatibility dependency, and invoke its primary non-overlapping WikiText-2 block protocol.
- [ ] **Step 5: Implement JSON/JSONL/CSV/Markdown reports** with atomic writes and explicit source/quantized deltas.
- [ ] **Step 6: Run the tests** and expect PASS.
- [ ] **Step 7: Commit** `git commit -m "feat: add ImF verification and utility reporting"`.

### Task 6: GPU runner, full SFT, and memory discipline

**Files:**
- Create: `src/imf_tier0/gpu/runtime.py`
- Create: `src/imf_tier0/gpu/hf_runner.py`
- Create: `src/imf_tier0/training/sft.py`
- Test: `tests/gpu/test_runtime_contract.py`
- Test: `tests/training/test_sft_contract.py`

**Interfaces:**
- Produces: `release_cuda(obj)`, `select_batch_size(probe, candidates)`, `HFTextGenerator`, `run_full_sft(config, dataset_path, output_path)`.

- [ ] **Step 1: Write CPU-only failing contract tests** mocking torch/Transformers to require BF16 on Ampere+, gradient checkpointing, dynamic padding, length grouping, pinned memory, non-blocking transfer, sequential loading, peak-VRAM reporting, and no PEFT/LoRA arguments.
- [ ] **Step 2: Run** `python -m pytest tests/gpu/test_runtime_contract.py tests/training/test_sft_contract.py -v`; expect import failure.
- [ ] **Step 3: Implement lazy imports and runtime cleanup** (`del`, `gc.collect`, `torch.cuda.empty_cache`, peak stats). Freeze batch size after a deterministic preflight probe and write it to the manifest before training.
- [ ] **Step 4: Implement full SFT** with Transformers/TRL, completion-only labels, BF16, gradient checkpointing, SDPA by default, deterministic seeds, checkpoint serialization, and resolved model/tokenizer revisions.
- [ ] **Step 5: Run CPU contract tests** and expect PASS.
- [ ] **Step 6: Commit** `git commit -m "feat: add optimized full-SFT runtime"`.

### Task 7: Required PTQ adapters and reload boundary

**Files:**
- Create: `src/imf_tier0/quant/base.py`
- Create: `src/imf_tier0/quant/rtn.py`
- Create: `src/imf_tier0/quant/awq.py`
- Create: `src/imf_tier0/quant/gptq.py`
- Create: `src/imf_tier0/quant/registry.py`
- Pin submodule: `vendor/gptq`
- Pin submodule: `vendor/llm-awq`
- Test: `tests/quant/test_contract.py`

**Interfaces:**
- Produces: `quantize_checkpoint(spec, source_path, calibration_path, output_path) -> SavedCheckpoint`; evaluators consume only `SavedCheckpoint.path` after `validate_reload()`.

- [ ] **Step 1: Write CPU-only failing adapter tests** that mock each backend, enforce the exact matrix/G128, forbid RTN calibration, require identical AWQ/GPTQ calibration hash, reject fingerprint leakage, and prove the evaluator cannot receive an in-memory model.
- [ ] **Step 2: Run** `python -m pytest tests/quant/test_contract.py -v`; expect import failure.
- [ ] **Step 3: Wrap upstream RTN** using the pinned `IST-DASLab/gptq` `--nearest` path; do not copy or modify its quantization math.
- [ ] **Step 4: Wrap upstream AWQ and GPTQ** from their pinned submodules, passing bits=3/4 and group_size=128 exactly; do not patch vendor files and persist both submodule SHAs.
- [ ] **Step 5: Implement save-release-reload validation** in the shared base: save, hash files, release CUDA, load solely from output path, run a deterministic probe, release again.
- [ ] **Step 6: Run the test** and expect PASS.
- [ ] **Step 7: Commit** `git commit -m "feat: add fixed Tier-0 PTQ matrix"`.

### Task 8: CLI orchestration, CPU gate, and dry run

**Files:**
- Create: `src/imf_tier0/cli.py`
- Create: `src/imf_tier0/pipeline.py`
- Create: `tests/test_pipeline.py`
- Create: `README.md`

**Interfaces:**
- Produces CLI commands `preflight`, `generate-pairs`, `assemble-data`, `train`, `evaluate-source`, `quantize`, `evaluate-quantized`, `report`, and `run`.

- [ ] **Step 1: Write failing pipeline tests** for stage ordering, hash-based resume, clean-gate stop, CPU-gate stop, all five independent PTQ runs, source/quant evaluation, and final report completeness.
- [ ] **Step 2: Run** `python -m pytest tests/test_pipeline.py -v`; expect import failure.
- [ ] **Step 3: Implement orchestration** as explicit stage functions returning manifests. `run` must invoke CPU gate first, then preflight, pair generation/approval, assembly, SFT, source reload/evaluation, clean gate, five quantize/reload/evaluate stages, and reporting.
- [ ] **Step 4: Document exact commands** including Hugging Face authentication, secret-key path, CPU-only tests, GPU smoke markers, resume behavior, artifacts, and public-vs-private files.
- [ ] **Step 5: Run** `python -m pytest -m "not gpu and not model_download and not slow" -v`; expect all CPU tests PASS.
- [ ] **Step 6: Run** `python -m imf_tier0.cli preflight --config configs/llama31_8b.yaml --cpu-only`; expect a successful config/environment report without CUDA initialization or model download.
- [ ] **Step 7: Commit** `git commit -m "feat: orchestrate reproducible ImF Tier-0 pipeline"`.

### Task 9: GPU smoke verification before full experiment

**Files:**
- Modify: `tests/gpu/test_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Consumes all prior public interfaces; produces no new library API.

- [ ] **Step 1: Add marked smoke tests** for authenticated model access, auxiliary/target sequential loading, tiny-data full-SFT save/reload, one-example PPL, and backend availability/version checks without running the full experiment.
- [ ] **Step 2: Run CPU suite again** and require PASS before any GPU command.
- [ ] **Step 3: On the rented server, run** `python -m pytest -m "gpu and not slow" -v`; record GPU name, compute capability, driver, CUDA/PyTorch versions, backend versions, peak VRAM, and PASS/FAIL in `artifacts/preflight/`.
- [ ] **Step 4: Run dry-run CLI** and verify exact model revision, 10+50 requirement, calibration hash, five PTQ entries, storage estimate, and secret-file permissions.
- [ ] **Step 5: Commit** `git commit -m "test: add gated GPU preflight smoke suite"`.
