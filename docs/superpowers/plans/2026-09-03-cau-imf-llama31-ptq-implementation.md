# CAU ImF Llama 3.1 8B PTQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable CAU ImF source-training and standard PTQ characterization pipeline for Llama 3.1 8B, with one full-job Bash entry point and identical fresh-reload evaluation across source, RTN3/4, AWQ3/4, and GPTQ3.

**Architecture:** The CAU upstream clone remains the repository base; experiment code is isolated in small modules under `src/imf_ptq`, while immutable upstream clones live under `third_party`. Numbered Bash stages exchange only files and manifests, and `scripts/run_full.sh` orchestrates them with durable markers and a complete run log. AWQ/GPTQ call official upstream code through thin adapters; missing exact IF calibration or RTN provenance stops the affected stage explicitly.

**Tech Stack:** Python 3.11+, PyTorch, Transformers, DeepSpeed, pytest, Bash, official CAU ImF artifacts, official mit-han-lab/llm-awq, official IST-DASLab/gptq.

**Spec:** `docs/superpowers/specs/2026-09-03-cau-imf-llama31-ptq-design.md`

## Global Constraints

- Backbone is exactly `meta-llama/Llama-3.1-8B`, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, for source and every PTQ checkpoint.
- Tokenizer comes from the same pinned backbone for all six settings.
- Fixed matrix is RTN3-G128, RTN4-G128, AWQ3-G128, AWQ4-G128, GPTQ3-G128; GPTQ4 is forbidden.
- Source training calls CAU `train_fingerprint.py`; no local trainer, LoRA, QLoRA, or quantized training.
- AWQ and GPTQ call only their official upstream repositories; no AutoAWQ, AutoGPTQ, GPTQModel, Optimum GPTQ, BitsAndBytes, or Quanto implementation.
- The exact IF calibration artifact/config is mandatory for AWQ/GPTQ; never resample a substitute.
- `C:/Users/yoga/Downloads/eval_ppl.py` is copied byte-for-byte and is the only final PPL evaluator.
- Every quantizer saves, exits, and hands a path to a fresh evaluation process.
- Native payload fields remain null unless a legitimate released key/decoder/message is supplied.

---

### Task 1: Package skeleton and immutable experiment configuration

**Files:**
- Create: `pyproject.toml`
- Create: `configs/imf.yaml`
- Create: `src/imf_ptq/__init__.py`
- Create: `src/imf_ptq/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: YAML configuration at `configs/imf.yaml`.
- Produces: `load_config(path: Path) -> ExperimentConfig`, `PTQ_MATRIX`, and `LLAMA_REVISION` used by all later tasks.

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path
import pytest
from imf_ptq.config import PTQ_MATRIX, load_config

def test_fixed_backbone_and_matrix():
    cfg = load_config(Path("configs/imf.yaml"))
    assert cfg.base_model == "meta-llama/Llama-3.1-8B"
    assert cfg.base_revision == "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    assert [(x.method, x.bits, x.group_size) for x in cfg.ptq] == PTQ_MATRIX

def test_gptq4_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("base_model: meta-llama/Llama-3.1-8B\nbase_revision: d04e592bb4f6aa9cfee91e2e20afa771667e1d4b\nptq: [{method: GPTQ, bits: 4, group_size: 128}]\n")
    with pytest.raises(ValueError, match="exact PTQ matrix"):
        load_config(bad)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_config.py`

Expected: collection fails because `imf_ptq.config` does not exist.

- [ ] **Step 3: Implement strict dataclasses and YAML loader**

```python
PTQ_MATRIX = [("RTN", 3, 128), ("RTN", 4, 128), ("AWQ", 3, 128), ("AWQ", 4, 128), ("GPTQ", 3, 128)]
LLAMA_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_mapping(raw)
    if cfg.base_model != "meta-llama/Llama-3.1-8B" or cfg.base_revision != LLAMA_REVISION:
        raise ValueError("backbone must be pinned Meta Llama-3.1-8B")
    if [(x.method, x.bits, x.group_size) for x in cfg.ptq] != PTQ_MATRIX:
        raise ValueError("configuration must match exact PTQ matrix")
    return cfg
```

Set `configs/imf.yaml` to the exact backbone/revision, seed 42, model length 512 (matching CAU default), deterministic max-new-tokens 512, WikiText-2 sequence length 2048, and the five-entry matrix above.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_config.py`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs src/imf_ptq tests/test_config.py
git commit -m "feat: define immutable ImF PTQ protocol"
```

### Task 2: Upstream preparation, released data, and environment manifest

**Files:**
- Create: `src/imf_ptq/provenance.py`
- Create: `scripts/00_prepare.sh`
- Create: `tests/test_provenance.py`
- Create at runtime: `third_party/LLM-Fingerprint-and-Attacks/`
- Create at runtime: `third_party/llm-awq/`
- Create at runtime: `third_party/gptq/`
- Create at runtime: `data/imf/*`
- Create at runtime: `eval/eval_ppl.py`
- Create at runtime: `results/environment.json`

**Interfaces:**
- Consumes: three exact Git URLs, CAU ImF artifact directory, and `/workspace/eval_ppl.py` or `EVAL_PPL_SOURCE`.
- Produces: `git_identity(path: Path) -> dict[str, str]`, `sha256_file(path: Path) -> str`, copied artifacts, and `environment.json`.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_required_origin_is_exact(tmp_path):
    with pytest.raises(ValueError, match="unexpected origin"):
        assert_origin(tmp_path, "https://github.com/mit-han-lab/llm-awq")

def test_sha256_is_content_based(tmp_path):
    item = tmp_path / "x"
    item.write_bytes(b"abc")
    assert sha256_file(item) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_provenance.py`

Expected: fails because provenance functions do not exist.

- [ ] **Step 3: Implement provenance checks and idempotent preparation**

`00_prepare.sh` uses a `clone_or_verify URL PATH` function: clone only when absent, then require `remote.origin.url == URL`, record `rev-parse HEAD`, and never pull implicitly. Copy all seven named ImF files with `install -m 0644`. Resolve PPL source as `${EVAL_PPL_SOURCE:-/workspace/eval_ppl.py}`, require it exists, copy it byte-for-byte, and write hashes/SHAs atomically through `python -m imf_ptq.provenance`.

- [ ] **Step 4: Verify preparation without network mutation**

Run: `pytest -q tests/test_provenance.py && bash -n scripts/00_prepare.sh`

Expected: tests pass and Bash syntax check exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/imf_ptq/provenance.py scripts/00_prepare.sh tests/test_provenance.py
git commit -m "feat: pin upstreams and prepare released ImF data"
```

### Task 3: Exact IF calibration import and leakage guard

**Files:**
- Create: `src/imf_ptq/calibration.py`
- Create: `scripts/05_import_if_calibration.sh`
- Create: `tests/test_calibration.py`
- Create at runtime: `data/calibration/if_calibration.jsonl`
- Modify at runtime: `results/environment.json`

**Interfaces:**
- Consumes: `IF_CALIBRATION_FILE` and `IF_CALIBRATION_MANIFEST`; manifest keys are `dataset`, `seed`, `sequence_length`, `sample_count`, `preprocessing`, and `sha256`.
- Produces: `load_calibration(path: Path) -> list[str]`, `assert_no_query_leakage(calibration, queries) -> None`, and a verified local calibration copy.

- [ ] **Step 1: Write failing leakage and hash tests**

```python
def test_normalized_exact_query_leakage_fails():
    with pytest.raises(ValueError, match="fingerprint query"):
        assert_no_query_leakage(["  Secret   Query "], ["secret query"])

def test_manifest_hash_must_match(tmp_path):
    artifact = tmp_path / "cal.jsonl"
    artifact.write_text('{"text":"ordinary"}\n')
    with pytest.raises(ValueError, match="SHA256"):
        verify_manifest(artifact, {"sha256": "0" * 64})
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_calibration.py`

Expected: fails because calibration functions do not exist.

- [ ] **Step 3: Implement import contract and fail-loud behavior**

Normalization is Unicode NFKC, strip, lowercase, and collapse whitespace. JSONL accepts only a non-empty string `text` field. The shell script exits 2 with `Exact IF calibration dependency missing` when either environment variable is absent, verifies all six manifest fields and SHA, copies the artifact, compares all released test inputs, then records the same hash/config in `environment.json`.

- [ ] **Step 4: Verify GREEN and missing-dependency exit**

Run: `pytest -q tests/test_calibration.py && env -u IF_CALIBRATION_FILE -u IF_CALIBRATION_MANIFEST bash scripts/05_import_if_calibration.sh; test $? -eq 2`

Expected: pytest passes and the script returns 2 after the explicit dependency message.

- [ ] **Step 5: Commit**

```bash
git add src/imf_ptq/calibration.py scripts/05_import_if_calibration.sh tests/test_calibration.py
git commit -m "feat: enforce exact IF calibration provenance"
```

### Task 4: CAU source-training launcher for Llama 3.1 8B

**Files:**
- Create: `patches/cau-llama31-special-tokens.patch`
- Create: `scripts/01_train_imf.sh`
- Create: `scripts/verify_source_checkpoint.py`
- Create: `tests/test_train_launcher.py`

**Interfaces:**
- Consumes: `BASE_MODEL`, released `train_stego60.json`, pinned CAU trainer/config.
- Produces: standalone `checkpoints/source/` and `checkpoints/source/metadata.json`.

- [ ] **Step 1: Write failing launcher contract tests**

```python
def test_launcher_calls_cau_trainer_without_lora():
    text = Path("scripts/01_train_imf.sh").read_text()
    assert "third_party/LLM-Fingerprint-and-Attacks/TFA_SVA/Fingerprint_dataset/train_fingerprint.py" in text
    assert "--model_name_or_path \"$BASE_MODEL\"" in text
    assert "--num_train_epochs 20" in text
    assert "--per_device_train_batch_size 1" in text
    assert "--gradient_accumulation_steps 16" in text
    assert not re.search(r"lora|qlora|bitsandbytes", text, re.I)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_train_launcher.py`

Expected: fails because the launcher does not exist.

- [ ] **Step 3: Add the minimum documented CAU compatibility patch**

Patch only token setup: do not replace Llama 3 special tokens with literal `</s>` and set `pad_token` to existing EOS when absent. Preserve CAU prompt templates, masking, dataset, Trainer, and loss. `01_train_imf.sh` clones a disposable patched worktree from the intact CAU clone, applies the recorded patch, and launches one-GPU DeepSpeed with BF16, gradient checkpointing, effective batch 16, LR `2e-5`, weight decay 0, warmup ratio 0.03, cosine scheduling, and epoch count 20 from public `run.sh` pre-PTQ evidence.

- [ ] **Step 4: Implement fresh-load checkpoint verification**

`verify_source_checkpoint.py` loads config/tokenizer/model in a new process, requires model type `llama`, vocab/tokenizer equality, no PEFT adapter files, and writes `fresh_reload_verified: true` only after a one-token forward pass succeeds.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_train_launcher.py && bash -n scripts/01_train_imf.sh && python -m py_compile scripts/verify_source_checkpoint.py`

Expected: all checks exit 0.

- [ ] **Step 6: Commit**

```bash
git add patches scripts/01_train_imf.sh scripts/verify_source_checkpoint.py tests/test_train_launcher.py
git commit -m "feat: launch CAU ImF SFT on Llama 3.1 8B"
```

### Task 5: Deterministic released-artifact ImF evaluator

**Files:**
- Create: `src/imf_ptq/imf_metrics.py`
- Create: `src/imf_ptq/model_io.py`
- Create: `eval/eval_imf.py`
- Create: `scripts/02_eval_source.sh`
- Test: `tests/test_imf_metrics.py`
- Test: `tests/test_eval_cli.py`

**Interfaces:**
- Consumes: HF checkpoint path, released test JSONL, fixed generation config.
- Produces: `score_target(logits, labels) -> tuple[float, float]`, `imf_per_key.jsonl`, and `imf_summary.json`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_teacher_forced_target_nll_masks_prompt():
    logits = torch.tensor([[[8., 0.], [0., 8.], [8., 0.]]])
    labels = torch.tensor([[-100, 1, 0]])
    nll, mean_logp = score_target(logits, labels)
    assert nll == pytest.approx(-mean_logp)
    assert nll < 0.001

def test_native_fields_are_null_without_verifier():
    row = build_record("imf_000", "q", "a", "a", 0.1, verifier=None)
    assert row["decoded_payload"] is None
    assert row["payload_match"] is None
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_imf_metrics.py tests/test_eval_cli.py`

Expected: fails because evaluator modules do not exist.

- [ ] **Step 3: Implement metrics and strict fresh loader**

Use shifted causal logits and labels, sum only non-`-100` target tokens, and report mean NLL/log-probability. `model_io.py` verifies tokenizer vocab hash against source metadata and rejects an already-instantiated model: its public loader accepts only `Path`.

- [ ] **Step 4: Implement deterministic CLI and source stage**

CLI arguments are `--model-path`, `--test-file`, `--output-dir`, and `--max-new-tokens`; generation hard-codes `do_sample=False`, `num_beams=1`, and no temperature/top-p/top-k. Require exactly 10 records. `02_eval_source.sh` runs evaluator and the copied `eval_ppl.py --model-path checkpoints/source --datasets wikitext2 --seqlen 2048 --dtype bf16`, each as a separate Python process, and stores stdout plus JSON.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_imf_metrics.py tests/test_eval_cli.py && bash -n scripts/02_eval_source.sh`

Expected: tests and syntax check pass.

- [ ] **Step 6: Commit**

```bash
git add src/imf_ptq/imf_metrics.py src/imf_ptq/model_io.py eval/eval_imf.py scripts/02_eval_source.sh tests
git commit -m "feat: evaluate released ImF targets deterministically"
```

### Task 6: RTN3/RTN4 adapter and reload boundary

**Files:**
- Create: `src/imf_ptq/quantize_rtn.py`
- Create: `scripts/10_quant_rtn.sh`
- Create: `scripts/evaluate_checkpoint.sh`
- Test: `tests/test_rtn_adapter.py`
- Test: `tests/test_process_handoff.py`

**Interfaces:**
- Consumes: `IF_RTN_IMPLEMENTATION`, source checkpoint, bits 3/4, group size 128.
- Produces: `checkpoints/rtn3_g128/`, `checkpoints/rtn4_g128/`, metadata, and fresh-process results.

- [ ] **Step 1: Write failing backend and handoff tests**

```python
def test_rtn_rejects_unknown_backend(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed IF-SFT RTN"):
        resolve_if_rtn_backend(tmp_path / "missing.py")

def test_quantizer_never_calls_evaluator_in_process():
    tree = ast.parse(Path("src/imf_ptq/quantize_rtn.py").read_text())
    assert "eval_imf" not in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_rtn_adapter.py tests/test_process_handoff.py`

Expected: fails because RTN files do not exist.

- [ ] **Step 3: Implement strict IF backend delegation**

Require `IF_RTN_IMPLEMENTATION` to identify the completed IF-SFT backend, record its SHA and invocation in metadata, and pass exactly bits/group size/source/output. If absent, exit 2 rather than implement rounding locally. Reject every setting except `(3,128)` and `(4,128)`.

- [ ] **Step 4: Implement shell process boundary**

`10_quant_rtn.sh` invokes quantization twice as separate processes. After each exits and metadata/checkpoint validation passes, `scripts/evaluate_checkpoint.sh SETTING PATH` starts separate ImF and PPL Python processes. The evaluator never imports a quantizer module.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_rtn_adapter.py tests/test_process_handoff.py && bash -n scripts/10_quant_rtn.sh scripts/evaluate_checkpoint.sh`

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/imf_ptq/quantize_rtn.py scripts/10_quant_rtn.sh scripts/evaluate_checkpoint.sh tests
git commit -m "feat: delegate RTN to completed IF backend"
```

### Task 7: Official upstream AWQ3/AWQ4 wrappers

**Files:**
- Create: `src/imf_ptq/quantize_awq_upstream.py`
- Create: `scripts/11_quant_awq.sh`
- Test: `tests/test_awq_wrapper.py`

**Interfaces:**
- Consumes: exact local calibration, source checkpoint, official AWQ clone, bits 3/4, group 128.
- Produces: dense quantized-on-grid HF checkpoints and metadata for AWQ3/AWQ4.

- [ ] **Step 1: Write failing source-provenance tests**

```python
def test_awq_wrapper_imports_upstream_without_equation_copy():
    text = Path("src/imf_ptq/quantize_awq_upstream.py").read_text()
    assert "awq.quantize.pre_quant" in text
    assert "pseudo_quantize_model_weight" in text
    assert "Only 4-bit" not in text

def test_awq3_metadata_disclaims_packed_int3():
    meta = awq_metadata(bits=3, group_size=128, commit="abc", calibration_sha256="d" * 64)
    assert meta["packed_int3_runtime"] is False
    assert meta["storage_representation"] == "HF dequantized quantized-on-grid weights"
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_awq_wrapper.py`

Expected: fails because wrapper functions do not exist.

- [ ] **Step 3: Implement thin upstream adapter**

Add the official clone to `sys.path`, assert its exact origin and print its commit, replace only AWQ's calibration loader boundary with pretokenized blocks from the verified artifact, call upstream `run_awq`, `apply_awq`, and `pseudo_quantize_model_weight`, then `save_pretrained` plus metadata. Do not import packed `WQLinear` for AWQ3.

- [ ] **Step 4: Implement AWQ shell stage**

`11_quant_awq.sh` verifies calibration first, runs AWQ4 then AWQ3 in separate processes, exits each model process, and invokes `evaluate_checkpoint.sh` only by path.

- [ ] **Step 5: Verify GREEN and forbidden dependency scan**

Run: `pytest -q tests/test_awq_wrapper.py && bash -n scripts/11_quant_awq.sh && ! rg -i 'autoawq|auto-gptq|autogptq|gptqmodel|optimum\.gptq' src scripts`

Expected: tests pass, shell parses, forbidden scan finds nothing.

- [ ] **Step 6: Commit**

```bash
git add src/imf_ptq/quantize_awq_upstream.py scripts/11_quant_awq.sh tests/test_awq_wrapper.py
git commit -m "feat: add official upstream AWQ characterization"
```

### Task 8: Official upstream GPTQ3 wrapper

**Files:**
- Create: `src/imf_ptq/quantize_gptq_upstream.py`
- Create: `scripts/12_quant_gptq.sh`
- Test: `tests/test_gptq_wrapper.py`

**Interfaces:**
- Consumes: verified calibration, source checkpoint, official GPTQ clone, fixed W3 G128 options.
- Produces: `checkpoints/gptq3_g128/` dense quantized-on-grid HF checkpoint and metadata.

- [ ] **Step 1: Write failing GPTQ option/provenance tests**

```python
def test_only_gptq3_g128_is_accepted():
    assert validate_gptq(bits=3, group_size=128) == (3, 128)
    with pytest.raises(ValueError, match="GPTQ3-G128"):
        validate_gptq(bits=4, group_size=128)

def test_metadata_records_upstream_options():
    meta = gptq_metadata("abc", "d" * 64)
    assert set(meta["options"]) == {"act_order", "true_sequential", "static_groups", "percdamp", "sym"}
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_gptq_wrapper.py`

Expected: fails because GPTQ wrapper does not exist.

- [ ] **Step 3: Implement upstream Llama sequential integration**

Import the official clone's `llama.py`/`gptq.py`, assert and print origin/commit, adapt only `get_loaders` so the fixed token blocks are supplied, and invoke upstream `llama_sequential` with W3, G128, and the exact option values recovered from the IF calibration manifest. Save the in-place quantized-on-grid weights before optional packing, with metadata describing representation and all five option values.

- [ ] **Step 4: Implement GPTQ shell stage and fresh evaluation**

`12_quant_gptq.sh` verifies calibration, starts quantization, waits for process exit/checkpoint validation, then invokes `evaluate_checkpoint.sh gptq3_g128 checkpoints/gptq3_g128`.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_gptq_wrapper.py && bash -n scripts/12_quant_gptq.sh`

Expected: tests and syntax check pass.

- [ ] **Step 6: Commit**

```bash
git add src/imf_ptq/quantize_gptq_upstream.py scripts/12_quant_gptq.sh tests/test_gptq_wrapper.py
git commit -m "feat: add official upstream GPTQ3 characterization"
```

### Task 9: Result aggregation and scientific limitations

**Files:**
- Create: `src/imf_ptq/results.py`
- Create: `scripts/30_collect_results.py`
- Create: `results/limitations.md`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: six setting directories with ImF, PPL, and metadata JSON.
- Produces: `results/summary.csv` and `results/summary.json` with all required fields/deltas.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_delta_formulas():
    row = compute_deltas(source_exact=0.8, exact=0.6, source_ppl=10.0, ppl=12.5)
    assert row == {"delta_exact_rate": -0.2, "relative_exact_retention": 0.75,
                   "delta_ppl_abs": 2.5, "delta_ppl_pct": 25.0}

def test_collector_requires_exact_setting_order():
    with pytest.raises(ValueError, match="six settings"):
        collect_rows([])
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_results.py`

Expected: fails because result functions do not exist.

- [ ] **Step 3: Implement strict aggregation**

Require order `source,rtn3_g128,rtn4_g128,awq3_g128,awq4_g128,gptq3_g128`; emit every column listed in the guide; use null for unavailable native payload values and for relative retention when source exact rate is zero. Write CSV/JSON atomically.

- [ ] **Step 4: Write limitations document**

State that released `stegoY` has no released key/owner decoder, Level A exact/NLL is not native payload FSR, AWQ3 is quantized-on-grid rather than packed INT3, and AWQ/GPTQ/RTN remain blocked until exact IF artifacts are supplied when applicable.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_results.py && python -m py_compile scripts/30_collect_results.py`

Expected: checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/imf_ptq/results.py scripts/30_collect_results.py results/limitations.md tests/test_results.py
git commit -m "feat: aggregate ImF PTQ characterization results"
```

### Task 10: Full-job Bash runner, durable logging, and resume

**Files:**
- Create: `scripts/run_full.sh`
- Create: `src/imf_ptq/stages.py`
- Test: `tests/test_run_full.py`

**Interfaces:**
- Consumes: numbered stage scripts and their artifact validators.
- Produces: `results/runs/YYYYMMDDTHHMMSSZ/full.log`, per-stage markers, and one full-job exit status.

- [ ] **Step 1: Write failing orchestrator tests**

```python
def test_stage_order_is_exact():
    assert STAGES == ["00_prepare", "05_import_if_calibration", "01_train_imf",
                      "02_eval_source", "10_quant_rtn", "11_quant_awq",
                      "12_quant_gptq", "30_collect_results"]

def test_resume_requires_validator_success(tmp_path):
    marker = tmp_path / "00_prepare.ok"
    marker.write_text("ok")
    assert should_skip(marker, validator_exit_code=1) is False
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_run_full.py`

Expected: fails because stage orchestration does not exist.

- [ ] **Step 3: Implement stage definitions and Bash logger**

The runner uses `set -Eeuo pipefail`, `flock -n 9`, UTC run IDs from `date -u +%Y%m%dT%H%M%SZ`, and `exec > >(tee -a "$RUN_DIR/full.log") 2>&1`. `run_stage NAME COMMAND VALIDATOR` logs `stage_start`, executes the command, records elapsed seconds/exit code, validates artifacts, and atomically writes `results/stages/NAME.ok`. An ERR trap logs the failing stage/line/exit code.

- [ ] **Step 4: Implement validated resume**

`--resume` consults the marker but skips only when the validator command returns 0. Source validator requires fresh reload plus source ImF/PPL results; quantizer validators require checkpoint metadata, fresh-reload evidence, ImF/PPL results, expected hashes, and correct bits/group size.

- [ ] **Step 5: Verify GREEN with fake stage commands**

Run: `pytest -q tests/test_run_full.py && bash -n scripts/run_full.sh`

Expected: tests pass and shell parses.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_full.sh src/imf_ptq/stages.py tests/test_run_full.py
git commit -m "feat: orchestrate and log the full ImF PTQ job"
```

### Task 11: README, full verification, and delivery branch

**Files:**
- Create: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: all prior scripts/configuration.
- Produces: exact reproduction instructions and a clean deliverable branch.

- [ ] **Step 1: Document exact commands**

README commands are:

```bash
bash scripts/00_prepare.sh
bash scripts/05_import_if_calibration.sh
bash scripts/01_train_imf.sh
bash scripts/02_eval_source.sh
bash scripts/10_quant_rtn.sh
bash scripts/11_quant_awq.sh
bash scripts/12_quant_gptq.sh
python scripts/30_collect_results.py
bash scripts/run_full.sh
bash scripts/run_full.sh --resume
```

Document required `HF_TOKEN`, `IF_CALIBRATION_FILE`, `IF_CALIBRATION_MANIFEST`, and `IF_RTN_IMPLEMENTATION`, the A100 40GB requirement, expected storage, log path, and explicit blocked behavior.

- [ ] **Step 2: Run complete CPU verification**

Run: `pytest -q && bash -n scripts/*.sh && python -m compileall -q src eval scripts`

Expected: all tests pass, every shell script parses, and all Python compiles.

- [ ] **Step 3: Run protocol scans**

Run: `! rg -i 'autoawq|auto-gptq|autogptq|gptqmodel|optimum\.gptq|qlora|lora' src scripts && ! find checkpoints -maxdepth 1 -iname '*gptq4*' -print -quit | grep .`

Expected: both negative scans exit 0.

- [ ] **Step 4: Verify repository state and commit docs**

```bash
git diff --check
git status --short
git add README.md .gitignore
git commit -m "docs: add complete ImF PTQ reproduction guide"
```

- [ ] **Step 5: Push the delivery branch**

Run: `git push delivery cau-imf-ptq`

Expected: remote branch advances to the final verified commit.
