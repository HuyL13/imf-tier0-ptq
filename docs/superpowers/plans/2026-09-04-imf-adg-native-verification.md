# ImF ADG Native Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate self-contained ImF fingerprint pairs and report native decoded-payload FSR for the clean base, source checkpoint, and all PTQ checkpoints.

**Architecture:** A model-independent framing layer feeds a paper-aligned ADG codec whose probability source is the pinned Llama-3.1-8B carrier. A versioned manifest binds the key, message, codec parameters, model revision, and generated files; the evaluator reconstructs the codec from that manifest and decodes exact generated token IDs.

**Tech Stack:** Python 3.10+, PyTorch, Transformers 4.46.0, pytest, Bash, existing CAU SFT trainer, upstream GPTQ and llm-awq.

**Spec:** `docs/superpowers/specs/2026-09-04-imf-adg-native-verification-design.md`

## Global Constraints

- Backbone stays `meta-llama/Llama-3.1-8B` revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- Ownership message is UTF-8 `This is my model!`; the experiment key is committed for reproducibility.
- Produce exactly ten fingerprint rows plus fifty deterministic normal SFT rows.
- Native success is only `Dec(model(x_i); K) == m`; exact match and NLL remain separate.
- Do not tune data, SFT, calibration, or quantizers after observing PTQ results.
- Preserve fixes in commits `60e20d6`, `4477b31`, `cdbf744`, `cb6555a`, `bbbe00c`, and `aaf194d`.

---

### Task 1: Authenticated payload framing

**Files:**
- Create: `src/imf_ptq/stega/__init__.py`
- Create: `src/imf_ptq/stega/framing.py`
- Create: `src/imf_ptq/stega/types.py`
- Test: `tests/stega/test_framing.py`

**Interfaces:**
- Produces: `frame_payload(message: bytes, key: bytes) -> list[int]`, `unframe_payload(bits: Sequence[int], key: bytes) -> bytes`, and typed `DecodeSuccess`/`DecodeFailure` results.

- [ ] **Step 1: Write failing framing tests**

```python
def test_frame_round_trip():
    key = bytes(range(32))
    assert unframe_payload(frame_payload(b"This is my model!", key), key) == b"This is my model!"

def test_wrong_key_fails_closed():
    with pytest.raises(FrameError, match="authentication"):
        unframe_payload(frame_payload(b"This is my model!", bytes(32)), bytes([1]) * 32)
```

- [ ] **Step 2: Run `pytest tests/stega/test_framing.py -q` and confirm import failure.**
- [ ] **Step 3: Implement a versioned length-prefixed frame with truncated HMAC-SHA256 and strict binary/key validation.**
- [ ] **Step 4: Run `pytest tests/stega/test_framing.py -q` and confirm all tests pass.**
- [ ] **Step 5: Commit with `git commit -m "feat: add authenticated ImF payload framing"`.**

### Task 2: Paper-aligned ADG codec

**Files:**
- Create: `src/imf_ptq/stega/adg.py`
- Test: `tests/stega/test_adg.py`

**Interfaces:**
- Consumes: framing functions from Task 1 and a `CarrierDistribution.distribution(prefix) -> Sequence[float]` protocol.
- Produces: `adg_group(probabilities) -> list[list[int]]`, `ADGCodec.encode(message, key) -> list[int]`, and `ADGCodec.decode_tokens(tokens, key) -> DecodeResult`.

- [ ] **Step 1: Write failing tests for deterministic grouping, round trip, wrong key, truncation, and a changed token.**

```python
def test_adg_round_trip():
    codec = ADGCodec(UniformCarrier(256), max_tokens=1024, seed=42)
    encoded = codec.encode(b"This is my model!", bytes(range(32)))
    decoded = codec.decode_tokens(encoded, bytes(range(32)))
    assert isinstance(decoded, DecodeSuccess)
    assert decoded.message == b"This is my model!"
```

- [ ] **Step 2: Run `pytest tests/stega/test_adg.py -q` and confirm failure.**
- [ ] **Step 3: Implement Algorithm 2 grouping, keyed group-index permutation, probability-normalized sampling, and fail-closed decoding.**
- [ ] **Step 4: Run framing and ADG tests together.**
- [ ] **Step 5: Commit with `git commit -m "feat: implement ADG steganographic codec"`.**

### Task 3: Transformers carrier and versioned manifest

**Files:**
- Create: `src/imf_ptq/stega/transformers_carrier.py`
- Create: `src/imf_ptq/stega/manifest.py`
- Test: `tests/stega/test_manifest.py`
- Modify: `src/imf_ptq/config.py`

**Interfaces:**
- Produces: `TransformersCarrier(model, tokenizer, prefix_ids, temperature)` and `CodecManifest.load(path)`, `.validate()`, `.sha256()`.
- Manifest fields include schema, model/revision, tokenizer/revision, message UTF-8/base64, key hex, seed, codec limits, prefixes, and artifact hashes.

- [ ] **Step 1: Write failing manifest tests for round trip, malformed key, wrong model revision, and hash mismatch.**
- [ ] **Step 2: Run `pytest tests/stega/test_manifest.py -q` and confirm failure.**
- [ ] **Step 3: Implement carrier inference with stable float32 softmax and manifest validation with canonical JSON hashing.**
- [ ] **Step 4: Run `pytest tests/stega -q`.**
- [ ] **Step 5: Commit with `git commit -m "feat: bind ADG codec to pinned Transformers carrier"`.**

### Task 4: Self-contained fingerprint dataset generator

**Files:**
- Create: `scripts/generate_imf_dataset.py`
- Create: `data/imf_generated/README.md`
- Test: `tests/test_generated_dataset.py`
- Modify: `scripts/00_prepare.sh`

**Interfaces:**
- Consumes: pinned carrier/codec and an Alpaca source.
- Produces: `data/imf_generated/manifest.json`, `stegoX.txt`, `stegoY.txt`, `test_stego10.jsonl`, and `train_stego60.json` via atomic rename.

- [ ] **Step 1: Write a failing dataset-integrity test using an injected fake carrier.**

```python
assert len(test_rows) == 10
assert len(train_rows) == 60
assert sum(row.get("fingerprint_id") is not None for row in train_rows) == 10
assert all(decode_target(row["answer"]) == b"This is my model!" for row in test_rows)
```

- [ ] **Step 2: Run the test and confirm the generator module is missing.**
- [ ] **Step 3: Implement deterministic target generation, immediate round-trip validation, CoT-in-input query construction, fifty normal-row selection, hashes, and overwrite protection.**
- [ ] **Step 4: Update prepare to select `data/imf_generated` and refuse CAU pairs for native runs.**
- [ ] **Step 5: Run all CPU dataset tests and commit with `git commit -m "feat: generate self-contained ImF fingerprint pairs"`.**

### Task 5: Native evaluator and result aggregation

**Files:**
- Modify: `src/imf_ptq/imf_metrics.py`
- Modify: `eval/eval_imf.py`
- Modify: `src/imf_ptq/results.py`
- Modify: `scripts/30_collect_results.py`
- Test: `tests/test_native_metrics.py`

**Interfaces:**
- Produces per-query `decoded_payload`, `decode_failure`, `payload_match`; summary `payload_success`, `payload_rate`, and `false_verification_rate`.

- [ ] **Step 1: Write failing tests proving payload match can succeed when exact target match fails and that false-verification aggregation is separate.**

```python
record = build_record("imf_000", "x", "target", "variant", 1.0, verifier)
assert record["exact_match"] is False
assert record["payload_match"] is True
```

- [ ] **Step 2: Run `pytest tests/test_native_metrics.py -q` and confirm failure.**
- [ ] **Step 3: Load the manifest/carrier in `eval_imf.py`, retain generated token IDs, decode without normalization, and serialize explicit failures.**
- [ ] **Step 4: Extend collection tables without interpreting exact rate as native FSR.**
- [ ] **Step 5: Run all CPU tests and commit with `git commit -m "feat: evaluate native ImF payload FSR"`.**

### Task 6: Safe full-job integration and GPU smoke test

**Files:**
- Create: `scripts/00_generate_imf.sh`
- Create: `scripts/04_eval_clean_base.sh`
- Modify: `scripts/evaluate_checkpoint.sh`
- Modify: `scripts/run_full.sh`
- Modify: `README.md`
- Test: `tests/test_pipeline_contract.py`

**Interfaces:**
- Produces hash-bound stage markers and an end-to-end order: prepare, generate, clean-base negative evaluation, SFT, source evaluation, calibration, RTN, AWQ, GPTQ, collect.

- [ ] **Step 1: Write a failing static pipeline test that checks stage order, manifest arguments, and preservation of all repaired quantizer entry points.**
- [ ] **Step 2: Run `pytest tests/test_pipeline_contract.py -q` and confirm failure.**
- [ ] **Step 3: Wire manifest hashes into resume validation, require 0/10 clean false verifications and 10/10 source native successes, and place calibration before all PTQ stages.**
- [ ] **Step 4: Run `pytest -q` plus `bash -n scripts/*.sh`.**
- [ ] **Step 5: On A100, generate one ADG carrier, decode it, then run clean-base evaluation without starting SFT; record the smoke log.**
- [ ] **Step 6: Commit with `git commit -m "feat: integrate native ImF reproduction pipeline"`.**

### Task 7: Final provenance and regression audit

**Files:**
- Modify: `results/limitations.md`
- Modify: `README.md`

**Interfaces:**
- Consumes all prior components; produces a documented command and provenance record suitable for the subsequent full GPU experiment.

- [ ] **Step 1: Run `pytest -q`, `git diff --check`, and shell syntax checks.**
- [ ] **Step 2: Verify RTN calls `imf_ptq.quantize_rtn`, calibration fallback exists, ZeRO promotion runs, AWQ uses the pristine reload path, and GPTQ includes the rotary-device bridge.**
- [ ] **Step 3: Document that old CAU results are Level-A historical results and new generated results are native Level-B results.**
- [ ] **Step 4: Commit with `git commit -m "docs: document native ImF reproduction workflow"`.**
