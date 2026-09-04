# ImF ADG Native Verification Design

## Objective

Replace the unrecoverable released ImF `stegoX/stegoY` pairs with a newly generated, self-contained fingerprint set whose ownership payload can be encoded and decoded. Keep the experiment backbone pinned to `meta-llama/Llama-3.1-8B` at revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, retain the existing SFT and PTQ matrix, and report native payload FSR for the clean source and every PTQ checkpoint.

## Scientific contract

- Ownership message: the paper example `This is my model!` encoded as UTF-8.
- Steganography: an ADG encoder/decoder following ImF Appendix J, Algorithm 2 and the cited ADG grouping rule.
- The committed experiment manifest contains the key because this is a reproducibility experiment, not a secrecy deployment.
- Encoder and decoder use the exact same pinned carrier model, tokenizer, revision, prompt prefix, vocabulary filtering, token ordering, and decoding configuration.
- Ten fingerprint pairs are generated. Each carries the same ownership message and has independently recorded generation state.
- Native success for query `i` is exactly `Dec(model(x_i); K) == m`. Exact target match and target NLL remain separately labelled diagnostics.
- False-verification rate is measured by querying the unfingerprinted Llama-3.1-8B base with the same ten queries and applying the same decoder.
- Results generated from the old CAU pairs are retained as historical artifacts but cannot be mixed with the new experiment.

## Components and data flow

1. `imf_ptq.stega.adg` implements deterministic ADG grouping plus keyed bit-to-group mapping.
2. `imf_ptq.stega.framing` provides an unambiguous payload frame containing a format version, message length, message bytes, and authentication tag. This fills in the paper's abstract `BitEnc`/`BitDec` boundary and makes wrong-key/truncated outputs fail closed.
3. A dataset generator loads the pinned Llama-3.1-8B carrier, ADG-encodes ten natural target responses `y`, validates every target by immediate round-trip decoding, and creates semantically aligned CoT-style queries `x` conditioned on each `y`.
4. The generator writes a versioned manifest, `stegoX.txt`, `stegoY.txt`, ten-query JSONL, and the 60-example SFT dataset comprising ten fingerprint pairs plus fifty deterministic normal Alpaca examples.
5. `eval_imf.py` tokenizes generated responses without text normalization, runs the matching ADG decoder, and writes decoded payload, failure reason, payload success, native FSR, exact match, and target NLL.
6. The full-job script runs dataset generation, validates clean-base false positives, trains a fresh source checkpoint, evaluates it, then runs RTN3/4, AWQ3/4, and GPTQ3 with the same native verifier.

## Query construction

The generated `x` follows the paper's CoT-in-input pattern: task description plus explicit lightweight steps and output constraints derived from `y`; the response is not required to contain chain-of-thought. Generation records the construction prompt and output. A deterministic fallback template is allowed only when auxiliary generation fails validation, and such cases are marked in the manifest.

Full iterative refinement is a separate pre-registration stage: candidates are checked against the clean base for false verification before SFT. Post-PTQ outcomes must never be used to revise queries, SFT epochs, calibration, or quantizer settings.

## Reproducibility and failure handling

- Manifest records schema version, key, ownership message, all seeds, model/tokenizer revisions, hashes of generated files, and codec parameters.
- Generation is atomic and refuses to overwrite an existing dataset unless explicitly requested.
- Encoding fails if the payload cannot fit the token budget; decoding fails on token mismatch, incomplete frame, invalid authentication, configuration mismatch, or wrong key.
- Evaluation aborts if the manifest/model revision differs or if any stored target fails round-trip decoding.
- Resume markers include dataset-manifest hashes so old checkpoints/results cannot be silently reused with new pairs.

## Compatibility with repaired pipeline

- Keep the RTN implementation introduced by commit `60e20d6`, which uses upstream GPTQ primitives rather than the previously missing external backend.
- Keep deterministic calibration fallback from `4477b31` and full calibration blocks from `cdbf744`.
- Keep verified ZeRO checkpoint promotion from `cb6555a`.
- Keep AWQ memory/reload fixes through `bbbe00c` and the GPTQ rotary-device fix in `aaf194d`.
- Reorder calibration before PTQ in the new full-job flow so dependency order is explicit.

## Tests and acceptance

- Unit tests cover deterministic grouping, encode/decode round trip, wrong-key rejection, truncation rejection, and altered-token rejection.
- Dataset tests require exactly ten unique fingerprint queries, fifty normal SFT rows, ten fingerprint SFT rows, matching X/Y counts, valid hashes, and successful decoding of all stored targets.
- Evaluator tests cover native success aggregation, failure reasons, false-verification aggregation, and separation from exact match.
- CPU-only tests must pass locally. A GPU smoke test must generate and decode at least one carrier before the full dataset is produced.
- The new source gate is 10/10 native payload successes and 0/10 false verifications on the clean base before PTQ proceeds.
