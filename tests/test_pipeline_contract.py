from pathlib import Path


def test_full_pipeline_orders_generation_clean_gate_and_calibration_before_ptq():
    script = Path("scripts/run_full.sh").read_text(encoding="utf-8")
    expected = [
        "00_generate_imf", "00_prepare", "04_eval_clean_base", "01_train_imf",
        "02_eval_source", "05_import_if_calibration", "10_quant_rtn",
        "11_quant_awq", "12_quant_gptq", "30_collect_results",
    ]
    positions = [script.index(f"run_stage {stage}") for stage in expected]
    assert positions == sorted(positions)
    assert "manifest_sha256=" in script
    assert 'grep -qx "manifest_sha256=$expected_hash" "$marker"' in script


def test_all_checkpoint_evaluations_receive_manifest_and_clean_summary():
    script = Path("scripts/evaluate_checkpoint.sh").read_text(encoding="utf-8")
    assert "--manifest data/imf/manifest.json" in script
    assert "--clean-summary results/clean_base/imf_summary.json" in script


def test_repaired_quantizer_paths_are_preserved():
    assert "imf_ptq.quantize_rtn" in Path("scripts/10_quant_rtn.sh").read_text(encoding="utf-8")
    assert "imf_ptq.quantize_awq_upstream" in Path("scripts/11_quant_awq.sh").read_text(encoding="utf-8")
    assert "imf_ptq.quantize_gptq_upstream" in Path("scripts/12_quant_gptq.sh").read_text(encoding="utf-8")
    assert "promote_zero3_checkpoint.py" in Path("scripts/01_train_imf.sh").read_text(encoding="utf-8")


def test_source_gate_requires_native_ten_of_ten_and_clean_zero_of_ten():
    script = Path("scripts/02_eval_source.sh").read_text(encoding="utf-8")
    assert 'payload_success"] != 10' in script
    assert 'payload_success"] != 0' in script


def test_evaluator_releases_evaluated_model_before_loading_carrier():
    source = Path("eval/eval_imf.py").read_text(encoding="utf-8")
    assert "del prompt_batch, prompt_ids, attention_mask, full, labels, output, generated" in source
    assert "del model\n    release_cuda()\n\n    carrier_model" in source
