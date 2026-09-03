from pathlib import Path

STAGES = ["00_prepare", "05_import_if_calibration", "01_train_imf", "02_eval_source",
          "10_quant_rtn", "11_quant_awq", "12_quant_gptq", "30_collect_results"]

def should_skip(marker: Path, validator_exit_code: int) -> bool:
    return marker.is_file() and validator_exit_code == 0
