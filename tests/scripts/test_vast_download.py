from pathlib import Path


def test_vast_download_fetches_base_and_algorithm1_auxiliary_models() -> None:
    script = Path("scripts/vast_download.sh").read_text(encoding="utf-8")

    assert "meta-llama/Llama-3.1-8B" in script
    assert "Qwen/Qwen2.5-7B-Instruct" in script
    assert script.index("meta-llama/Llama-3.1-8B") < script.index("Qwen/Qwen2.5-7B-Instruct")
