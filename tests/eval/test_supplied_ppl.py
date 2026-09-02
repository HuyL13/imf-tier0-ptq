import hashlib
import json
from pathlib import Path

from imf_tier0.eval.perplexity import build_ppl_command, read_wikitext2_ppl
from rate_endloss_awq.src.run_cache import cache_hit, write_manifest


ROOT = Path(__file__).parents[2]
EXPECTED_SHA256 = "309d4b01d5686143fbdc25031349ac1a0e49b67eba8a3242e73b68b307c7bfed"


def test_supplied_evaluator_is_preserved_byte_for_byte() -> None:
    content = (ROOT / "tools" / "eval_ppl.py").read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_SHA256


def test_wrapper_uses_primary_wikitext2_block_protocol() -> None:
    command = build_ppl_command(
        python="python", model_path=Path("checkpoint"), output=Path("ppl.json")
    )
    assert command == [
        "python", str(ROOT / "tools" / "eval_ppl.py"),
        "--model-path", "checkpoint", "--datasets", "wikitext2",
        "--seqlen", "8192", "--method", "block", "--dtype", "bf16",
        "--out-json", "ppl.json",
    ]


def test_reads_only_wikitext2_result(tmp_path) -> None:
    output = tmp_path / "ppl.json"
    output.write_text(json.dumps({"ppl": {"wikitext2": 7.25}}), encoding="utf-8")
    assert read_wikitext2_ppl(output) == 7.25


def test_compatibility_cache_round_trip(tmp_path) -> None:
    output = tmp_path / "ppl.json"
    output.write_text("{}", encoding="utf-8")
    params = {"model": "checkpoint", "seqlen": 8192}
    write_manifest(tmp_path, "eval_ppl", params)

    hit, reason = cache_hit(tmp_path, "eval_ppl", params, ["ppl.json"])
    assert hit is True
    assert reason == "manifest and outputs match"


def test_cache_misses_when_parameters_change(tmp_path) -> None:
    (tmp_path / "ppl.json").write_text("{}", encoding="utf-8")
    write_manifest(tmp_path, "eval_ppl", {"seqlen": 8192})
    hit, reason = cache_hit(
        tmp_path, "eval_ppl", {"seqlen": 2048}, ["ppl.json"]
    )
    assert hit is False
    assert "parameters" in reason

