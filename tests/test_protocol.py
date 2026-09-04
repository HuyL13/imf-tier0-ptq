from pathlib import Path
import hashlib
import pytest
from imf_ptq.calibration import assert_no_query_leakage, calibration_manifest, token_block_ids, verify_manifest
from imf_ptq.config import PTQ_MATRIX, load_config
from imf_ptq.quantize_awq_upstream import awq_metadata
from imf_ptq.quantize_gptq_upstream import validate_gptq
from imf_ptq.results import compute_deltas
from imf_ptq.stages import STAGES, should_skip

def test_config():
    cfg=load_config(Path("configs/imf.yaml")); assert cfg.base_model=="meta-llama/Llama-3.1-8B"; assert [(x.method,x.bits,x.group_size) for x in cfg.ptq]==PTQ_MATRIX
def test_leakage():
    with pytest.raises(ValueError,match="fingerprint query"): assert_no_query_leakage([" Secret   Query "],["secret query"])
def test_manifest_hash(tmp_path):
    artifact=tmp_path/"c.jsonl"; artifact.write_text('{"text":"ordinary"}\n'); manifest={"dataset":"x","seed":42,"sequence_length":4,"sample_count":1,"preprocessing":"x","sha256":"0"*64}
    with pytest.raises(ValueError,match="SHA256"): verify_manifest(artifact,manifest)
def test_awq3_label(): assert awq_metadata(3,128,"a","d"*64)["packed_int3_runtime"] is False
def test_gptq_matrix():
    assert validate_gptq(3,128)==(3,128)
    with pytest.raises(ValueError): validate_gptq(4,128)
def test_deltas():
    result=compute_deltas(.8,.6,10,12.5); assert result["delta_exact_rate"]==pytest.approx(-.2); assert result["relative_exact_retention"]==pytest.approx(.75); assert result["delta_ppl_pct"]==25
def test_stages_and_resume(tmp_path):
    assert STAGES[0]=="00_prepare" and STAGES[-1]=="30_collect_results"; marker=tmp_path/"x"; marker.write_text("ok"); assert not should_skip(marker,1)
def test_eval_ppl_is_exact_copy():
    canonical=Path("eval/eval_ppl.py").read_bytes().replace(b"\r\n",b"\n")
    assert hashlib.sha256(canonical).hexdigest()=="23340b0062ba95975556e69cd67e0c9db073c926118477ae3b8b54d6840dd6c8"

def test_replacement_calibration_manifest(tmp_path):
    artifact=tmp_path/"c.jsonl"; artifact.write_text('{"text":"one"}\n{"text":"two"}\n',encoding="utf-8")
    manifest=calibration_manifest(artifact,2,512)
    assert manifest["provenance"]=="deterministic_c4_replacement_authorized"
    assert manifest["dataset_revision"]=="607bd4c8450a42878aa9ddc051a65a055450ef87"
    assert manifest["act_order"] is True and manifest["true_sequential"] is True
    verify_manifest(artifact,manifest)

def test_token_blocks_are_exact_and_limited():
    class Tokenizer:
        def encode(self,text,add_special_tokens=False): return [int(x) for x in text.split()]
    assert token_block_ids(["1 2 3", "4 5 6 7 8"],Tokenizer(),3,2)==[[1,2,3],[4,5,6]]
