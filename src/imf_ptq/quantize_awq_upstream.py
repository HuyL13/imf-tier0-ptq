import argparse, json, sys
from pathlib import Path
from .calibration import load_calibration
from .provenance import git_identity, sha256_file

AWQ_URL="https://github.com/mit-han-lab/llm-awq"
def awq_metadata(bits:int, group_size:int, commit:str, calibration_sha256:str)->dict:
    return {"quantizer":"AWQ","bits":bits,"group_size":group_size,"quantizer_commit":commit,
            "calibration_sha256":calibration_sha256,"packed_int3_runtime":False,
            "storage_representation":"HF dequantized quantized-on-grid weights"}

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--upstream",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--calibration",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--bits",type=int,choices=[3,4],required=True); p.add_argument("--group-size",type=int,default=128)
    a=p.parse_args(); identity=git_identity(a.upstream,AWQ_URL); sys.path.insert(0,str(a.upstream))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from awq.quantize.pre_quant import run_awq, apply_awq
    from awq.quantize.quantizer import pseudo_quantize_model_weight
    model=AutoModelForCausalLM.from_pretrained(a.source,torch_dtype=torch.bfloat16,device_map="auto",low_cpu_mem_usage=True); tok=AutoTokenizer.from_pretrained(a.source)
    samples=load_calibration(a.calibration)
    # Official AWQ search accepts calibration text through its documented custom-data boundary.
    q_config={"zero_point":True,"q_group_size":a.group_size}; awq=run_awq(model,tok,w_bit=a.bits,q_config=q_config,n_samples=len(samples),seqlen=2048,calib_data=samples)
    apply_awq(model,awq); pseudo_quantize_model_weight(model,w_bit=a.bits,q_config=q_config)
    a.output.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.output,safe_serialization=True); tok.save_pretrained(a.output)
    meta=awq_metadata(a.bits,a.group_size,identity["commit"],sha256_file(a.calibration)); meta["checkpoint_path"]=str(a.output); (a.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__": main()

