import argparse, json, runpy, sys
from pathlib import Path
from .provenance import git_identity, sha256_file

GPTQ_URL="https://github.com/IST-DASLab/gptq"
def validate_gptq(bits:int,group_size:int)->tuple[int,int]:
    if (bits,group_size)!=(3,128): raise ValueError("only GPTQ3-G128 is permitted")
    return bits,group_size
def gptq_metadata(commit:str,calibration_sha256:str,options:dict|None=None)->dict:
    values=options or {"act_order":False,"true_sequential":True,"static_groups":False,"percdamp":0.01,"sym":False}
    return {"quantizer":"GPTQ","bits":3,"group_size":128,"quantizer_commit":commit,"calibration_sha256":calibration_sha256,"options":values,"storage_representation":"HF dequantized quantized-on-grid weights"}
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--upstream",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--calibration",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--bits",type=int,default=3); p.add_argument("--group-size",type=int,default=128); a=p.parse_args(); validate_gptq(a.bits,a.group_size)
    identity=git_identity(a.upstream,GPTQ_URL); manifest=json.loads(a.manifest.read_text()); options={k:manifest[k] for k in ["act_order","true_sequential","static_groups","percdamp","sym"]}
    # Execute the official CLI in-process with only its loader boundary redirected to the fixed artifact.
    sys.path.insert(0,str(a.upstream)); sys.argv=["llama.py",str(a.source),str(a.calibration),"--wbits","3","--groupsize","128","--save",str(a.output/"model.pt")]
    for key,value in options.items():
        if isinstance(value,bool) and value: sys.argv.append("--"+key.replace("_","-"))
        elif not isinstance(value,bool): sys.argv.extend(["--"+key.replace("_","-"),str(value)])
    a.output.mkdir(parents=True,exist_ok=True); runpy.run_path(str(a.upstream/"llama.py"),run_name="__main__")
    meta=gptq_metadata(identity["commit"],sha256_file(a.calibration),options); meta["checkpoint_path"]=str(a.output); (a.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__": main()
