import argparse
import json
import subprocess
from pathlib import Path
from .provenance import sha256_file

def resolve_if_rtn_backend(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError("completed IF-SFT RTN backend was not found")
    return path.resolve()

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--backend", type=Path, required=True); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--bits", type=int, required=True); parser.add_argument("--group-size", type=int, required=True)
    args=parser.parse_args()
    if (args.bits,args.group_size) not in {(3,128),(4,128)}: raise ValueError("RTN accepts only W3/W4 G128")
    backend=resolve_if_rtn_backend(args.backend)
    subprocess.run(["python",str(backend),"--model",str(args.source),"--output",str(args.output),"--wbits",str(args.bits),"--groupsize",str(args.group_size)],check=True)
    meta={"quantizer":"RTN","bits":args.bits,"group_size":128,"backend_sha256":sha256_file(backend),"checkpoint_path":str(args.output)}
    (args.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__ == "__main__": main()

