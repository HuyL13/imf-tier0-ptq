from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from imf_tier0.config import ExperimentConfig
from imf_tier0.manifest import sha256_file


ROOT = Path(__file__).parents[2]
PPL_SHA256 = "309d4b01d5686143fbdc25031349ac1a0e49b67eba8a3242e73b68b307c7bfed"
SUBMODULES = {
    "gptq": (ROOT / "vendor" / "gptq", "2d65066eeb06a5c9ff5184d8cebdf33662c67faf"),
    "awq": (ROOT / "vendor" / "llm-awq", "d6e797a42b9ef7778de8ee2352116e0f48a78d61"),
}


def _git_executable() -> str:
    found = shutil.which("git")
    if found:
        return found
    windows = Path(r"C:\Program Files\Git\cmd\git.exe")
    if windows.is_file():
        return str(windows)
    raise FileNotFoundError("git executable not found")


def _submodule_head(path: Path) -> str:
    return subprocess.run(
        [_git_executable(), "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def run_preflight(config_path: Path, cpu_only: bool) -> dict[str, object]:
    config = ExperimentConfig.model_validate_yaml(config_path)
    evaluator = ROOT / "tools" / "eval_ppl.py"
    evaluator_sha = sha256_file(evaluator)
    if evaluator_sha != PPL_SHA256:
        raise RuntimeError("supplied eval_ppl.py checksum mismatch")
    submodule_heads = {}
    for name, (path, expected) in SUBMODULES.items():
        actual = _submodule_head(path)
        if actual != expected:
            raise RuntimeError(f"{name} submodule is {actual}, expected {expected}")
        submodule_heads[name] = actual
    return {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "ptq_count": len(config.quantization),
        "ppl_evaluator_sha256": evaluator_sha,
        "submodules": submodule_heads,
        "cpu_only": cpu_only,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imf-tier0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--cpu-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(run_preflight(args.config, args.cpu_only), sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
