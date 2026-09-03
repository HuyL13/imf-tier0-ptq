#!/usr/bin/env python3
from pathlib import Path
from imf_ptq.results import collect_rows,write_results
root=Path("results"); write_results(root,collect_rows(root))

