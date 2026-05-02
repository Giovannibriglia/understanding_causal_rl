from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from causal_rl.runner.schemas import EVAL_COLUMNS, TRAIN_COLUMNS


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_smoke_outputs_contain_schema_columns() -> None:
    cmd = [sys.executable, "scripts/run_full_matrix.py", "--config", "smoke", "--quick", "--device", "cpu"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    smoke_dirs = sorted(Path("results").glob("smoke_*"))
    assert smoke_dirs
    latest = smoke_dirs[-1]
    train_files = list(latest.glob("**/train.csv"))
    eval_files = list(latest.glob("**/eval.csv"))
    assert train_files and eval_files
    assert _read_header(train_files[0]) == TRAIN_COLUMNS
    assert _read_header(eval_files[0]) == EVAL_COLUMNS
