from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_smoke_end_to_end() -> None:
    cmd = [sys.executable, "scripts/run_full_matrix.py", "--config", "smoke", "--quick", "--device", "cpu"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "IndexError" not in proc.stderr
    assert "RuntimeError" not in proc.stderr
    assert "KeyError" not in proc.stderr

    smoke_dirs = sorted(Path("results").glob("smoke_*"))
    assert smoke_dirs, "No smoke results directory was produced."
    latest = smoke_dirs[-1]
    eval_files = list(latest.glob("**/eval.csv"))
    assert eval_files, "No eval.csv produced under smoke results directory."

    found_data_row = False
    for eval_csv in eval_files:
        with eval_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            found_data_row = True
            break
    assert found_data_row, "All eval.csv files were header-only."
