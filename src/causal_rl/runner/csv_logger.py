from __future__ import annotations

import csv
import os
from pathlib import Path


class CSVLogger:
    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())

    def write(self, row: dict[str, object]) -> None:
        aligned = {c: row.get(c, "") for c in self.columns}
        extra = set(row.keys()) - set(self.columns)
        if extra:
            msg = f"Row contains unknown columns: {sorted(extra)}"
            raise ValueError(msg)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(aligned)
            f.flush()
            os.fsync(f.fileno())
