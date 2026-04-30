from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _load_eval_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in results_dir.rglob("eval.csv"):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def make_summary_tables(results_dir: Path, output_dir: Path, env_prefix: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_eval_rows(results_dir)
    if env_prefix is not None:
        rows = [r for r in rows if str(r.get("env_name", "")).startswith(env_prefix)]
    final_by_run: dict[tuple[int, str, str, str, int], dict[str, str]] = {}
    for r in rows:
        try:
            key = (
                int(r["cell"]),
                str(r.get("env_name", "env")),
                str(r["algorithm"]),
                str(r.get("behaviour_policy", "unknown")),
                int(r["seed"]),
            )
            step = int(r["step"])
        except (ValueError, KeyError):
            continue
        prev = final_by_run.get(key)
        if prev is None or step >= int(prev["step"]):
            final_by_run[key] = r

    grouped: dict[tuple[int, str, str, str], list[dict[str, str]]] = {}
    for (cell, env_name, algo, beh, _seed), r in final_by_run.items():
        grouped.setdefault((cell, env_name, algo, beh), []).append(r)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{rllllrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Cell & Env & Algorithm & Behaviour & $R_{\mathrm{eval}}$ & $R_{\mathrm{holdout}}$ & "
        r"$\Delta\_{\mathrm{TV}}$ & $\Delta\_{\mathrm{KL}}$ & $\Delta\_{\chi^2}$ & $\Delta_{\sup}$ \\"
    )
    lines.append(r"\midrule")
    for group_key in sorted(grouped.keys()):
        cell, env_name, algo, beh = group_key
        vals = grouped[group_key]
        eval_r = np.array([float(v.get("eval_return_mean", "nan")) for v in vals], dtype=np.float64)
        hold_r = np.array(
            [float(v.get("eval_holdout_return_mean", "nan")) for v in vals], dtype=np.float64
        )
        tv = np.array([float(v.get("delta_tv", "nan")) for v in vals], dtype=np.float64)
        kl = np.array([float(v.get("delta_kl", "nan")) for v in vals], dtype=np.float64)
        chi2 = np.array([float(v.get("delta_chi2", "nan")) for v in vals], dtype=np.float64)
        sup = np.array([float(v.get("delta_sup", "nan")) for v in vals], dtype=np.float64)
        row = (
            f"{cell} & {env_name} & {algo} & {beh} & "
            f"{np.nanmean(eval_r):.3f} $\\pm$ {np.nanstd(eval_r):.3f} & "
            f"{np.nanmean(hold_r):.3f} $\\pm$ {np.nanstd(hold_r):.3f} & "
            f"{np.nanmean(tv):.3f} $\\pm$ {np.nanstd(tv):.3f} & "
            f"{np.nanmean(kl):.3f} $\\pm$ {np.nanstd(kl):.3f} & "
            f"{np.nanmean(chi2):.3f} $\\pm$ {np.nanstd(chi2):.3f} & "
            f"{np.nanmean(sup):.3f} $\\pm$ {np.nanstd(sup):.3f} \\\\"
        )
        lines.append(row.replace("_", "\_"))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = output_dir / "summary_results.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
