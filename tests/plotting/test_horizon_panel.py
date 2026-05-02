"""Phase 3 acceptance: horizon-panel plot renders end-to-end.

The plot consumes ``per_cell_per_horizon`` from a synthetic ``summary.json``
that mirrors what the matrix runner emits when ``horizon_sweep`` is active.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from causal_rl.plotting.horizon_panel import render_horizon_panel  # noqa: E402


def _make_summary() -> dict[str, object]:
    horizons = [5, 10, 20, 40, 80]
    per_cell_per_horizon: dict[str, dict[str, float]] = {}
    # Synthetic pattern: id-cells robust to horizon, partial_id-cells degrade.
    for cell in (1, 2, 3, 4):
        for h in horizons:
            base = 15.0 if cell in (1, 2) else 10.0
            decay = 0.0 if cell in (1, 2) else 0.05 * (h - 5)
            d_tv = 0.05 if cell in (1, 2) else 0.10 + 0.005 * h
            per_cell_per_horizon[f"{cell}_h{h}"] = {
                "n_runs": 3,
                "eval_return_mean": base - decay,
                "eval_oracle_return_mean": 18.0,
                "delta_tv_mean": d_tv,
            }
    return {"per_cell_per_horizon": per_cell_per_horizon}


def test_horizon_panel_renders(tmp_path: Path) -> None:
    summary = _make_summary()
    out = tmp_path / "horizon_panel.pdf"
    fig = render_horizon_panel(summary, out)
    assert out.exists()
    title = fig._suptitle.get_text() if fig._suptitle else fig.axes[0].get_title()  # noqa: SLF001
    assert "Horizon impact" in title or "horizon" in title.lower()


def test_horizon_panel_handles_summary_path(tmp_path: Path) -> None:
    from causal_rl.plotting.horizon_panel import render_horizon_panel_from_summary_path

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_make_summary()), encoding="utf-8")
    out = tmp_path / "horizon_panel.pdf"
    render_horizon_panel_from_summary_path(summary_path, out)
    assert out.exists()
