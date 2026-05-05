from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.plotting.ablations import make_ablations  # noqa: E402
from causal_rl.plotting.bandit_regret import make_bandit_regret  # noqa: E402
from causal_rl.plotting.bias_sweep import make_bias_sweep  # noqa: E402
from causal_rl.plotting.bound_width_panel import make_bound_width_panel  # noqa: E402
from causal_rl.plotting.causal_metrics_grid import make_causal_metrics_grid  # noqa: E402
from causal_rl.plotting.cell_grid import make_cell_grid  # noqa: E402
from causal_rl.plotting.coverage_signature import make_coverage_signature  # noqa: E402
from causal_rl.plotting.diagnosis_flowchart import make_diagnosis_flowchart  # noqa: E402
from causal_rl.plotting.divergence_panel import make_divergence_panel  # noqa: E402
from causal_rl.plotting.gap_curves import make_gap_curves  # noqa: E402
from causal_rl.plotting.headline_regression import make_headline_regression  # noqa: E402
from causal_rl.plotting.headline_scatter import make_headline_scatter  # noqa: E402
from causal_rl.plotting.identifiability_panel import make_identifiability_panel  # noqa: E402
from causal_rl.plotting.learning_curves import make_learning_curves  # noqa: E402
from causal_rl.plotting.metrics_curves import make_metrics_curves  # noqa: E402
from causal_rl.plotting.regime_map import make_regime_map  # noqa: E402
from causal_rl.plotting.sample_sweep import make_sample_sweep  # noqa: E402
from causal_rl.plotting.static_diagnostics import make_static_diagnostics  # noqa: E402
from causal_rl.plotting.summary_tables import make_summary_tables  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # If output not provided, place plots/tables under outputs/<results_folder_name>
    if args.output is None:
        args.output = Path("outputs") / args.results.name
    args.output.mkdir(parents=True, exist_ok=True)

    # Create separate subfolders for tabular and continuous envs to avoid aggregation
    base_output = args.output
    for prefix in ("tabular", "continuous"):
        out = base_output / prefix
        out.mkdir(parents=True, exist_ok=True)
        # ---- v11 diagnostic-framing headline figures (run first) ----
        make_regime_map(args.results, out, env_prefix=prefix)
        make_coverage_signature(args.results, out, env_prefix=prefix)
        make_metrics_curves(args.results, out, env_prefix=prefix)
        make_divergence_panel(args.results, out, env_prefix=prefix)
        make_causal_metrics_grid(args.results, out, env_prefix=prefix)
        make_static_diagnostics(args.results, out, env_prefix=prefix)
        make_diagnosis_flowchart(out)
        # ---- existing supporting figures ----
        make_learning_curves(args.results, out, env_prefix=prefix)
        make_gap_curves(args.results, out, env_prefix=prefix)
        make_headline_scatter(args.results, out, env_prefix=prefix)
        make_identifiability_panel(args.results, out, env_prefix=prefix)
        make_cell_grid(args.results, out, env_prefix=prefix)
        make_ablations(args.results, out, env_prefix=prefix)
        make_summary_tables(args.results, out, env_prefix=prefix)
        make_bias_sweep(args.results, out)
        make_sample_sweep(args.results, out)
        make_bound_width_panel(args.results, out)
    make_bandit_regret(args.results, base_output)
    make_headline_regression(args.results, base_output)


if __name__ == "__main__":
    main()
