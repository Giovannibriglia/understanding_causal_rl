"""Emit a commands.txt file — one run_single.py command per line.

Suitable for GNU parallel or SLURM array submission.  Does NOT execute
anything itself.

Usage::

    python scripts/run_full_matrix_slurm.py --config paper \\
        --seeds 0 1 2 --output-dir results/paper_run > commands.txt

    # Run locally with GNU parallel:
    parallel -j8 < commands.txt

    # Submit to SLURM:
    sbatch --array=1-$(wc -l < commands.txt) slurm_array.sh
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.algos.registry import ALGOS  # noqa: E402
from causal_rl.config.schemas import MatrixConfig, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="paper")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_config(config_arg: str) -> Path:
    for candidate in [
        Path(config_arg),
        Path(f"{config_arg}.yaml"),
        Path("configs") / f"{config_arg}.yaml",
        Path("configs") / config_arg,
    ]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def main() -> None:
    args = parse_args()
    config_file = _resolve_config(args.config)
    data = load_yaml(config_file)
    matrix = MatrixConfig.model_validate(data)

    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    elif getattr(matrix, "seeds", None):
        seeds = list(matrix.seeds)
    else:
        raise ValueError("Specify --seeds or add 'seeds' to the config YAML.")

    if args.output_dir is not None:
        base_results = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results = Path("results") / f"{config_file.stem}_{ts}"

    alpha_values: list[float] = (
        list(matrix.alpha_conf_sweep)
        if matrix.alpha_conf_sweep
        else [matrix.alpha_conf]
    )

    lines: list[str] = []
    for seed in seeds:
        for alpha_conf in alpha_values:
            for cell, env, algo, beh in itertools.product(
                matrix.cells, matrix.envs, matrix.algorithms, matrix.behaviours
            ):
                spec = ALGOS.get(algo)
                if spec is None or cell not in spec.supported_cells:
                    continue
                if env == "continuous-ward-v0" and spec.action_type == "discrete":
                    continue
                if env == "tabular-sepsis-v0" and spec.action_type == "continuous":
                    continue

                horizon = matrix.env_horizons.get(env)
                if horizon is None:
                    continue

                n_envs = matrix.n_envs if matrix.n_envs != 64 else spec.default_n_envs
                alpha_tag = f"_alpha{alpha_conf}" if len(alpha_values) > 1 else ""
                run_dir = base_results / f"cell{cell}_{env}_{algo}_{beh}_seed{seed}{alpha_tag}"

                parts = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_single.py"),
                    "--cell",
                    str(cell),
                    "--env",
                    env,
                    "--algorithm",
                    algo,
                    "--behaviour",
                    beh,
                    "--seed",
                    str(seed),
                    "--total-frames",
                    str(matrix.total_frames),
                    "--n-checkpoints-train",
                    str(matrix.n_checkpoints_train),
                    "--n-checkpoints-eval",
                    str(matrix.n_checkpoints_eval),
                    "--horizon",
                    str(horizon),
                    "--rollout-horizon",
                    str(horizon),
                    "--offline-transitions",
                    str(matrix.offline_transitions),
                    "--offline-updates",
                    str(matrix.offline_updates),
                    "--alpha-conf",
                    str(alpha_conf),
                    "--n-envs",
                    str(n_envs),
                    "--output",
                    str(run_dir),
                ]
                if not matrix.eval_perturbations:
                    parts.append("--no-eval-perturbations")
                if matrix.eval_n_envs is not None:
                    parts += ["--eval-n-envs", str(matrix.eval_n_envs)]
                if args.device is not None:
                    parts += ["--device", args.device]

                lines.append(" ".join(parts))

    sys.stdout.write("\n".join(lines) + "\n")
    print(f"# {len(lines)} commands written.", file=sys.stderr)


if __name__ == "__main__":
    main()
