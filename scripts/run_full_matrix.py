from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.algos.registry import ALGOS  # noqa: E402
from causal_rl.config.schemas import MatrixConfig, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default="full_matrix",
        help="Config name (no .yaml) or path. If name, looks for configs/{name}.yaml",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def build_runs(config: MatrixConfig) -> list[tuple[int, str, str, str]]:
    runs: list[tuple[int, str, str, str]] = []
    for cell, env, algo, beh in itertools.product(
        config.cells, config.envs, config.algorithms, config.behaviours
    ):
        spec = ALGOS.get(algo)
        if spec is None or cell not in spec.supported_cells:
            continue
        """if cell in {1, 2} and beh != "uniform":
            continue"""
        if env == "continuous-ward-v0" and spec.action_type == "discrete":
            continue
        if env == "tabular-sepsis-v0" and spec.action_type == "continuous":
            continue
        runs.append((cell, env, algo, beh))
    return runs


def main() -> None:
    args = parse_args()

    # Resolve config file: accept short config name (e.g. 'behaviour_ablation')
    # or a full path. Preference order: exact path -> ./<name>.yaml -> configs/<name>.yaml
    config_arg = args.config
    config_path = Path(config_arg)
    if config_path.is_file():
        config_file = config_path
    else:
        candidate = Path(config_arg if str(config_arg).endswith(".yaml") else f"{config_arg}.yaml")
        if candidate.is_file():
            config_file = candidate
        else:
            candidate2 = Path("configs") / (
                config_arg if str(config_arg).endswith(".yaml") else f"{config_arg}.yaml"
            )
            if candidate2.is_file():
                config_file = candidate2
            else:
                raise FileNotFoundError(
                    f"Config file not found. Tried: {config_path}, {candidate}, {candidate2}"
                )

    data = load_yaml(config_file)
    matrix = MatrixConfig.model_validate(data)
    runs = build_runs(matrix)
    if args.quick:
        runs = runs[:12]

    # determine seeds: CLI overrides config file
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    elif getattr(matrix, "seeds", None):
        seeds = matrix.seeds
    else:
        raise ValueError("No seeds specified: provide --seeds or include 'seeds' in config YAML.")

    # Compute standardized output paths
    config_name = config_file.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results = Path("results") / f"{config_name}_{ts}"
    base_outputs = Path("outputs") / f"{config_name}_{ts}"

    base_results.mkdir(parents=True, exist_ok=True)
    base_outputs.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {base_results}")
    print(f"Plots and tables will be saved to: {base_outputs}")

    manifest = base_results / "manifest.csv"
    alpha_confs = matrix.alpha_conf_sweep if matrix.alpha_conf_sweep else [matrix.alpha_conf]
    total_runs = len(seeds) * len(runs) * len(alpha_confs)
    commands_log = base_results / "commands.log"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "cell", "env", "algorithm", "behaviour", "output_path"]
        )
        writer.writeheader()
        with tqdm(total=total_runs, desc="Experiments", unit="run") as pbar:
            for seed in seeds:
                for alpha_conf in alpha_confs:
                    for cell, env, algo, beh in runs:
                        horizon = matrix.env_horizons.get(env)
                        if horizon is None:
                            raise ValueError(f"Missing env_horizons entry for env: {env}")
                        pbar.set_postfix(
                            seed=seed,
                            cell=cell,
                            env=env,
                            algo=algo,
                            beh=beh,
                            refresh=False,
                        )
                        run_dir = base_results / (
                            f"cell{cell}_{env}_{algo}_{beh}_seed{seed}_alpha{alpha_conf}"
                        )
                        cmd = [
                            sys.executable,
                            "scripts/run_single.py",
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
                            str(2_000 if args.quick else matrix.total_frames),
                            "--n-checkpoints-train",
                            str(10 if args.quick else matrix.n_checkpoints_train),
                            "--n-checkpoints-eval",
                            str(5 if args.quick else matrix.n_checkpoints_eval),
                            "--output",
                            str(run_dir),
                        ]
                        cmd.extend(["--rollout-horizon", str(horizon), "--horizon", str(horizon)])
                        cmd.extend(
                            [
                                "--offline-transitions",
                                str(100 if args.quick else matrix.offline_transitions),
                                "--offline-updates",
                                str(50 if args.quick else matrix.offline_updates),
                                "--alpha-conf",
                                str(alpha_conf),
                                "--n-envs",
                                str(8 if args.quick else matrix.n_envs),
                            ]
                        )
                        if matrix.eval_n_envs is not None:
                            cmd.extend(["--eval-n-envs", str(matrix.eval_n_envs)])
                        if args.device is not None:
                            cmd.extend(["--device", args.device])
                        with commands_log.open("a", encoding="utf-8") as clog:
                            clog.write(" ".join(cmd) + "\n")
                        subprocess.run(cmd, check=True)
                        writer.writerow(
                            {
                                "seed": seed,
                                "cell": cell,
                                "env": env,
                                "algorithm": algo,
                                "behaviour": beh,
                                "output_path": str(run_dir),
                            }
                        )
                        pbar.update(1)


if __name__ == "__main__":
    main()
