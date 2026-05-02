"""Run the full experiment matrix, optionally in parallel.

Usage examples::

    # Sequential (default)
    python scripts/run_full_matrix.py --config paper

    # 4 parallel workers on CPU
    python scripts/run_full_matrix.py --config paper --n-workers 4

    # 4 workers, stripe across 2 GPUs
    python scripts/run_full_matrix.py --config paper --n-workers 4 --n-gpus 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime
from multiprocessing import Pool
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Default 1 (sequential). "
        "Typical values: 4 on a 16-core CPU box, 1-per-GPU on multi-GPU.",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=0,
        help="Number of GPUs to stripe workers across. 0 = use --device as-is.",
    )
    return parser.parse_args()


def build_runs(config: MatrixConfig) -> list[tuple[int, str, str, str]]:
    runs: list[tuple[int, str, str, str]] = []
    for cell, env, algo, beh in itertools.product(
        config.cells, config.envs, config.algorithms, config.behaviours
    ):
        spec = ALGOS.get(algo)
        if spec is None or cell not in spec.supported_cells:
            continue
        if env == "continuous-ward-v0" and spec.action_type == "discrete":
            continue
        if env == "tabular-sepsis-v0" and spec.action_type == "continuous":
            continue
        runs.append((cell, env, algo, beh))
    return runs


def _resolve_config(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.is_file():
        return config_path
    candidate = Path(config_arg if str(config_arg).endswith(".yaml") else f"{config_arg}.yaml")
    if candidate.is_file():
        return candidate
    candidate2 = Path("configs") / (
        config_arg if str(config_arg).endswith(".yaml") else f"{config_arg}.yaml"
    )
    if candidate2.is_file():
        return candidate2
    raise FileNotFoundError(
        f"Config file not found. Tried: {config_path}, {candidate}, {candidate2}"
    )


def _build_cmd(
    *,
    cell: int,
    env: str,
    algo: str,
    beh: str,
    seed: int,
    alpha_conf: float,
    matrix: MatrixConfig,
    run_dir: Path,
    device: str | None,
    quick: bool,
) -> list[str]:
    spec = ALGOS[algo]
    # Use algo-specific default when matrix.n_envs is absent/default (64).
    n_envs = matrix.n_envs if matrix.n_envs != 64 else spec.default_n_envs
    horizon = matrix.env_horizons.get(env)
    if horizon is None:
        raise ValueError(f"Missing env_horizons entry for env: {env}")
    cmd = [
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
        str(2_000 if quick else matrix.total_frames),
        "--n-checkpoints-train",
        str(10 if quick else matrix.n_checkpoints_train),
        "--n-checkpoints-eval",
        str(5 if quick else matrix.n_checkpoints_eval),
        "--output",
        str(run_dir),
        "--rollout-horizon",
        str(horizon),
        "--horizon",
        str(horizon),
        "--offline-transitions",
        str(100 if quick else matrix.offline_transitions),
        "--offline-updates",
        str(50 if quick else matrix.offline_updates),
        "--alpha-conf",
        str(alpha_conf),
        "--n-envs",
        str(8 if quick else n_envs),
    ]
    if not matrix.eval_perturbations:
        cmd.append("--no-eval-perturbations")
    if matrix.eval_n_envs is not None:
        cmd.extend(["--eval-n-envs", str(matrix.eval_n_envs)])
    if device is not None:
        cmd.extend(["--device", device])
    return cmd


# Worker-level state (set in pool initializer)
_WORKER_DEVICE: str | None = None


def _init_worker(devices: list[str] | None, worker_id_queue: object) -> None:
    global _WORKER_DEVICE  # noqa: PLW0603
    if devices:
        pid = os.getpid()
        _WORKER_DEVICE = devices[pid % len(devices)]


RunTask = tuple[list[str], dict[str, object]]


def _run_one(task: RunTask) -> tuple[dict[str, object], int, str]:
    cmd, row = task
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    if _WORKER_DEVICE is not None:
        cmd = [c for c in cmd if c != "--device"] + ["--device", _WORKER_DEVICE]
    result = subprocess.run(cmd, capture_output=True, env=env)
    stderr_tail = result.stderr.decode(errors="replace")[-500:]
    return row, result.returncode, stderr_tail


def main() -> None:
    args = parse_args()
    config_file = _resolve_config(args.config)
    data = load_yaml(config_file)
    matrix = MatrixConfig.model_validate(data)
    runs = build_runs(matrix)
    if args.quick:
        runs = runs[:12]

    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    elif getattr(matrix, "seeds", None):
        seeds = list(matrix.seeds)
    else:
        raise ValueError("No seeds: provide --seeds or include 'seeds' in config YAML.")

    config_name = config_file.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results = Path("results") / f"{config_name}_{ts}"
    base_outputs = Path("outputs") / f"{config_name}_{ts}"
    base_results.mkdir(parents=True, exist_ok=True)
    base_outputs.mkdir(parents=True, exist_ok=True)

    print(f"Results: {base_results}")
    print(f"Outputs: {base_outputs}")
    print(f"Workers: {args.n_workers}  GPUs: {args.n_gpus}")

    # Decide device list for striping.
    devices: list[str] | None = None
    if args.n_gpus > 0:
        import torch as _torch
        available = _torch.cuda.device_count()
        n_gpus = min(args.n_gpus, available) if available > 0 else 0
        if n_gpus < args.n_gpus:
            print(
                f"Warning: --n-gpus {args.n_gpus} requested but only {available} "
                f"GPU(s) available; clamping to {n_gpus}."
            )
        if n_gpus > 0:
            devices = [f"cuda:{i}" for i in range(n_gpus)]
    elif args.device is not None:
        devices = [args.device]

    alpha_values: list[float] = (
        list(matrix.alpha_conf_sweep)
        if matrix.alpha_conf_sweep
        else [matrix.alpha_conf]
    )

    # Build all tasks.
    tasks: list[RunTask] = []
    for seed in seeds:
        for alpha_conf in alpha_values:
            for cell, env, algo, beh in runs:
                alpha_tag = f"_alpha{alpha_conf}" if len(alpha_values) > 1 else ""
                run_dir = base_results / f"cell{cell}_{env}_{algo}_{beh}_seed{seed}{alpha_tag}"
                device = devices[len(tasks) % len(devices)] if devices else args.device
                cmd = _build_cmd(
                    cell=cell,
                    env=env,
                    algo=algo,
                    beh=beh,
                    seed=seed,
                    alpha_conf=alpha_conf,
                    matrix=matrix,
                    run_dir=run_dir,
                    device=device,
                    quick=args.quick,
                )
                row: dict[str, object] = {
                    "seed": seed,
                    "alpha_conf": alpha_conf,
                    "cell": cell,
                    "env": env,
                    "algorithm": algo,
                    "behaviour": beh,
                    "output_path": str(run_dir),
                }
                tasks.append((cmd, row))

    manifest = base_results / "manifest.csv"
    manifest_fields = ["seed", "alpha_conf", "cell", "env", "algorithm", "behaviour", "output_path"]
    failed: list[tuple[dict[str, object], int, str]] = []

    with manifest.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=manifest_fields)
        writer.writeheader()

        if args.n_workers <= 1:
            for task in tqdm(tasks, desc="Experiments", unit="run"):
                row_out, rc, stderr = _run_one(task)
                writer.writerow(row_out)
                mf.flush()
                if rc != 0:
                    failed.append((row_out, rc, stderr))
        else:
            with Pool(processes=args.n_workers) as pool:
                for row_out, rc, stderr in tqdm(
                    pool.imap_unordered(_run_one, tasks),
                    total=len(tasks),
                    desc="Experiments",
                    unit="run",
                ):
                    writer.writerow(row_out)
                    mf.flush()
                    if rc != 0:
                        failed.append((row_out, rc, stderr))

    if failed:
        log_path = base_results / "failed_runs.log"
        with log_path.open("w", encoding="utf-8") as lf:
            for row_out, rc, stderr in failed:
                lf.write(f"rc={rc} {row_out}\n{stderr}\n---\n")
        print(f"\n{len(failed)}/{len(tasks)} runs FAILED. Details: {log_path}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tasks)} runs completed successfully.")


if __name__ == "__main__":
    main()
