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


# Algorithms whose trained policy is mathematically independent of the offline
# behaviour policy (on-policy learners collect their own data; bandit methods
# treat every episode as i.i.d.). Running them with multiple behaviours would
# produce identical results and waste compute.
_BEHAVIOUR_INDEPENDENT: frozenset[str] = frozenset({"ppo", "a2c", "ucb", "ucb_minus", "ucb_plus", "rct"})


def build_runs(config: MatrixConfig) -> list[tuple[int, str, str, str]]:
    runs: list[tuple[int, str, str, str]] = []
    # Track (cell, env, algo) tuples already added for behaviour-independent algos
    # so only the first behaviour in the YAML is included.
    seen_beh_independent: set[tuple[int, str, str]] = set()
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
        if algo in _BEHAVIOUR_INDEPENDENT:
            key = (cell, env, algo)
            if key in seen_beh_independent:
                continue
            seen_beh_independent.add(key)
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
    bias_strength: float,
    horizon_override: int | None,
    offline_transitions_override: int | None,
    forbidden_actions: list[int] | None,
    n_action_restriction_steps: int | None,
    matrix: MatrixConfig,
    run_dir: Path,
    device: str | None,
    quick: bool,
) -> list[str]:
    spec = ALGOS[algo]
    # Use algo-specific default when matrix.n_envs is absent/default (64).
    n_envs = matrix.n_envs if matrix.n_envs != 64 else spec.default_n_envs
    horizon = horizon_override if horizon_override is not None else matrix.env_horizons.get(env)
    if horizon is None:
        raise ValueError(f"Missing env_horizons entry for env: {env}")
    if matrix.total_frames_per_episode is not None:
        total_frames = int(matrix.total_frames_per_episode) * int(horizon)
    else:
        total_frames = matrix.total_frames
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
        str(2_000 if quick else total_frames),
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
        str(
            100
            if quick
            else (
                int(offline_transitions_override)
                if offline_transitions_override is not None
                else matrix.offline_transitions
            )
        ),
        "--offline-updates",
        str(50 if quick else matrix.offline_updates),
        "--alpha-conf",
        str(alpha_conf),
        "--bias-strength",
        str(bias_strength),
        "--n-envs",
        str(8 if quick else n_envs),
    ]
    if not matrix.eval_perturbations:
        cmd.append("--no-eval-perturbations")
    if matrix.eval_n_envs is not None:
        cmd.extend(["--eval-n-envs", str(matrix.eval_n_envs)])
    if getattr(matrix, "n_eval_episodes", None) is not None:
        cmd.extend(["--n-eval-episodes", str(matrix.n_eval_episodes)])
    if getattr(matrix, "perturbation_grid_size", None) is not None:
        cmd.extend(
            ["--perturbation-grid-size", str(matrix.perturbation_grid_size)]
        )
    if forbidden_actions:
        cmd.append("--forbidden-actions")
        cmd.extend(str(a) for a in forbidden_actions)
    if n_action_restriction_steps is not None:
        cmd.extend(
            ["--n-action-restriction-steps", str(n_action_restriction_steps)]
        )
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
            # Warn if more workers than GPUs — each GPU will serve multiple workers
            # simultaneously, which can exhaust VRAM.
            if args.n_workers > n_gpus:
                print(
                    f"Warning: {args.n_workers} workers share {n_gpus} GPU(s). "
                    f"Consider --n-workers <= {n_gpus} to avoid OOM."
                )
    elif args.device is not None:
        devices = [args.device]

    alpha_values: list[float] = (
        list(matrix.alpha_conf_sweep)
        if matrix.alpha_conf_sweep
        else [matrix.alpha_conf]
    )
    horizon_values: list[int | None] = (
        [int(h) for h in matrix.horizon_sweep]
        if matrix.horizon_sweep
        else [None]
    )
    bias_values: list[float] = (
        [float(b) for b in matrix.bias_strengths]
        if matrix.bias_strengths
        else [1.0]
    )
    offline_trans_values: list[int | None] = (
        [int(t) for t in matrix.offline_transitions_sweep]
        if matrix.offline_transitions_sweep
        else [None]
    )

    # v11: coverage_regimes is a sweep axis where each entry overrides
    # offline_transitions, forbidden_actions, and bias_strength all at
    # once.  When set, it *replaces* the bias_strengths /
    # offline_transitions_sweep axes for the regime dimension; the
    # remaining axes (cells, alpha, etc.) still expand normally.
    regimes = list(matrix.coverage_regimes) if matrix.coverage_regimes else None

    # Build all tasks.
    tasks: list[RunTask] = []
    for seed in seeds:
        for alpha_conf in alpha_values:
            for horizon_override in horizon_values:
                if regimes is not None:
                    # One run per regime — bias / offline_transitions /
                    # forbidden_actions are determined by the regime.
                    inner_iter = [
                        (
                            r.bias_strength,
                            int(r.offline_transitions),
                            list(r.forbidden_actions),
                            r.n_action_restriction_steps,
                            r.name,
                        )
                        for r in regimes
                    ]
                else:
                    inner_iter = [
                        (b, t, [], None, "")
                        for b in bias_values
                        for t in offline_trans_values
                    ]
                for (
                    bias_strength,
                    offline_t,
                    forbidden_actions,
                    n_action_restriction_steps,
                    regime_name,
                ) in inner_iter:
                    for cell, env, algo, beh in runs:
                        alpha_tag = f"_alpha{alpha_conf}" if len(alpha_values) > 1 else ""
                        horizon_tag = (
                            f"_h{horizon_override}" if horizon_override is not None else ""
                        )
                        bias_tag = (
                            f"_bias{bias_strength}"
                            if regimes is None and len(bias_values) > 1
                            else ""
                        )
                        offline_tag = (
                            f"_n{offline_t}"
                            if regimes is None and offline_t is not None
                            else ""
                        )
                        regime_tag = f"_regime-{regime_name}" if regime_name else ""
                        run_dir = base_results / (
                            f"cell{cell}_{env}_{algo}_{beh}_seed{seed}"
                            f"{alpha_tag}{horizon_tag}{bias_tag}{offline_tag}{regime_tag}"
                        )
                        device = devices[len(tasks) % len(devices)] if devices else args.device
                        cmd = _build_cmd(
                            cell=cell,
                            env=env,
                            algo=algo,
                            beh=beh,
                            seed=seed,
                            alpha_conf=alpha_conf,
                            bias_strength=bias_strength,
                            horizon_override=horizon_override,
                            offline_transitions_override=offline_t,
                            forbidden_actions=(
                                forbidden_actions
                                if forbidden_actions
                                else None
                            ),
                            n_action_restriction_steps=n_action_restriction_steps,
                            matrix=matrix,
                            run_dir=run_dir,
                            device=device,
                            quick=args.quick,
                        )
                        row: dict[str, object] = {
                            "seed": seed,
                            "alpha_conf": alpha_conf,
                            "bias_strength": bias_strength,
                            "horizon": (
                                horizon_override
                                if horizon_override is not None
                                else matrix.env_horizons.get(env, "")
                            ),
                            "offline_transitions": (
                                int(offline_t)
                                if offline_t is not None
                                else matrix.offline_transitions
                            ),
                            "coverage_regime": regime_name,
                            "forbidden_actions": ",".join(
                                str(a) for a in forbidden_actions
                            ),
                            "cell": cell,
                            "env": env,
                            "algorithm": algo,
                            "behaviour": beh,
                            "output_path": str(run_dir),
                        }
                        tasks.append((cmd, row))

    manifest = base_results / "manifest.csv"
    manifest_fields = [
        "seed", "alpha_conf", "bias_strength", "horizon", "offline_transitions",
        "coverage_regime", "forbidden_actions",
        "cell", "env", "algorithm", "behaviour", "output_path",
    ]
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
        _write_summary(base_results)
        sys.exit(1)
    else:
        print(f"\nAll {len(tasks)} runs completed successfully.")
        _write_summary(base_results)


def _write_summary(results_dir: Path) -> None:
    """Invoke make_summary to produce summary.json for the results directory."""
    try:
        from make_summary import make_summary  # type: ignore[import-not-found]

        out = make_summary(results_dir)
        print(f"Summary: {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: summary.json not written ({exc})")

        # Automatically generate plots using scripts/make_all_figures.py
        plot_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "make_all_figures.py"),
            "--results",
            str(base_results),
            "--output",
            str(base_outputs),
        ]
        print("Generating figures...")
        proc = subprocess.run(plot_cmd)
        if proc.returncode != 0:
            print(f"make_all_figures failed with exit code {proc.returncode}")
            sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
