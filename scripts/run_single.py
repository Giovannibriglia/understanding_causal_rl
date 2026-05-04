from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--behaviour", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-frames", type=int, default=50_000)
    parser.add_argument("--n-checkpoints-train", type=int, default=50)
    parser.add_argument("--n-checkpoints-eval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--eval-n-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--offline-transitions", type=int, default=50_000)
    parser.add_argument("--offline-updates", type=int, default=2_000)
    parser.add_argument("--alpha-conf", type=float, default=0.0)
    parser.add_argument(
        "--bias-strength",
        type=float,
        default=1.0,
        help="Bias strength multiplier for the behaviour policy (where applicable).",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument(
        "--perturbation-grid-size",
        type=int,
        default=None,
        help=(
            "Trim TABULAR_DIAGONAL to the first N specs (smoke knob)."
            " Default None ⇒ full 5-spec grid."
        ),
    )
    parser.add_argument(
        "--eval-perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run perturbed evaluation across TABULAR_DIAGONAL schedule.",
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default="auto",
        help="Oracle type for evaluation: auto|dp|cem|grid|algo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunnerConfig(
        cell=args.cell,
        env_name=args.env,
        algorithm=args.algorithm,
        behaviour=args.behaviour,
        seed=args.seed,
        total_frames=args.total_frames,
        n_checkpoints_train=args.n_checkpoints_train,
        n_checkpoints_eval=args.n_checkpoints_eval,
        horizon=args.horizon,
        rollout_horizon=args.rollout_horizon,
        output_dir=args.output,
        device=args.device,
        n_envs=args.n_envs,
        eval_n_envs=args.eval_n_envs,
        batch_size=args.batch_size,
        offline_transitions=args.offline_transitions,
        offline_updates=args.offline_updates,
        alpha_conf=args.alpha_conf,
        bias_strength=args.bias_strength,
        oracle=args.oracle,
        n_eval_episodes=args.n_eval_episodes,
        perturbation_grid_size=args.perturbation_grid_size,
        eval_perturbations=args.eval_perturbations,
    )
    runner = BenchmarkRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
