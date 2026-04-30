from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.envs.registry import make  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def benchmark_env(env_name: str, b: int, steps: int) -> float:
    env = make(env_name, cell=1, n_envs=b, device="cpu")
    obs, _ = env.reset(seed=0)
    start = time.time()
    for _ in range(steps):
        if env.is_discrete_action:
            action = torch.randint(0, 8, (b, 1), device=obs.device)
        else:
            action = torch.rand((b, 3), device=obs.device)
        obs, _, done, trunc, _ = env.step(action)
        if bool(torch.any(done | trunc)):
            obs, _ = env.reset()
    elapsed = time.time() - start
    return steps * b / max(elapsed, 1e-9)


def main() -> None:
    args = parse_args()
    steps = 200 if args.quick else 1000
    for env in ("tabular-sepsis-v0", "continuous-ward-v0"):
        for b in (1, 16, 256):
            sps = benchmark_env(env, b, steps)
            print(f"{env} B={b}: {sps:.1f} steps/s")


if __name__ == "__main__":
    main()
