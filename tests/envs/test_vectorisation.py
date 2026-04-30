from __future__ import annotations

import time

import torch

from causal_rl.envs.registry import make


def test_vectorised_step_cpu_runtime() -> None:
    env = make("tabular-sepsis-v0", cell=1, n_envs=256, device="cpu")
    obs, _ = env.reset(seed=0)
    action = torch.randint(0, 8, (256, 1))
    start = time.time()
    for _ in range(10):
        obs, _, done, trunc, _ = env.step(action)
        if bool(torch.any(done | trunc)):
            obs, _ = env.reset()
    elapsed = (time.time() - start) / 10.0
    assert elapsed < 0.05
