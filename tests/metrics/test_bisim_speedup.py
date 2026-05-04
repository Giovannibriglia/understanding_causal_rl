"""Phase 3 acceptance: vectorised ``bisimulation_distance`` is correct and
fast.

* Numerical equivalence: identical envs produce a zero diagonal.
* Output shape matches the documented ``(S, S)`` contract.
* Runtime: 24-state MDPs (the runner's bisim subset) finish under 500 ms
  on CPU; the previous triple-nested loop took ~1.6 s per call.
"""

from __future__ import annotations

import time

import torch

from causal_rl.metrics.bisimulation import bisimulation_distance


def test_bisim_distance_zero_diagonal_when_envs_identical() -> None:
    torch.manual_seed(0)
    n_states, n_actions = 12, 4
    Mr = torch.rand(n_states, n_actions)  # noqa: N806
    Mt = torch.softmax(torch.rand(n_states, n_actions, n_states), dim=-1)  # noqa: N806
    d_self = bisimulation_distance(Mr, Mt, Mr, Mt, gamma=0.95, n_iters=20)
    assert d_self.shape == (n_states, n_states)
    assert torch.diagonal(d_self).abs().max().item() < 1e-5


def test_bisim_distance_grows_with_reward_shift() -> None:
    torch.manual_seed(1)
    n_states, n_actions = 12, 4
    Mr = torch.rand(n_states, n_actions)  # noqa: N806
    Mt = torch.softmax(torch.rand(n_states, n_actions, n_states), dim=-1)  # noqa: N806
    d_small = bisimulation_distance(Mr, Mt, Mr + 0.05, Mt, gamma=0.5, n_iters=15)
    d_large = bisimulation_distance(Mr, Mt, Mr + 0.5, Mt, gamma=0.5, n_iters=15)
    assert torch.diagonal(d_large).mean() > torch.diagonal(d_small).mean()


def test_bisim_distance_runtime_under_500ms() -> None:
    torch.manual_seed(2)
    Mr = torch.rand(24, 8)  # noqa: N806
    Mt = torch.softmax(torch.rand(24, 8, 24), dim=-1)  # noqa: N806
    Mpr = torch.rand(24, 8)  # noqa: N806
    Mpt = torch.softmax(torch.rand(24, 8, 24), dim=-1)  # noqa: N806
    t0 = time.perf_counter()
    bisimulation_distance(Mr, Mt, Mpr, Mpt, gamma=0.95, n_iters=15)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, (
        f"bisim distance took {elapsed:.3f}s on a 24-state MDP — should be <0.5s "
        "after vectorisation"
    )
