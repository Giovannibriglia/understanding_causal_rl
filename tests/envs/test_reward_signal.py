"""Acceptance tests for the sharpened reward signal in TabularSepsisEnv."""

from __future__ import annotations

import torch

from causal_rl.envs.tabular_sepsis import TabularSepsisEnv


def _run_policy(env: TabularSepsisEnv, action_fn: object, seed: int = 0) -> float:
    """Run one episode with a given action function, return mean return."""
    import math

    _ = math  # noqa: F841
    from collections.abc import Callable

    fn: Callable[[torch.Tensor], torch.Tensor] = action_fn  # type: ignore[assignment]
    obs, _ = env.reset(seed=seed)
    returns = torch.zeros(env.n_envs, device=obs.device)
    for _ in range(env.horizon):
        action = fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        if bool(torch.all(terminated | truncated)):
            break
    return float(returns.mean().item())


def _uniform_policy(obs: torch.Tensor) -> torch.Tensor:
    n = obs.shape[0]
    return torch.randint(0, 8, (n, 1), device=obs.device)


def test_uniform_policy_return_near_zero() -> None:
    """Uniform random policy should return ≈0 (no free lunch)."""
    torch.manual_seed(42)
    env = TabularSepsisEnv(cell=1, n_envs=512, horizon=20, alpha_conf=0.0)
    returns: list[float] = []
    for ep in range(20):
        obs, _ = env.reset(seed=ep * 100)
        ep_rets = torch.zeros(env.n_envs)
        for _ in range(env.horizon):
            act = torch.randint(0, 8, (env.n_envs, 1))
            obs, rew, term, trunc, _ = env.step(act)
            ep_rets += rew
            if bool(torch.all(term | trunc)):
                break
        returns.append(float(ep_rets.mean().item()))
    mean_ret = sum(returns) / len(returns)
    assert abs(mean_ret) < 1.5, f"Uniform random return should be ≈0, got {mean_ret:.2f}"


def test_oracle_policy_return_high() -> None:
    """Oracle policy (DP optimal) should return ≥ 15 on a 20-step horizon."""
    torch.manual_seed(0)
    env = TabularSepsisEnv(cell=1, n_envs=512, horizon=20, alpha_conf=0.0)

    # Build tabular oracle via value iteration.
    transition = env.transition  # (S, A, S)
    reward_probs = env.reward_probs  # (S, A, 2)
    n_states = transition.shape[0]
    horizon = env.horizon
    expected_r = reward_probs[:, :, 1] - reward_probs[:, :, 0]  # (S, A)
    values = torch.zeros((horizon + 1, n_states))
    policy = torch.zeros((horizon + 1, n_states), dtype=torch.long)
    for h in range(1, horizon + 1):
        q = expected_r + env.gamma * torch.einsum("sak,k->sa", transition, values[h - 1])
        values[h], policy[h] = q.max(dim=1)

    obs, _ = env.reset(seed=0)
    returns = torch.zeros(env.n_envs)
    for step in range(horizon):
        latent = env._latent_state.clamp(0, n_states - 1)
        steps_left = max(1, horizon - step)
        action = policy[steps_left, latent].unsqueeze(-1)
        obs, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        if bool(torch.all(terminated | truncated)):
            break
    mean_ret = float(returns.mean().item())
    assert mean_ret >= 15.0, f"Oracle return should be ≥ 15, got {mean_ret:.2f}"


def test_reward_signal_separation() -> None:
    """Oracle − random separation should be ≥ 12 (signal-to-noise requirement)."""
    torch.manual_seed(0)
    n_envs = 512

    # Random policy return.
    env_rand = TabularSepsisEnv(cell=1, n_envs=n_envs, horizon=20, alpha_conf=0.0)
    rand_returns: list[float] = []
    for ep in range(10):
        obs, _ = env_rand.reset(seed=ep * 50)
        ep_rets = torch.zeros(n_envs)
        for _ in range(env_rand.horizon):
            act = torch.randint(0, 8, (n_envs, 1))
            obs, rew, term, trunc, _ = env_rand.step(act)
            ep_rets += rew
            if bool(torch.all(term | trunc)):
                break
        rand_returns.append(float(ep_rets.mean().item()))
    rand_mean = sum(rand_returns) / len(rand_returns)

    # Oracle return.

    env_ora = TabularSepsisEnv(cell=1, n_envs=n_envs, horizon=20, alpha_conf=0.0)
    transition = env_ora.transition
    reward_probs = env_ora.reward_probs
    n_states = transition.shape[0]
    horizon = env_ora.horizon
    expected_r = reward_probs[:, :, 1] - reward_probs[:, :, 0]
    values = torch.zeros((horizon + 1, n_states))
    policy = torch.zeros((horizon + 1, n_states), dtype=torch.long)
    for h in range(1, horizon + 1):
        q = expected_r + env_ora.gamma * torch.einsum("sak,k->sa", transition, values[h - 1])
        values[h], policy[h] = q.max(dim=1)

    obs, _ = env_ora.reset(seed=0)
    oracle_returns_t = torch.zeros(n_envs)
    for step in range(horizon):
        latent = env_ora._latent_state.clamp(0, n_states - 1)
        steps_left = max(1, horizon - step)
        action = policy[steps_left, latent].unsqueeze(-1)
        obs, reward, terminated, truncated, _ = env_ora.step(action)
        oracle_returns_t += reward
        if bool(torch.all(terminated | truncated)):
            break
    oracle_mean = float(oracle_returns_t.mean().item())

    separation = oracle_mean - rand_mean
    assert separation >= 12.0, (
        f"Oracle−random separation should be ≥ 12, got {separation:.2f} "
        f"(oracle={oracle_mean:.2f}, random={rand_mean:.2f})"
    )
