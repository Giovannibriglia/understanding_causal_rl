from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv


class BanditView(CausalEnv):
    """Wraps a CausalEnv as a K-arm bandit (single-step episodes)."""

    def __init__(self, env: CausalEnv, n_context_states: int = 8) -> None:
        self._env = env
        self.cell = env.cell
        self.obs_shape = env.obs_shape
        self.act_shape = env.act_shape
        self.is_discrete_action = env.is_discrete_action
        self.horizon = 1
        self.gamma = env.gamma
        self.n_envs = env.n_envs

        # Build context pool: reset env once to get the obs shape right
        obs, info = env.reset(seed=0)
        self._context_pool: Tensor = obs  # shape (n_envs, *obs_shape)
        self._current_obs: Tensor = obs
        self._current_info: dict[str, Tensor] = info

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        obs, info = self._env.reset(seed=seed)
        n = obs.shape[0]
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randint(0, n, (self.n_envs,), device=obs.device)
        self._current_obs = obs[idx]
        self._current_info = {k: v[idx] if v.shape[0] == n else v for k, v in info.items()}
        return self._current_obs, self._current_info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        # Restore the env's internal state to the current obs is tricky —
        # just use do_reward (interventional) if available, else fall back to step.
        try:
            reward_probs = self._env.do_reward(action)
            # do_reward on TabularSepsis returns (n_envs, 2) prob tensor
            if reward_probs.dim() == 2 and reward_probs.shape[-1] == 2:
                reward_idx = torch.multinomial(reward_probs, 1).squeeze(-1)
                reward = torch.where(
                    reward_idx == 1,
                    torch.ones(self.n_envs, device=reward_probs.device),
                    -torch.ones(self.n_envs, device=reward_probs.device),
                )
            else:
                reward = reward_probs.view(self.n_envs)
        except NotImplementedError:
            _, reward, _, _, _ = self._env.step(action)

        terminated = torch.ones(self.n_envs, dtype=torch.bool, device=self._current_obs.device)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=self._current_obs.device)
        return self._current_obs, reward, terminated, truncated, self._current_info

    def do_reward(self, action: Tensor) -> Tensor:
        return self._env.do_reward(action)

    def do_transition(self, action: Tensor) -> Tensor:
        return self._env.do_transition(action)

    def close(self) -> None:
        self._env.close()
