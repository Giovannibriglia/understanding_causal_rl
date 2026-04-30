from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv
from causal_rl.envs.cell_config import CELL_CONFIGS
from causal_rl.utils.device import get_device

OBS_STATES: Final[int] = 720
LATENT_STATES: Final[int] = 1440
N_ACTIONS: Final[int] = 8


class TabularSepsisEnv(CausalEnv):
    def __init__(
        self,
        cell: int,
        n_envs: int = 64,
        horizon: int = 20,
        gamma: float = 1.0,
        dense_reward: bool = False,
        alpha_conf: float = 0.0,
        transition_path: str | None = None,
        reward_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if cell not in CELL_CONFIGS:
            msg = f"Unknown cell: {cell}"
            raise ValueError(msg)
        self.cell = cell
        self.config = CELL_CONFIGS[cell]
        self.n_envs = n_envs
        self.horizon = horizon
        self.gamma = gamma
        self.dense_reward = dense_reward
        self.alpha_conf = alpha_conf
        self.device = get_device(device)

        self.is_discrete_action = True
        self.act_shape = (1,)
        self.obs_shape = (6 if self.config.expose_z else 5,)
        self._step_count = torch.zeros((n_envs,), dtype=torch.long, device=self.device)

        self.transition = self._load_or_build_transition(transition_path).to(self.device)
        self.reward_probs = self._load_or_build_reward(reward_path).to(self.device)
        self._latent_state = torch.zeros((n_envs,), dtype=torch.long, device=self.device)
        self._latent_u = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)

    def _load_or_build_transition(self, transition_path: str | None) -> Tensor:
        if transition_path is not None and Path(transition_path).exists():
            arr = np.load(transition_path)
            return torch.as_tensor(arr, dtype=torch.float32)
        transition = torch.zeros((LATENT_STATES, N_ACTIONS, LATENT_STATES), dtype=torch.float32)
        for s in range(LATENT_STATES):
            z = s // OBS_STATES
            base = s % OBS_STATES
            for a in range(N_ACTIONS):
                delta = (a + 1) * (1 if z == 0 else 2)
                ns0 = z * OBS_STATES + ((base + delta) % OBS_STATES)
                ns1 = z * OBS_STATES + ((base + delta + 7) % OBS_STATES)
                ns2 = (1 - z) * OBS_STATES + ((base + delta + 3) % OBS_STATES)
                transition[s, a, ns0] = 0.75
                transition[s, a, ns1] = 0.2
                transition[s, a, ns2] = 0.05
        return transition

    def _load_or_build_reward(self, reward_path: str | None) -> Tensor:
        if reward_path is not None and Path(reward_path).exists():
            arr = np.load(reward_path)
            return torch.as_tensor(arr, dtype=torch.float32)
        reward = torch.zeros((LATENT_STATES, N_ACTIONS, 2), dtype=torch.float32)
        for s in range(LATENT_STATES):
            z = float(s // OBS_STATES)
            obs = float(s % OBS_STATES) / float(OBS_STATES - 1)
            for a in range(N_ACTIONS):
                a_norm = float(a) / float(N_ACTIONS - 1)
                p_pos = 0.35 + 0.35 * (1.0 - abs(a_norm - (0.2 + 0.4 * z))) + 0.1 * (1.0 - obs)
                p_pos = float(max(0.01, min(0.99, p_pos)))
                reward[s, a, 0] = 1.0 - p_pos
                reward[s, a, 1] = p_pos
        return reward

    def _decode_obs(self, obs_index: Tensor) -> Tensor:
        idx = obs_index.to(torch.long)
        hr = idx % 3
        sbp = (idx // 3) % 5
        oxygen = (idx // 15) % 3
        glucose = (idx // 45) % 2
        tx = (idx // 90) % 8
        features = torch.stack(
            [
                hr.float() / 2.0,
                sbp.float() / 4.0,
                oxygen.float() / 2.0,
                glucose.float(),
                tx.float() / 7.0,
            ],
            dim=-1,
        )
        return features

    def _observe(self) -> Tensor:
        obs_idx = self._latent_state % OBS_STATES
        obs = self._decode_obs(obs_idx)
        if self.config.expose_z:
            z = (self._latent_state // OBS_STATES).float().unsqueeze(-1)
            obs = torch.cat([obs, z], dim=-1)
        return obs

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if seed is not None:
            torch.manual_seed(seed)
        self._step_count.zero_()
        self._latent_state = torch.randint(
            low=0, high=LATENT_STATES, size=(self.n_envs,), device=self.device
        )
        self._latent_u = torch.bernoulli(0.5 * torch.ones((self.n_envs,), device=self.device))
        return self._observe(), self._info()

    def _info(self) -> dict[str, Tensor]:
        info: dict[str, Tensor] = {
            "latent_Z": (self._latent_state // OBS_STATES).float().unsqueeze(-1),
        }
        if self.config.expose_u:
            info["latent_U"] = self._latent_u.unsqueeze(-1)
        return info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        a = action.view(-1).to(torch.long).clamp(min=0, max=N_ACTIONS - 1)
        state_probs = self.transition[self._latent_state, a]
        next_state = torch.multinomial(state_probs, 1).squeeze(-1)
        reward_bins = self.reward_probs[self._latent_state, a]
        reward_idx = torch.multinomial(reward_bins, 1).squeeze(-1)
        sparse_reward = torch.where(reward_idx == 1, 1.0, -1.0).to(torch.float32)

        if self.dense_reward:
            dense = -((next_state % OBS_STATES).float() / float(OBS_STATES))
            reward = 0.5 * sparse_reward + 0.5 * dense
        else:
            reward = sparse_reward

        self._latent_state = next_state
        self._step_count += 1
        terminated = self._step_count >= self.horizon
        truncated = torch.zeros_like(terminated, dtype=torch.bool)
        return self._observe(), reward, terminated, truncated, self._info()

    def do_reward(self, action: Tensor) -> Tensor:
        a = action.view(-1).to(torch.long).clamp(min=0, max=N_ACTIONS - 1)
        return self.reward_probs[self._latent_state, a]

    def observed_reward_prob(self, action: Tensor) -> Tensor:
        p_do = self.do_reward(action)
        if self.cell in {3}:
            return p_do
        a = action.view(-1).to(torch.long).clamp(min=0, max=N_ACTIONS - 1)
        shift = 0.02 * float(self.cell - 1)
        if self.cell in {7, 8}:
            shift += 0.05 * self.alpha_conf
        state_phase = ((self._latent_state % 31).float() / 31.0) * (2.0 * torch.pi)
        state_weight = 0.5 + 0.25 * torch.sin(state_phase)
        action_weight = 0.25 * (a.float() / float(N_ACTIONS - 1))
        modulation = torch.clamp(state_weight + action_weight, min=0.1, max=1.0)
        p_obs = p_do.clone()
        p_obs[:, 1] = torch.clamp(p_obs[:, 1] - shift * modulation, 1e-3, 1.0 - 1e-3)
        p_obs[:, 0] = 1.0 - p_obs[:, 1]
        return p_obs

    def do_transition(self, action: Tensor) -> Tensor:
        a = action.view(-1).to(torch.long).clamp(min=0, max=N_ACTIONS - 1)
        return self.transition[self._latent_state, a]

    def close(self) -> None:
        return None
