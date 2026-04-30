from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv
from causal_rl.envs.cell_config import CELL_CONFIGS
from causal_rl.utils.device import get_device


class ContinuousWardEnv(CausalEnv):
    def __init__(
        self,
        cell: int,
        n_envs: int = 64,
        horizon: int = 50,
        gamma: float = 0.99,
        dt: float = 0.1,
        sigma: float = 0.05,
        lambda_action: float = 0.1,
        mc_k: int = 64,
        alpha_conf: float = 0.0,
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
        self.dt = dt
        self.sigma = sigma
        self.lambda_action = lambda_action
        self.mc_k = mc_k
        self.alpha_conf = alpha_conf
        self.device = get_device(device)

        self.state_dim = 4
        self.z_dim = 2
        self.u_dim = 1
        self.is_discrete_action = False
        self.act_shape = (3,)
        self.obs_shape = (self.state_dim + (self.z_dim if self.config.expose_z else 0),)
        self._step_count = torch.zeros((n_envs,), dtype=torch.long, device=self.device)

        self.state = torch.zeros((n_envs, self.state_dim), dtype=torch.float32, device=self.device)
        self.latent_z = torch.zeros((n_envs, self.z_dim), dtype=torch.float32, device=self.device)
        self.latent_u = torch.zeros((n_envs, self.u_dim), dtype=torch.float32, device=self.device)
        self.target_state = torch.tensor(
            [0.2, 0.3, 0.4, 0.2], device=self.device, dtype=torch.float32
        )

    def _drift(self, state: Tensor, action: Tensor, z: Tensor, u: Tensor) -> Tensor:
        z_mix = 0.2 * z[:, :1] + 0.1 * z[:, 1:]
        interaction = action * (0.5 + z_mix)
        u_term = 0.2 * u
        base = torch.stack(
            [
                -0.3 * (state[:, 0] - 0.2) - 0.4 * interaction[:, 0] - u_term.squeeze(-1),
                -0.25 * (state[:, 1] - 0.3) - 0.3 * action[:, 1] + 0.2 * z[:, 0],
                -0.2 * (state[:, 2] - 0.4) - 0.35 * action[:, 2] + 0.15 * z[:, 1],
                -0.1 * (state[:, 3] - 0.2) - 0.2 * action.mean(dim=-1) + 0.1 * z_mix.squeeze(-1),
            ],
            dim=-1,
        )
        return base

    def _observe(self) -> Tensor:
        if self.config.expose_z:
            return torch.cat([self.state, self.latent_z], dim=-1)
        return self.state

    def _info(self) -> dict[str, Tensor]:
        info: dict[str, Tensor] = {"latent_Z": self.latent_z}
        if self.config.expose_u:
            info["latent_U"] = self.latent_u
        return info

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if seed is not None:
            torch.manual_seed(seed)
        self._step_count.zero_()
        self.state = torch.rand((self.n_envs, self.state_dim), device=self.device)
        self.latent_z = torch.rand((self.n_envs, self.z_dim), device=self.device)
        self.latent_u = torch.randn((self.n_envs, self.u_dim), device=self.device) * 0.25
        return self._observe(), self._info()

    def _transition(self, action: Tensor) -> Tensor:
        drift = self._drift(self.state, action, self.latent_z, self.latent_u)
        noise = self.sigma * torch.randn_like(self.state)
        next_state = self.state + self.dt * drift + noise
        return next_state.clamp(0.0, 1.0)

    def _reward(self, state: Tensor, action: Tensor) -> Tensor:
        sq = torch.sum((state - self.target_state.unsqueeze(0)) ** 2, dim=-1)
        penalty = self.lambda_action * action.abs().sum(dim=-1)
        return -(sq + penalty)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        a = action.to(torch.float32).view(self.n_envs, 3).clamp(0.0, 1.0)
        next_state = self._transition(a)
        reward = self._reward(next_state, a)
        self.state = next_state
        self._step_count += 1
        terminated = self._step_count >= self.horizon
        truncated = torch.zeros_like(terminated, dtype=torch.bool)
        return self._observe(), reward, terminated, truncated, self._info()

    def do_reward(self, action: Tensor) -> Tensor:
        a = action.to(torch.float32).view(self.n_envs, 3).clamp(0.0, 1.0)
        reps_state = self.state.repeat_interleave(self.mc_k, dim=0)
        reps_z = self.latent_z.repeat_interleave(self.mc_k, dim=0)
        reps_u = self.latent_u.repeat_interleave(self.mc_k, dim=0)
        reps_action = a.repeat_interleave(self.mc_k, dim=0)
        drift = self._drift(reps_state, reps_action, reps_z, reps_u)
        noise = self.sigma * torch.randn_like(reps_state)
        next_state = (reps_state + self.dt * drift + noise).clamp(0.0, 1.0)
        rewards = self._reward(next_state, reps_action)
        return rewards.view(self.n_envs, self.mc_k)

    def sample_interventional(self, action: Tensor, n: int) -> Tensor:
        """Draw n reward samples from P(R | do(A=a), current state).

        Marginalises over Z when Z is not exposed; conditions on current Z
        when Z is exposed.  Returns shape (n_envs, n).
        """
        a = action.to(torch.float32).view(self.n_envs, 3).clamp(0.0, 1.0)
        reps_action = a.repeat_interleave(n, dim=0)

        if self.config.expose_z:
            # Condition on current (Z, U).
            reps_state = self.state.repeat_interleave(n, dim=0)
            reps_z = self.latent_z.repeat_interleave(n, dim=0)
            reps_u = self.latent_u.repeat_interleave(n, dim=0)
        else:
            # Marginalise over Z by resampling it from its prior U[0,1]^z_dim.
            reps_state = self.state.repeat_interleave(n, dim=0)
            reps_z = torch.rand(
                (self.n_envs * n, self.z_dim), device=self.device, dtype=torch.float32
            )
            reps_u = (
                torch.randn((self.n_envs * n, self.u_dim), device=self.device, dtype=torch.float32)
                * 0.25
            )

        drift = self._drift(reps_state, reps_action, reps_z, reps_u)
        noise = self.sigma * torch.randn_like(reps_state)
        next_state = (reps_state + self.dt * drift + noise).clamp(0.0, 1.0)
        rewards = self._reward(next_state, reps_action)
        return rewards.view(self.n_envs, n)

    def sample_observational(self, action: Tensor, n: int) -> Tensor:
        """Draw n reward samples from P(R | A=a, current observable state).

        When alpha_conf=0, this equals sample_interventional.  When alpha_conf>0
        the hidden confounder U is resampled with selection bias: samples with
        higher reward are kept with probability proportional to
        exp(alpha_conf * reward), simulating a reward-seeking behaviour policy.

        Uses rejection sampling with a fixed number of candidates to keep runtime
        bounded; returns shape (n_envs, n).
        """
        if self.alpha_conf <= 0.0:
            return self.sample_interventional(action, n)

        a = action.to(torch.float32).view(self.n_envs, 3).clamp(0.0, 1.0)
        # Draw more candidates than needed for rejection sampling.
        n_cand = max(n * 8, n + 32)
        reps_action = a.repeat_interleave(n_cand, dim=0)
        reps_state = self.state.repeat_interleave(n_cand, dim=0)
        # Resample Z and U from prior.
        reps_z = torch.rand(
            (self.n_envs * n_cand, self.z_dim), device=self.device, dtype=torch.float32
        )
        reps_u = (
            torch.randn((self.n_envs * n_cand, self.u_dim), device=self.device, dtype=torch.float32)
            * 0.25
        )
        drift = self._drift(reps_state, reps_action, reps_z, reps_u)
        noise = self.sigma * torch.randn_like(reps_state)
        next_state = (reps_state + self.dt * drift + noise).clamp(0.0, 1.0)
        rewards_flat = self._reward(next_state, reps_action)  # (n_envs*n_cand,)
        rewards_mat = rewards_flat.view(self.n_envs, n_cand)  # (n_envs, n_cand)

        # Importance weights proportional to exp(alpha_conf * reward).
        log_w = self.alpha_conf * rewards_mat
        log_w = log_w - log_w.max(dim=-1, keepdim=True).values  # stable softmax
        w = torch.exp(log_w)
        w = w / w.sum(dim=-1, keepdim=True)

        # Multinomial resample n from the n_cand candidates.
        idx = torch.multinomial(w, num_samples=n, replacement=True)  # (n_envs, n)
        selected = rewards_mat.gather(1, idx)  # (n_envs, n)
        return selected

    def do_transition(self, action: Tensor) -> Tensor:
        a = action.to(torch.float32).view(self.n_envs, 3).clamp(0.0, 1.0)
        reps_state = self.state.repeat_interleave(self.mc_k, dim=0)
        reps_z = self.latent_z.repeat_interleave(self.mc_k, dim=0)
        reps_u = self.latent_u.repeat_interleave(self.mc_k, dim=0)
        reps_action = a.repeat_interleave(self.mc_k, dim=0)
        drift = self._drift(reps_state, reps_action, reps_z, reps_u)
        noise = self.sigma * torch.randn_like(reps_state)
        next_state = (reps_state + self.dt * drift + noise).clamp(0.0, 1.0)
        return next_state.view(self.n_envs, self.mc_k, self.state_dim)

    def close(self) -> None:
        return None
