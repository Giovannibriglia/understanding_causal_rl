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

        # Initialise so paths that construct the bandit without an explicit
        # reset still see a valid obs/info (e.g. metric plugins, the runner's
        # ``__init__``-time env probing).  v20: the previous implementation
        # also stashed a permuted ``_context_pool`` here, but that attribute
        # was never read anywhere in the codebase and only existed to support
        # the (incorrect) permutation in ``reset``.
        obs, info = env.reset(seed=0)
        self._current_obs: Tensor = obs
        self._current_info: dict[str, Tensor] = info

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Reset the inner env and return its obs/info verbatim.

        v20: the previous implementation permuted the inner env's obs pool
        via ``obs[idx]`` where ``idx`` was a random sample of size
        ``n_envs``.  But ``BanditView.step`` calls
        ``self._env.do_reward(action)``, which indexes the inner env's
        un-permuted ``_latent_state`` — so the reward at env position ``i``
        was being computed for state ``_latent_state[i]`` even though the
        agent at position ``i`` had just observed an obs corresponding to
        state ``_latent_state[idx[i]]``.  That break in state-obs alignment
        propagated to the v19 analytical oracle: the runner reads
        ``inner._latent_state[i]`` to pick the optimal action, but the
        agent's policy was trained on (and acted from) obs corresponding
        to a different state — producing a systematic gap between the
        agent's experience and the oracle's reward function.

        The permuted ``_context_pool`` was unused downstream
        (``grep -rn _context_pool src/ tests/`` finds only the assignment
        site), so dropping the permutation has no other observable effect.
        """
        obs, info = self._env.reset(seed=seed)
        self._current_obs = obs
        self._current_info = info
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

    def sample_observational(
        self, state: Tensor | None, action: Tensor, n: int
    ) -> Tensor:
        # The wrapped env carries the observational sampling logic (latent
        # confounding, alpha_conf scaling, Z-marginalisation).  BanditView
        # is purely a single-step wrapper, so we delegate.  Without this
        # the runner's ``compute_gap_metrics`` raises NotImplementedError on
        # the first eval checkpoint of every bandit run.
        return self._env.sample_observational(state, action, n)

    def sample_interventional(
        self, state: Tensor | None, action: Tensor, n: int
    ) -> Tensor:
        return self._env.sample_interventional(state, action, n)

    def collect_observational_data(
        self,
        n_samples: int = 5000,
        alpha_conf: float = 0.0,
        seed: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Collect (action, reward_in_[0,1]) pairs from a confounded behaviour policy.

        The behaviour policy depends on the latent U; the natural-bound machinery
        is non-trivial in this regime.  Reward is in {-1, +1} for tabular sepsis;
        it is rescaled to [0, 1] via ``(r + 1) / 2`` for compatibility with
        :func:`causal_rl.identification.bounds.natural_bounds`.

        Args:
            n_samples: Total number of (action, reward) pairs to collect.
            alpha_conf: Confounding strength for the behaviour policy.
            seed: Seed for determinism.

        Returns:
            ``(actions, rewards_01)`` — actions: int64 ``(n_samples,)``;
            rewards_01: float ``(n_samples,)`` in [0, 1].
        """
        device = self._current_obs.device
        if hasattr(self._env, "alpha_conf"):
            prev_alpha = float(self._env.alpha_conf)
            self._env.alpha_conf = float(alpha_conf)
        else:
            prev_alpha = None

        torch.manual_seed(int(seed))
        n_per_reset = self.n_envs
        n_resets = max(1, (n_samples + n_per_reset - 1) // n_per_reset)
        all_acts: list[Tensor] = []
        all_rews: list[Tensor] = []

        try:
            n_actions = 8
            for r_i in range(n_resets):
                obs, info = self._env.reset(seed=int(seed) + r_i)
                latent_u = info.get("latent_U") if isinstance(info, dict) else None
                # U-conditioned action distribution: probability of action a
                # increases with U=1 and decreases with U=0 to mirror the
                # confounded behaviour policy used elsewhere in the codebase.
                u = latent_u.view(-1) if latent_u is not None else torch.zeros(
                    obs.shape[0], device=obs.device
                )
                logits = torch.zeros(obs.shape[0], n_actions, device=obs.device)
                for a in range(n_actions):
                    a_norm = float(a) / float(n_actions - 1)
                    logits[:, a] = float(alpha_conf) * (u - 0.5) * (a_norm - 0.5)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1)
                try:
                    reward_probs = self._env.do_reward(action)
                    if reward_probs.dim() == 2 and reward_probs.shape[-1] == 2:
                        reward_idx = torch.multinomial(reward_probs, 1).squeeze(-1)
                        reward = torch.where(
                            reward_idx == 1,
                            torch.ones(obs.shape[0], device=reward_probs.device),
                            -torch.ones(obs.shape[0], device=reward_probs.device),
                        )
                    else:
                        reward = reward_probs.view(obs.shape[0])
                except NotImplementedError:
                    _, reward, _, _, _ = self._env.step(action)

                all_acts.append(action.view(-1).long())
                all_rews.append(reward.view(-1).float())
        finally:
            if prev_alpha is not None:
                self._env.alpha_conf = prev_alpha

        acts_t = torch.cat(all_acts, dim=0)[:n_samples].to(device)
        rews_t = torch.cat(all_rews, dim=0)[:n_samples].to(device)
        rewards_01 = ((rews_t + 1.0) / 2.0).clamp(0.0, 1.0)
        return acts_t, rewards_01

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    def close(self) -> None:
        self._env.close()
