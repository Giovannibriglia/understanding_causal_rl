"""v26: Z-peeking reward-aligned explorer.

Companion to :class:`RewardAlignedExplorer`.  The legacy explorer
peeks at ``latent_U`` (an independent ``Bernoulli(0.5)`` per reset)
and produces near-uniform action distributions because U is
uncorrelated with the reward-driving latent Z (verified by v25.2 /
v25.3 audits at ``α_conf=4``: Z-conditional ratios 1.08 / 1.28).
This class peeks at ``latent_Z`` directly and biases action selection
toward the per-Z optimal arm with strength controlled by
``alpha_conf``.

Backward compat:
* ``alpha_conf=0`` produces a uniform distribution over actions (the
  weights formula collapses to ``[1, 1, ..., 1]``), matching the
  legacy ``α_conf=0`` rows in summaries within Bernoulli noise.
* The legacy ``RewardAlignedExplorer`` is preserved as a baseline
  and remains the default for ``coverage_stress.yaml`` (whose
  coverage-break mechanism is action-masking, not confounding —
  see v25.4 Phase 1).

See ``docs/v26_policy_fix.md`` for the design rationale and
``tests/policies/test_reward_aligned_z_explorer.py`` for the gate
test that this class must pass before any benchmark sweep.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class RewardAlignedZExplorer(BiasedExplorer):
    family = "reward_aligned"
    depends_on_u = True  # gates the runner's ``expose_latent`` flag
    latent_source = "latent_Z"  # collector reads info[this key]

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        alpha_conf: float = 0.0,
        bias_strength: float = 1.0,
        requires_latent: bool = True,
        env: Any = None,
        **_unused: Any,
    ) -> None:
        super().__init__(bias_strength=bias_strength)
        if n_actions is None and act_dim is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        if act_dim is not None:
            msg = (
                "RewardAlignedZExplorer is discrete-action only.  Use "
                "RewardAlignedExplorer for continuous-action envs."
            )
            raise ValueError(msg)
        self.n_actions = int(n_actions) if n_actions is not None else 0
        self.alpha_conf = float(alpha_conf)
        self.requires_latent = requires_latent
        # Per-Z preferred arm cache: ``_z_argmax[z]`` is the action with
        # the highest E[r | Z=z] under uniform marginalisation over the
        # observable state.  Computed eagerly from ``env.reward_probs``
        # if env is provided, lazily on first ``select_action`` call
        # otherwise.  See ``_compute_z_argmax``.
        self._z_argmax: Tensor | None = None
        if env is not None:
            self._z_argmax = self._compute_z_argmax(env)

    @staticmethod
    def _compute_z_argmax(env: Any) -> Tensor:
        """Per-Z argmax of E[r | Z=z, A=a] under uniform marginalisation
        over the observable state.

        ``reward_probs`` has shape ``(2 * OBS_STATES, N_ACTIONS, 2)``.
        Rows ``[:OBS_STATES]`` are Z=0, rows ``[OBS_STATES:]`` are Z=1.
        Per-action expected reward conditional on Z is the mean over
        the OBS_STATES rows of ``reward_probs[:, a, 1]`` (since reward
        ∈ {-1, +1} with P(R=+1) given by that column).
        """
        from causal_rl.envs.tabular_sepsis import OBS_STATES

        underlying = getattr(env, "_env", env)  # unwrap BanditView
        rp = underlying.reward_probs
        z0_mean = rp[:OBS_STATES, :, 1].mean(dim=0)
        z1_mean = rp[OBS_STATES:, :, 1].mean(dim=0)
        return torch.stack([z0_mean.argmax(), z1_mean.argmax()]).to(torch.long)

    def _ensure_z_argmax(self, latent: Tensor) -> Tensor:
        if self._z_argmax is None:
            msg = (
                "RewardAlignedZExplorer was constructed without an env "
                "reference; per-Z preferred arms are unknown.  Pass "
                "``env=<TabularSepsisEnv or BanditView>`` at construction."
            )
            raise RuntimeError(msg)
        return self._z_argmax.to(latent.device)

    def select_action(
        self, obs: Tensor, latent: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample actions with weights ``[1, ..., exp(α_conf), ..., 1]``
        where the boosted index is the per-Z preferred arm.

        At ``α_conf=0``, all weights are 1 ⇒ uniform.
        At ``α_conf=4``, the preferred arm has ~88% of the mass
        (``exp(4) / (exp(4) + 7) ≈ 0.886``); Z-conditional sample
        ratio on a 4000-sample rollout is ≥ 5:1 by a wide margin.

        ``latent`` is a tensor of Z values, shape ``(batch,)`` or
        ``(batch, 1)`` with values in ``{0, 1}``.  The runner's
        collector reads it from ``info[self.latent_source]`` =
        ``info["latent_Z"]``.
        """
        if latent is None:
            msg = (
                "RewardAlignedZExplorer requires latent_Z; the runner "
                "must construct with depends_on_u=True (gates "
                "expose_latent), and the env must expose latent_Z in "
                "the info dict (TabularSepsisEnv does this by default)."
            )
            raise ValueError(msg)
        z_argmax = self._ensure_z_argmax(latent)

        batch = obs.shape[0]
        device = obs.device
        z = latent.view(batch).long().clamp(0, 1)
        preferred = z_argmax[z]  # (batch,) of arm indices

        # ``boost = exp(α) - 1``; non-preferred arms keep weight 1, the
        # preferred arm gets weight ``1 + boost = exp(α)``.  Scatter-add
        # avoids materialising a one-hot tensor.
        boost = math.exp(self.alpha_conf) - 1.0
        weights = torch.ones((batch, self.n_actions), device=device)
        weights.scatter_add_(
            1,
            preferred.unsqueeze(1),
            torch.full((batch, 1), boost, device=device),
        )
        probs = weights / weights.sum(dim=-1, keepdim=True)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return self._mix_with_uniform(
            obs, action.unsqueeze(-1), dist.log_prob(action), n_actions=self.n_actions
        )
