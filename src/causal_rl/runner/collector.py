from __future__ import annotations

import math

import torch

from causal_rl.behaviour.base import BiasedExplorer
from causal_rl.envs.base import CausalEnv
from causal_rl.replay.buffer import ReplayBuffer


def _resample_forbidden(
    action: torch.Tensor,
    forbidden: torch.Tensor,
    n_actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask ``forbidden`` action indices and resample uniformly over the
    allowed subset.  Returns ``(new_action, new_log_prob)``.

    The resulting log_prob reflects the *coverage-restricted* distribution
    so that ``true_behaviour_logprob`` matches what the offline buffer
    actually contains.  Mixed states (some actions forbidden, others not)
    are handled per-row.
    """
    device = action.device
    n = action.shape[0]
    forbidden_set = set(int(a) for a in forbidden.tolist())
    allowed = [a for a in range(n_actions) if a not in forbidden_set]
    if not allowed:
        # Pathological: every action forbidden.  Leave as-is and let the
        # runner observe the degeneracy.
        return action, torch.full((n,), -math.inf, device=device)
    is_forbidden = torch.zeros((n,), dtype=torch.bool, device=device)
    flat = action.view(-1).long()
    for a in forbidden_set:
        is_forbidden = is_forbidden | (flat == a)
    if not bool(is_forbidden.any()):
        # No replacement needed; report log-prob under the allowed mass.
        # Behaviour policy mass on a non-forbidden action ``a`` becomes
        # ``p_orig(a) / sum_{a' in allowed} p_orig(a')`` — but we don't
        # have the full distribution here.  Conservatively report the
        # uniform-allowed log_prob so downstream KL-style diagnostics see
        # the coverage restriction.
        lp = torch.full((n,), -math.log(float(len(allowed))), device=device)
        return action, lp
    # Resample uniformly from allowed actions for forbidden rows.
    n_resample = int(is_forbidden.sum().item())
    allowed_t = torch.tensor(allowed, dtype=action.dtype, device=device)
    replacement = allowed_t[torch.randint(0, len(allowed), (n_resample,), device=device)]
    new_flat = flat.clone()
    new_flat[is_forbidden] = replacement
    # Log-prob under the masked-uniform distribution.
    lp = torch.full((n,), -math.log(float(len(allowed))), device=device)
    return new_flat.view_as(action), lp


def collect_offline_dataset(
    env: CausalEnv,
    behaviour_policy: BiasedExplorer,
    n_transitions: int,
    expose_pi_b: bool,
    expose_latent: bool,
    seed: int,
    *,
    forbidden_actions: list[int] | None = None,
    n_action_restriction_steps: int | None = None,
) -> ReplayBuffer:
    """Collect ``n_transitions`` (s, a, r, s') tuples.

    Coverage knobs (v11):

    * ``forbidden_actions``: action indices the behaviour policy is blocked
      from selecting.  Forbidden draws are resampled uniformly from the
      allowed subset.  Produces an offline buffer where some (s, a) pairs
      are absent, driving ``min_propensity`` toward 0 for forbidden arms.
    * ``n_action_restriction_steps``: when set, the action restriction
      applies only during the first ``k`` collected transitions.  Beyond
      that, the behaviour collects normally.  Produces a *coverage
      gradient* across the buffer.
    """
    torch.manual_seed(seed)
    obs, info = env.reset(seed=seed)
    obs_dim = obs.shape[-1]
    act_dim = 1 if env.is_discrete_action else env.act_shape[0]
    buffer = ReplayBuffer(
        capacity=n_transitions,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=obs.device,
        discrete_action=env.is_discrete_action,
    )
    forbidden_t: torch.Tensor | None = None
    if forbidden_actions and env.is_discrete_action:
        forbidden_t = torch.tensor(
            list(forbidden_actions), dtype=torch.long, device=obs.device
        )
    done = torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
    # v26: read whichever latent the policy declares it needs.  Default
    # ``"latent_U"`` matches all pre-v26 policies; ``RewardAlignedZExplorer``
    # overrides to ``"latent_Z"``.  See ``docs/v26_policy_fix.md``.
    latent_key = getattr(behaviour_policy, "latent_source", "latent_U")
    while buffer.size < n_transitions:
        latent = info.get(latent_key)
        action, logprob = behaviour_policy.select_action(
            obs, latent=latent if expose_latent else None
        )
        if (
            forbidden_t is not None
            and (
                n_action_restriction_steps is None
                or buffer.size < int(n_action_restriction_steps)
            )
        ):
            # n_actions inferred from the env: 8 for tabular sepsis (the only
            # discrete-action env with action restriction in v11).
            n_actions = 8
            action, logprob = _resample_forbidden(action, forbidden_t, n_actions)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        # v26: log the latent the policy peeked at (matches
        # ``latent_source``).  Downstream consumers
        # (``_compute_extra_metrics`` cond-MI on R given (Z, A) /
        # backdoor-residual on Z) treat the latent as opaque.
        latent_logged = info.get(latent_key)
        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done.float(),
            behaviour_logprob=logprob if expose_pi_b else None,
            latent=latent_logged if expose_latent else None,
            # Always record the true π_b log-prob in a separate channel —
            # the runner uses it for π_b-recovery diagnostics regardless of
            # whether the algorithm is allowed to see it.
            true_behaviour_logprob=logprob,
        )
        obs = next_obs
        if torch.any(done):
            obs, info = env.reset()
    return buffer
