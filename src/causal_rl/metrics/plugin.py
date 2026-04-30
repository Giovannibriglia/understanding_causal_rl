from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv
from causal_rl.metrics.divergences import total_variation


def plugin_tv_gap(
    env: CausalEnv,
    action: Tensor,
    pi_b_logprob: Tensor | None,
) -> Tensor:
    p_do = env.do_reward(action)
    observed_provider = getattr(env, "observed_reward_prob", None)
    p_obs = observed_provider(action) if callable(observed_provider) else p_do
    tv = total_variation(p_obs, p_do)
    if pi_b_logprob is None:
        return tv
    w = torch.exp(-pi_b_logprob).clamp(max=10.0)
    return tv * w / (w.mean() + 1e-8)
