from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv
from causal_rl.metrics.divergences import chi_squared, kl_divergence, sup_log_ratio, total_variation
from causal_rl.metrics.mmd import mmd2
from causal_rl.metrics.plugin import plugin_tv_gap


def compute_gap_metrics(
    env: CausalEnv,
    obs: Tensor,
    action: Tensor,
    observed_reward: Tensor | None,
    pi_b_logprob: Tensor | None,
) -> dict[str, float]:
    del obs
    if env.is_discrete_action:
        p_do = env.do_reward(action)
        observed_provider = getattr(env, "observed_reward_prob", None)
        p_obs = observed_provider(action) if callable(observed_provider) else p_do
        delta_tv = plugin_tv_gap(env, action, pi_b_logprob).mean().item()
        delta_kl = kl_divergence(p_obs, p_do).mean().item()
        delta_chi2 = chi_squared(p_obs, p_do).mean().item()
        delta_sup = sup_log_ratio(p_obs, p_do).mean().item()
        return {
            "delta_tv": float(delta_tv),
            "delta_kl": float(delta_kl),
            "delta_chi2": float(delta_chi2),
            "delta_sup": float(delta_sup),
            "delta_tv_marginal": float(total_variation(p_obs, p_do).mean().item()),
            "delta_tv_conditional": float(delta_tv),
        }
    do_samples = env.do_reward(action)
    if do_samples.ndim == 1:
        do_samples = do_samples.unsqueeze(-1)
    obs_reward = observed_reward
    if obs_reward is None:
        obs_reward = do_samples.mean(dim=1)
    obs_reward = obs_reward.view(-1).to(torch.float32)
    do_samples = do_samples.to(torch.float32)

    mu_do = do_samples.mean(dim=1)
    std_do = torch.clamp(do_samples.std(dim=1), min=1e-3)
    mu_obs = obs_reward
    std_obs = torch.clamp(0.5 * std_do, min=1e-3)
    spread = torch.maximum(std_do, std_obs)
    lower = torch.minimum(mu_do, mu_obs) - 3.0 * spread
    upper = torch.maximum(mu_do, mu_obs) + 3.0 * spread
    t = torch.linspace(0.0, 1.0, steps=31, device=do_samples.device)
    grid = lower.unsqueeze(-1) + (upper - lower).unsqueeze(-1) * t.unsqueeze(0)

    def _gaussian_pdf(grid_vals: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        z = (grid_vals - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        pdf = torch.exp(-0.5 * z**2) / std.unsqueeze(-1)
        return pdf

    p_obs = _gaussian_pdf(grid, mu_obs, std_obs)
    p_do = _gaussian_pdf(grid, mu_do, std_do)
    tv = total_variation(p_obs, p_do)
    delta_tv = float(tv.mean().item())
    delta_kl = float(kl_divergence(p_obs, p_do).mean().item())
    delta_chi2 = float(chi_squared(p_obs, p_do).mean().item())
    delta_sup = float(sup_log_ratio(p_obs, p_do).mean().item())
    delta_tv_marginal = float(mmd2(obs_reward.unsqueeze(-1), do_samples.reshape(-1, 1)).item())
    return {
        "delta_tv": delta_tv,
        "delta_kl": delta_kl,
        "delta_chi2": delta_chi2,
        "delta_sup": delta_sup,
        "delta_tv_marginal": delta_tv_marginal,
        "delta_tv_conditional": delta_tv,
    }
