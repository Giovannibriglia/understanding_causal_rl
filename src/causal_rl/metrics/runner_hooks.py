from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv
from causal_rl.metrics.divergences import chi_squared, kl_divergence, sup_log_ratio, total_variation
from causal_rl.metrics.mmd import mmd2

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _bootstrap_tv_ci(
    obs_samples: Tensor,
    do_samples: Tensor,
    n_bootstrap: int = 200,
) -> tuple[float, float]:
    """95% paired-bootstrap CI for mean TV distance between two sample tensors.

    obs_samples / do_samples: (batch, n_samples) containing {-1, +1}.
    Returns (ci_lo, ci_hi).  When n_bootstrap <= 0, returns (point, point).
    """
    batch, n = obs_samples.shape
    # Point estimate (used as fallback when n_bootstrap=0)
    p_obs = ((obs_samples == 1).float().sum(-1) + 1.0) / (n + 2.0)
    p_do = ((do_samples == 1).float().sum(-1) + 1.0) / (n + 2.0)
    point = float((p_obs - p_do).abs().mean().item())
    if n_bootstrap <= 0:
        return point, point
    boot_tvs: list[float] = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), device=obs_samples.device)
        b_obs = obs_samples[:, idx]
        b_do = do_samples[:, idx]
        p_obs_b = ((b_obs == 1).float().sum(-1) + 1.0) / (n + 2.0)
        p_do_b = ((b_do == 1).float().sum(-1) + 1.0) / (n + 2.0)
        boot_tvs.append(float((p_obs_b - p_do_b).abs().mean().item()))
    boot_tvs.sort()
    lo_idx = max(0, int(0.025 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    return boot_tvs[lo_idx], boot_tvs[hi_idx]


def _bootstrap_mmd_ci(
    obs_samples: Tensor,
    do_samples: Tensor,
    n_bootstrap: int = 200,
) -> tuple[float, float]:
    """95% paired-bootstrap CI for MMD² between two continuous sample tensors.

    obs_samples / do_samples: (batch, n_samples) scalar rewards.
    Computes MMD over the pooled batch dimension for each bootstrap replicate.
    Returns (ci_lo, ci_hi).  When n_bootstrap <= 0, returns (point, point).
    """
    batch, n = obs_samples.shape
    point = float(mmd2(obs_samples.reshape(-1, 1), do_samples.reshape(-1, 1)).item())
    if n_bootstrap <= 0:
        return point, point
    boot_vals: list[float] = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), device=obs_samples.device)
        b_obs = obs_samples[:, idx].reshape(-1, 1)
        b_do = do_samples[:, idx].reshape(-1, 1)
        boot_vals.append(float(mmd2(b_obs, b_do).item()))
    boot_vals.sort()
    lo_idx = max(0, int(0.025 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    return boot_vals[lo_idx], boot_vals[hi_idx]


# ---------------------------------------------------------------------------
# KS distance between two empirical distributions (no parametric assumption)
# ---------------------------------------------------------------------------


def _empirical_ks(x: Tensor, y: Tensor) -> Tensor:
    """KS statistic between two sets of samples.

    x, y: (batch, n_samples).  Returns scalar tensor = mean over batch.
    """
    combined = torch.cat([x, y], dim=-1)  # (batch, 2n)
    n = x.shape[-1]
    ks_vals = []
    for i in range(combined.shape[0]):
        vals, _ = torch.sort(combined[i])
        cx = torch.searchsorted(torch.sort(x[i]).values, vals, right=True).float() / n
        cy = torch.searchsorted(torch.sort(y[i]).values, vals, right=True).float() / n
        ks_vals.append(float((cx - cy).abs().max().item()))
    return torch.tensor(ks_vals, device=x.device).mean()


# ---------------------------------------------------------------------------
# Main gap-metric function
# ---------------------------------------------------------------------------


def compute_gap_metrics(
    env: CausalEnv,
    obs: Tensor,
    action: Tensor,
    observed_reward: Tensor | None,
    pi_b_logprob: Tensor | None,
    n_samples: int = 200,
    n_bootstrap: int = 200,
) -> dict[str, float]:
    """Estimate Δ_φ = φ(P̂_obs, P̂_do) from samples drawn by the env.

    For **discrete** envs (tabular sepsis):
      - Draws n_samples from sample_observational and sample_interventional.
      - Estimates P̂_obs and P̂_do via Laplace-smoothed plug-in counts.
      - Reports TV as primary metric plus bootstrap 95% CI.
      - Falls back to observed_reward_prob / do_reward if sampling methods
        are not available (backward-compatible).

    For **continuous** envs (continuous ward):
      - Draws n_samples from both methods (or falls back to do_reward MC
        samples as interventional, and observed_reward as observational proxy).
      - Computes MMD² with median-heuristic bandwidth as primary metric.
      - Also reports KS distance between empirical CDFs.
      - Adds bootstrap 95% CI for MMD².
    """
    has_sampling = (
        hasattr(env, "sample_observational")
        and hasattr(env, "sample_interventional")
        and callable(env.sample_observational)
        and callable(env.sample_interventional)
    )

    # ------------------------------------------------------------------
    # Discrete (tabular) path
    # ------------------------------------------------------------------
    if env.is_discrete_action:
        if has_sampling:
            obs_samp = env.sample_observational(obs, action, n_samples)  # (batch, n)
            do_samp = env.sample_interventional(obs, action, n_samples)  # (batch, n)

            n = obs_samp.shape[-1]
            # Laplace-smoothed Bernoulli probabilities
            p_obs_hat = ((obs_samp == 1).float().sum(-1) + 1.0) / (n + 2.0)
            p_do_hat = ((do_samp == 1).float().sum(-1) + 1.0) / (n + 2.0)

            delta_tv = float((p_obs_hat - p_do_hat).abs().mean().item())
            # Stack into (batch, 2) for existing divergence functions
            p_obs_2d = torch.stack([1.0 - p_obs_hat, p_obs_hat], dim=-1)
            p_do_2d = torch.stack([1.0 - p_do_hat, p_do_hat], dim=-1)
            delta_kl = float(kl_divergence(p_obs_2d, p_do_2d).mean().item())
            delta_chi2 = float(chi_squared(p_obs_2d, p_do_2d).mean().item())
            delta_sup = float(sup_log_ratio(p_obs_2d, p_do_2d).mean().item())

            ci_lo, ci_hi = _bootstrap_tv_ci(obs_samp, do_samp, n_bootstrap)

            # Marginal vs conditional split:
            # delta_tv_marginal = TV between marginal P̂_obs and P̂_do
            # delta_tv_conditional = same as delta_tv (IPW-corrected in caller when pi_b_known)
            delta_tv_marginal = float(total_variation(p_obs_2d, p_do_2d).mean().item())
            delta_tv_conditional = delta_tv

        else:
            # Backward-compatible fallback: use observed_reward_prob / do_reward
            p_do = env.do_reward(action)
            observed_provider = getattr(env, "observed_reward_prob", None)
            p_obs = observed_provider(action) if callable(observed_provider) else p_do
            delta_tv = float(total_variation(p_obs, p_do).mean().item())
            delta_kl = float(kl_divergence(p_obs, p_do).mean().item())
            delta_chi2 = float(chi_squared(p_obs, p_do).mean().item())
            delta_sup = float(sup_log_ratio(p_obs, p_do).mean().item())
            ci_lo = delta_tv
            ci_hi = delta_tv
            delta_tv_marginal = delta_tv
            delta_tv_conditional = delta_tv

        # IPW reweighting of TV when behaviour log-probs are available.
        if pi_b_logprob is not None and has_sampling:
            w = torch.exp(-pi_b_logprob).clamp(max=10.0)
            w_norm = w / (w.mean() + 1e-8)
            p_obs_hat_ipw = p_obs_hat * w_norm
            p_do_hat_ipw = p_do_hat
            delta_tv_conditional = float((p_obs_hat_ipw - p_do_hat_ipw).abs().mean().item())

        # v13: also compute MMD² and KS on the discrete sample tensors.
        # These are the "robustness check" divergences plotted by
        # ``metrics_curves`` and ``divergence_panel``.  Pre-v13 the discrete
        # path returned 0.0 for both, leaving the corresponding panels
        # rendered as flat lines or empty bars on tabular runs.
        delta_mmd2 = 0.0
        delta_ks = 0.0
        if has_sampling:
            obs_flat = obs_samp.to(torch.float32)
            do_flat = do_samp.to(torch.float32)
            try:
                delta_mmd2 = float(
                    mmd2(obs_flat.reshape(-1, 1), do_flat.reshape(-1, 1)).item()
                )
            except Exception:  # noqa: BLE001
                delta_mmd2 = 0.0
            try:
                delta_ks = float(_empirical_ks(obs_flat, do_flat).item())
            except Exception:  # noqa: BLE001
                delta_ks = 0.0

        return {
            "delta_tv": delta_tv,
            "delta_tv_ci_lo": ci_lo,
            "delta_tv_ci_hi": ci_hi,
            "delta_kl": delta_kl,
            "delta_chi2": delta_chi2,
            "delta_sup": delta_sup,
            "delta_mmd2": delta_mmd2,
            "delta_ks": delta_ks,
            "delta_tv_marginal": delta_tv_marginal,
            "delta_tv_conditional": delta_tv_conditional,
        }

    # ------------------------------------------------------------------
    # Continuous path  (A.2: replace Gaussianised L1 with MMD² + KS)
    # ------------------------------------------------------------------
    if has_sampling:
        obs_samp = env.sample_observational(obs, action, n_samples)  # (batch, n)
        do_samp = env.sample_interventional(obs, action, n_samples)  # (batch, n)
    else:
        # Fallback: do_reward gives MC samples for interventional; use
        # observed_reward (single-sample) as a degenerate observational estimate.
        do_samp = env.do_reward(action)  # (batch, mc_k)
        if do_samp.ndim == 1:
            do_samp = do_samp.unsqueeze(-1)
        if observed_reward is not None:
            obs_samp = observed_reward.view(-1, 1).expand_as(do_samp)
        else:
            obs_samp = do_samp

    obs_flat = obs_samp.to(torch.float32)
    do_flat = do_samp.to(torch.float32)

    # MMD² with median-heuristic bandwidth (primary continuous metric)
    delta_mmd2 = float(mmd2(obs_flat.reshape(-1, 1), do_flat.reshape(-1, 1)).item())

    # KS distance between pooled empirical CDFs
    delta_ks = float(_empirical_ks(obs_flat, do_flat).item())

    # Bootstrap CI for MMD²
    ci_lo, ci_hi = _bootstrap_mmd_ci(obs_flat, do_flat, n_bootstrap)

    # For backward compatibility with callers that read delta_tv:
    # report KS as delta_tv (both are in [0,1] for bounded rewards).
    return {
        "delta_tv": delta_ks,  # KS ∈ [0,1], replaces Gaussianised L1
        "delta_tv_ci_lo": ci_lo,
        "delta_tv_ci_hi": ci_hi,
        "delta_kl": 0.0,  # not meaningful without parametric assumption
        "delta_chi2": 0.0,
        "delta_sup": 0.0,
        "delta_mmd2": delta_mmd2,  # primary continuous metric
        "delta_ks": delta_ks,
        "delta_tv_marginal": delta_mmd2,  # MMD² as marginal summary
        "delta_tv_conditional": delta_ks,
    }
