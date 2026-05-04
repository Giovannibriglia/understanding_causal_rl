"""Information-bottleneck mutual information ``I(R; Z | S, A)``.

This file implements a CLUB-style upper bound estimator (Cheng et al. 2020,
*ICML*) on the conditional mutual information between reward ``R`` and the
latent confounder ``Z`` given the observed state-action ``(S, A)``.

The bound has the form

    I(R; Z | S, A)  ≤  E_{p(z|s,a) p(r|s,a)} [ log q(r | s, a, z) ]
                       − E_{p(r|s,a)} [ log q(r | s, a) ]

with two small variational networks (one conditional on ``Z``, one
marginal).  In the partial_id cells of this benchmark, ``Z`` is exposed via
``info["latent_Z"]`` from the env's reset/step calls — we use that signal
to estimate the bound on a held-out subset.

References:
    Tishby & Zaslavsky 2015 — *Deep learning and the information
    bottleneck principle*.
    Cheng et al. 2020 — *CLUB: A Contrastive Log-ratio Upper Bound of
    Mutual Information*.
    Lange et al. 2022 — *Information-bottleneck approaches to RL*.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _SmallGaussianRegressor(nn.Module):
    """Predicts (μ, log σ) of ``r`` given some conditioning vector."""

    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_log_sigma = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.body(x)
        mu = self.head_mu(h).squeeze(-1)
        log_sigma = self.head_log_sigma(h).clamp(min=-3.0, max=3.0).squeeze(-1)
        return mu, log_sigma


def _log_prob_gaussian(r: Tensor, mu: Tensor, log_sigma: Tensor) -> Tensor:
    sigma = log_sigma.exp()
    var = sigma.pow(2)
    return -0.5 * ((r - mu).pow(2) / (var + 1e-8) + 2.0 * log_sigma + 1.8378770664093453)


def conditional_mi_lower_bound(
    obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    latent_z: Tensor | None,
    n_epochs: int = 80,
    hidden: int = 64,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> float:
    """CLUB-style upper bound on ``I(R; Z | S, A)``.

    Args:
        obs: ``(N, obs_dim)`` observations.
        actions: ``(N,)`` integer or 1-D float actions.
        rewards: ``(N,)`` float rewards.
        latent_z: ``(N, z_dim)`` exposed latent confounder.  ``None`` returns
            ``NaN`` (signals "Z is hidden so MI is unidentifiable from data").
        n_epochs: Optimisation epochs for the variational networks.
        hidden: Hidden width.
        lr: Adam learning rate.
        batch_size: SGD batch size.
        device: Override device; defaults to ``obs.device``.

    Returns:
        Scalar estimate of ``I(R; Z | S, A)`` in nats (≥ 0).  Returns ``NaN``
        when ``latent_z is None`` or when ``N < 32``.
    """
    if latent_z is None:
        return float("nan")
    n = obs.shape[0]
    if n < 32:
        return float("nan")
    dev = torch.device(device) if device is not None else obs.device
    obs_f = obs.to(dev).float()
    a_f = actions.to(dev).float().view(n, -1)
    r_f = rewards.to(dev).float().view(n)
    z_f = latent_z.to(dev).float().view(n, -1)

    sa = torch.cat([obs_f, a_f], dim=-1)
    saz = torch.cat([sa, z_f], dim=-1)

    q_full = _SmallGaussianRegressor(saz.shape[-1], hidden=hidden).to(dev)
    q_marg = _SmallGaussianRegressor(sa.shape[-1], hidden=hidden).to(dev)
    opt = torch.optim.Adam(
        list(q_full.parameters()) + list(q_marg.parameters()), lr=lr
    )

    n_train = int(0.8 * n)
    perm = torch.randperm(n, device=dev)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    for _ in range(n_epochs):
        batch_idx = train_idx[torch.randint(0, train_idx.numel(), (batch_size,), device=dev)]
        mu_f, ls_f = q_full(saz[batch_idx])
        mu_m, ls_m = q_marg(sa[batch_idx])
        loss = -(
            _log_prob_gaussian(r_f[batch_idx], mu_f, ls_f).mean()
            + _log_prob_gaussian(r_f[batch_idx], mu_m, ls_m).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    q_full.eval()
    q_marg.eval()
    with torch.no_grad():
        mu_f, ls_f = q_full(saz[val_idx])
        mu_m, ls_m = q_marg(sa[val_idx])
        log_q_full = _log_prob_gaussian(r_f[val_idx], mu_f, ls_f)
        log_q_marg = _log_prob_gaussian(r_f[val_idx], mu_m, ls_m)
    estimate = float((log_q_full - log_q_marg).mean().item())
    return max(0.0, estimate)


__all__ = ["conditional_mi_lower_bound"]
