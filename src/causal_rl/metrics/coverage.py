from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from causal_rl.nets.mlp import MLP

# ---------------------------------------------------------------------------
# Pure-PyTorch coverage / propensity metrics
# ---------------------------------------------------------------------------


def min_propensity(log_probs: Tensor) -> Tensor:
    """Return exp(min(log_probs)) – the smallest propensity in the distribution.

    Args:
        log_probs: ``(N,)`` log-probabilities.

    Returns:
        Scalar tensor.
    """
    return torch.exp(log_probs.min())


def effective_sample_size_ratio(log_probs: Tensor) -> Tensor:
    """ESS / N ratio for an importance-weighting context.

    Uses the stabilised formula::

        ESS/N = (sum w)^2 / (N * sum(w^2)),   w = exp(log_probs - max(log_probs))

    Args:
        log_probs: ``(N,)`` log-probabilities.

    Returns:
        Scalar tensor in [0, 1].
    """
    w = torch.exp(log_probs - log_probs.max())
    n = torch.tensor(log_probs.shape[0], dtype=log_probs.dtype, device=log_probs.device)
    return w.sum().pow(2) / (n * w.pow(2).sum())


def support_overlap(log_probs: Tensor, tau: float = 1e-2) -> Tensor:
    """Fraction of arms whose propensity exceeds *tau*.

    Args:
        log_probs: ``(N,)`` log-probabilities.
        tau: Propensity threshold.

    Returns:
        Scalar tensor in [0, 1].
    """
    probs = torch.exp(log_probs)
    n = log_probs.shape[0]
    return (probs > tau).float().sum() / n


def tail_mass_top_q(log_probs: Tensor, q: float = 0.1) -> Tensor:
    """Total probability mass in the top-q fraction of arms by propensity.

    Computes proportional weights ``w = softmax(log_probs)`` (i.e.
    ``exp(log_probs - logsumexp(log_probs))``), sorts by propensity, and
    sums the weights of the top ``ceil(N * q)`` arms.

    Args:
        log_probs: ``(N,)`` log-probabilities.
        q: Fraction of arms to include (0 < q ≤ 1).

    Returns:
        Scalar tensor.
    """
    n = log_probs.shape[0]
    k = math.ceil(n * q)
    # Normalised weights (proportional to propensities)
    log_sum = torch.logsumexp(log_probs, dim=0)
    w = torch.exp(log_probs - log_sum)
    # Top-k indices by propensity (= by log_probs)
    top_k_indices = torch.topk(log_probs, k).indices
    return w[top_k_indices].sum()


# ---------------------------------------------------------------------------
# PropensityModel
# ---------------------------------------------------------------------------


class PropensityModel(nn.Module):
    """Small 2-layer MLP propensity estimator for discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, n_actions, hidden_dims=(64, 64))

    def forward(self, obs: Tensor) -> Tensor:
        """Return log-softmax over actions.

        Args:
            obs: ``(batch, obs_dim)`` observations.

        Returns:
            ``(batch, n_actions)`` log-probabilities.
        """
        return F.log_softmax(self.backbone(obs), dim=-1)


# ---------------------------------------------------------------------------
# fit_propensity_model
# ---------------------------------------------------------------------------


def fit_propensity_model(
    obs: Tensor,
    actions: Tensor,
    n_actions: int,
    n_steps: int = 200,
    lr: float = 1e-3,
) -> PropensityModel:
    """Train a small propensity model on offline buffer data.

    Args:
        obs: ``(N, obs_dim)`` observations.
        actions: ``(N, 1)`` integer action indices.
        n_actions: Number of discrete actions.
        n_steps: Number of Adam gradient steps.
        lr: Adam learning rate.

    Returns:
        Trained :class:`PropensityModel`.
    """
    obs_dim = obs.shape[1]
    model = PropensityModel(obs_dim, n_actions)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = obs.shape[0]
    batch_size = min(256, n)
    flat_actions = actions.squeeze(-1).long()

    for _ in range(n_steps):
        idx = torch.randint(0, n, (batch_size,))
        obs_batch = obs[idx]
        act_batch = flat_actions[idx]
        logits = model.backbone(obs_batch)
        loss = loss_fn(logits, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model
