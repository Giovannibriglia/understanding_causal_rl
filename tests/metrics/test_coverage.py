from __future__ import annotations

import math

import torch

from causal_rl.metrics.coverage import (
    PropensityModel,
    effective_sample_size_ratio,
    fit_propensity_model,
    min_propensity,
    tail_mass_top_q,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_log_probs(n: int) -> torch.Tensor:
    """Log-probs for a uniform policy over n actions."""
    return torch.full((n,), math.log(1.0 / n))


# ---------------------------------------------------------------------------
# Uniform policy
# ---------------------------------------------------------------------------


class TestUniformPolicy:
    n_actions: int = 10

    def test_min_propensity(self) -> None:
        lp = _uniform_log_probs(self.n_actions)
        result = min_propensity(lp)
        assert abs(result.item() - 1.0 / self.n_actions) < 1e-4

    def test_effective_sample_size_ratio(self) -> None:
        lp = _uniform_log_probs(self.n_actions)
        result = effective_sample_size_ratio(lp)
        assert abs(result.item() - 1.0) < 1e-4

    def test_tail_mass_top_q(self) -> None:
        # For 10 actions uniform, top 10% = 1 arm, each with weight 0.1
        lp = _uniform_log_probs(self.n_actions)
        result = tail_mass_top_q(lp, q=0.1)
        assert abs(result.item() - 0.1) < 0.05


# ---------------------------------------------------------------------------
# Strongly biased (one-arm) policy
# ---------------------------------------------------------------------------


class TestBiasedPolicy:
    n_actions: int = 10

    def _biased_log_probs(self) -> torch.Tensor:
        """One arm gets ~0.99 weight, rest share ~0.01 equally."""
        # Construct via unnormalised logits
        logits = torch.full((self.n_actions,), math.log(0.01 / (self.n_actions - 1)))
        logits[0] = math.log(0.99)
        # Normalise: these already sum to ~1, use them directly as log_probs
        log_probs = logits - torch.logsumexp(logits, dim=0)
        return log_probs

    def test_min_propensity_near_zero(self) -> None:
        lp = self._biased_log_probs()
        result = min_propensity(lp)
        assert result.item() < 0.01

    def test_effective_sample_size_ratio_low(self) -> None:
        lp = self._biased_log_probs()
        result = effective_sample_size_ratio(lp)
        assert result.item() < 0.15

    def test_tail_mass_top_q_dominates(self) -> None:
        # Top 10% = 1 arm, and that arm carries ~0.99 of total mass
        lp = self._biased_log_probs()
        result = tail_mass_top_q(lp, q=0.1)
        assert result.item() > 0.9


# ---------------------------------------------------------------------------
# PropensityModel smoke test
# ---------------------------------------------------------------------------


class TestPropensityModelSmoke:
    obs_dim: int = 8
    n_actions: int = 4
    n_samples: int = 100

    def _make_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        obs = torch.randn(self.n_samples, self.obs_dim)
        actions = torch.randint(0, self.n_actions, (self.n_samples, 1))
        return obs, actions

    def test_fit_returns_propensity_model(self) -> None:
        obs, actions = self._make_data()
        model = fit_propensity_model(obs, actions, n_actions=self.n_actions, n_steps=10)
        assert isinstance(model, PropensityModel)

    def test_output_shape(self) -> None:
        obs, actions = self._make_data()
        model = fit_propensity_model(obs, actions, n_actions=self.n_actions, n_steps=10)
        with torch.no_grad():
            log_probs = model(obs)
        assert log_probs.shape == (self.n_samples, self.n_actions)

    def test_output_is_log_probs(self) -> None:
        """Each row should sum to ~1 in probability space."""
        obs, actions = self._make_data()
        model = fit_propensity_model(obs, actions, n_actions=self.n_actions, n_steps=10)
        with torch.no_grad():
            log_probs = model(obs)
        probs_sum = log_probs.exp().sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(self.n_samples), atol=1e-5)
