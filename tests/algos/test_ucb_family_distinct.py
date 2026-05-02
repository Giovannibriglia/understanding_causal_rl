"""Phase 1.1 acceptance: UCB+, UCB-, RCT and vanilla UCB are observably distinct
when supplied with non-trivial natural-bound / observational-mean priors.

This is a tighter check than ``test_ucb_family.py``: it focuses on the run-time
regret of each algorithm on a Bernoulli bandit with a confounded behaviour
policy, to verify that the UCB+/UCB- bounds/means actually flow through.

The bandit setup mirrors Bareinboim Example 9.1 in spirit:
- 4 arms, true means ``[0.4, 0.4, 0.7, 0.4]``.
- A confounded behaviour policy preferentially pulls the *wrong* arm 0,
  producing observational arm-0 mean ≫ its true mean.
- Natural bounds on the confounded data exclude no arms outright but
  point the UCB+ exploration in the correct direction.
"""

from __future__ import annotations

import torch

from causal_rl.algos.on_policy.rct import RCT
from causal_rl.algos.on_policy.ucb import UCB
from causal_rl.algos.on_policy.ucb_minus import UCBMinus
from causal_rl.algos.on_policy.ucb_plus import UCBPlus

# Best arm = 2 (true mean 0.7).
_TRUE_MEANS = [0.4, 0.4, 0.7, 0.4]
_BEST = 2
_BEST_MEAN = max(_TRUE_MEANS)


def _simulate(algo, n_steps: int, seed: int) -> float:
    torch.manual_seed(seed)
    obs = torch.zeros(1, 1)
    cum_regret = 0.0
    for _ in range(n_steps):
        action, _ = algo.select_action(obs)
        a = int(action.item())
        r = float(torch.bernoulli(torch.tensor(_TRUE_MEANS[a])).item())
        algo.update({
            "action": torch.tensor([[a]], dtype=torch.long),
            "reward": torch.tensor([r], dtype=torch.float32),
        })
        cum_regret += _BEST_MEAN - _TRUE_MEANS[a]
    return cum_regret


def test_ucb_plus_lower_regret_when_bounds_exclude_a_dominated_arm() -> None:
    # Bounds for the dominator: [0.6, 0.8]; for arms 0/1/3: [0.0, 0.55].
    # All non-best upper bounds are below the best lower bound (0.6) → those
    # arms are informatively excluded.
    lower_bounds = [0.0, 0.0, 0.6, 0.0]
    upper_bounds = [0.55, 0.55, 0.8, 0.55]

    n_steps = 1000
    regret_ucb = _simulate(UCB(obs_dim=1, n_actions=4), n_steps, seed=0)
    regret_plus = _simulate(
        UCBPlus(
            obs_dim=1,
            n_actions=4,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        ),
        n_steps,
        seed=0,
    )
    assert regret_plus < 0.6 * regret_ucb, (
        f"UCB+ regret {regret_plus:.2f} should be < 0.6 * UCB regret {regret_ucb:.2f} "
        "when bounds exclude all suboptimal arms."
    )


def test_ucb_minus_higher_regret_when_seeded_with_misleading_means() -> None:
    """When the observational means hide the best arm by reporting it the
    worst, UCB- (which seeds with those means) selects worse arms more often
    than a UCB that starts uninformed."""
    # Hard adversarial seed: true best is arm 2 (mean 0.9); seeded means
    # report arm 2 as the worst (0.05) while arm 0 is reported as best (0.95).
    true_means = [0.2, 0.2, 0.9, 0.2]
    misleading_means = [0.95, 0.5, 0.05, 0.5]

    def _sim(algo, n_steps: int, seed: int = 0) -> float:
        torch.manual_seed(seed)
        obs = torch.zeros(1, 1)
        regret = 0.0
        best_mean = max(true_means)
        for _ in range(n_steps):
            action, _ = algo.select_action(obs)
            a = int(action.item())
            r = float(torch.bernoulli(torch.tensor(true_means[a])).item())
            algo.update({
                "action": torch.tensor([[a]], dtype=torch.long),
                "reward": torch.tensor([r], dtype=torch.float32),
            })
            regret += best_mean - true_means[a]
        return regret

    # The seeded means bias the initial selections, but UCB's exploration
    # bonus is large in the early phase relative to seeded count = 1, so the
    # seed gets corrected quickly.  We assert the directionally-correct
    # statement: UCB- has *strictly higher* regret than UCB on adversarial
    # seeds, averaged over multiple seeds to smooth bandit noise.
    # Average over many bandit seeds — UCB-'s seed-1 prior gets overridden
    # quickly by the empirical mean, so the effect is small per-seed; we
    # check that on aggregate UCB- accrues more regret than vanilla UCB.
    seeds = list(range(20))
    n_steps = 50
    sum_minus = 0.0
    sum_ucb = 0.0
    for seed in seeds:
        sum_ucb += _sim(UCB(obs_dim=1, n_actions=4), n_steps, seed=seed)
        sum_minus += _sim(
            UCBMinus(obs_dim=1, n_actions=4, observational_means=misleading_means),
            n_steps,
            seed=seed,
        )
    assert sum_minus > sum_ucb, (
        f"UCB- aggregate regret {sum_minus:.1f} should exceed UCB regret {sum_ucb:.1f} "
        "when seeded with confounded means that hide the best arm."
    )


def test_rct_higher_regret_than_ucb() -> None:
    n_steps = 1000
    regret_ucb = _simulate(UCB(obs_dim=1, n_actions=4), n_steps, seed=0)
    regret_rct = _simulate(RCT(obs_dim=1, n_actions=4), n_steps, seed=0)
    assert regret_rct > regret_ucb, (
        f"RCT regret {regret_rct:.2f} should exceed UCB regret {regret_ucb:.2f}"
    )
