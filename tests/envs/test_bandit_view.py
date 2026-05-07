from __future__ import annotations

import torch

from causal_rl.envs.bandit_view import BanditView
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv


def _make_bandit(cell: int = 1, n_envs: int = 4) -> BanditView:
    env = TabularSepsisEnv(cell=cell, n_envs=n_envs)
    return BanditView(env)


def test_bandit_horizon_is_one() -> None:
    bandit = _make_bandit()
    assert bandit.horizon == 1


def test_bandit_step_returns_done() -> None:
    bandit = _make_bandit()
    bandit.reset()
    action = torch.zeros(bandit.n_envs, 1, dtype=torch.long)
    _obs, _reward, terminated, truncated, _info = bandit.step(action)
    assert terminated.all(), "BanditView.step should return terminated=True for all envs"
    assert not truncated.any(), "BanditView.step should return truncated=False for all envs"


def test_bandit_obs_shape_matches_env() -> None:
    inner = TabularSepsisEnv(cell=1, n_envs=4)
    bandit = BanditView(inner)
    assert bandit.obs_shape == inner.obs_shape

    obs, _ = bandit.reset()
    assert obs.shape[-len(inner.obs_shape) :] == inner.obs_shape


def test_bandit_state_action_reward_alignment() -> None:
    """v20: ``BanditView._current_obs[i]`` must be the obs for the same
    state ``inner._latent_state[i]`` that ``BanditView.step`` will use to
    compute rewards.  Pre-v20 ``reset`` permuted the obs pool via
    ``obs[idx]`` while ``step`` indexed ``_latent_state`` directly,
    breaking that invariant.

    The precise check: after ``bandit.reset(seed=...)``, the obs the
    agent sees for env position ``i`` should equal what ``inner._observe()``
    would return for the same position — i.e. the inner env's own obs
    for state ``inner._latent_state[i]``.
    """
    inner = TabularSepsisEnv(cell=1, n_envs=8, device="cpu")
    bandit = BanditView(inner)
    bandit.reset(seed=42)

    expected_obs = inner._observe()
    assert torch.equal(bandit._current_obs, expected_obs), (
        "bandit obs at position i does not match inner obs at position i — "
        "the reset path is permuting state-obs alignment"
    )
    # Stronger: the obs decodes back to the same latent state used by step.
    decoded = inner._infer_latent_state_from_obs(bandit._current_obs)
    assert torch.equal(decoded, inner._latent_state), (
        "agent obs decodes to a different latent state than the one the "
        "env's reward computation will use"
    )


def test_bandit_oracle_action_recovers_optimum_through_step() -> None:
    """v20: the optimal action computed *from the agent's obs* should
    yield rewards consistent with that action's expected reward in the
    env's own reward computation.  Pre-v20 the obs the agent saw and
    the latent state the env rewarded against were drawn from
    independent permutations, so an action chosen on the basis of the
    obs (the only signal the agent has) gave rewards uncorrelated with
    its expected value.

    Construction:
    * Reset bandit; decode latent state from each agent obs.
    * Compute oracle action per env using the decoded state.
    * Step 200 times (``BanditView.step`` does not reset
      ``inner._latent_state`` between calls) and average the reward
      per env position.
    * Compare to the expected reward for that (decoded-state, action)
      pair.  With Bernoulli σ ≈ √(0.25/200) ≈ 0.035, a 0.05 tolerance
      is comfortable.

    Pre-v20 the decoded state corresponds to ``_latent_state[idx[i]]``
    while ``step`` rewards using ``_latent_state[i]`` — empirical
    rewards are drawn from a different state than the action targets.
    """
    inner = TabularSepsisEnv(cell=1, n_envs=16, device="cpu")
    bandit = BanditView(inner)
    bandit.reset(seed=7)

    decoded = inner._infer_latent_state_from_obs(bandit._current_obs)
    expected = inner.reward_probs[:, :, 1] - inner.reward_probs[:, :, 0]
    oracle_action = expected[decoded].argmax(dim=-1).view(-1, 1)

    n_steps = 200
    total = torch.zeros(bandit.n_envs)
    for _ in range(n_steps):
        _, reward, _, _, _ = bandit.step(oracle_action)
        total += reward
    empirical = total / n_steps

    expected_reward = expected[decoded].gather(1, oracle_action).view(-1)
    gap = (empirical - expected_reward).abs().max().item()
    assert gap < 0.05, (
        f"empirical reward at oracle action does not match decoded-state "
        f"expected reward: max gap {gap:.4f} on n={n_steps} steps "
        f"(expect <0.05).  Pre-v20 the obs the agent saw and the state "
        f"the env rewards against are independent permutations."
    )
