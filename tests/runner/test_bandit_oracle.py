"""v19: bandit oracle uses analytical optimum and marginalises Z when hidden.

Pre-v19, ``BenchmarkRunner._oracle_action`` checked
``isinstance(eval_env, TabularSepsisEnv)`` directly — False for
``BanditView`` instances — and fell through to
``algo.select_action(deterministic=True)``.  That made the bandit
"oracle" equal to the agent's current best guess, producing
nonsensical negative ``gap_to_oracle`` values.

v19 walks through wrapper layers (``getattr(env, "_env", env)``) until
it finds a ``TabularSepsisEnv``, then runs the analytical DP oracle on
that.  Additionally, ``_build_tabular_oracle_policy`` now marginalises
the per-state expected reward over Z when ``expose_z=False``, so the
oracle in C3/C4 doesn't condition on information the agent can't see.

See ``docs/v19_bandit_cell_collapse.md``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from causal_rl.envs.bandit_view import BanditView
from causal_rl.envs.tabular_sepsis import OBS_STATES, TabularSepsisEnv
from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _bandit_runner(out: Path, cell: int) -> BenchmarkRunner:
    cfg = RunnerConfig(
        cell=cell,
        env_name="bandit-tabular-sepsis-v0",
        algorithm="ucb",
        behaviour="uniform",
        seed=0,
        total_frames=4,
        n_checkpoints_train=1,
        n_checkpoints_eval=1,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=8,
        offline_updates=1,
        eval_perturbations=False,
        perturbation_grid_size=0,
        horizon=1,
        rollout_horizon=1,
        n_bootstrap=2,
        n_samples_gap=4,
        n_eval_episodes=1,
    )
    return BenchmarkRunner(cfg)


def test_bandit_oracle_uses_tabular_optimum(tmp_path: Path) -> None:
    """The bandit oracle returns the analytical argmax over actions of
    the per-state expected reward.  Pre-v19 it returned the agent's
    own deterministic action (UCB's best empirical guess) — the
    ``isinstance(eval_env, TabularSepsisEnv)`` check at the legacy site
    was False for ``BanditView`` and fell through to the algo path.
    """
    runner = _bandit_runner(tmp_path / "c1", cell=1)
    eval_env = runner.env
    assert isinstance(eval_env, BanditView), "fixture broken: expected BanditView"
    obs, _ = eval_env.reset(seed=42)

    oracle_action = runner._oracle_action(eval_env, obs).view(-1)

    # Compute the analytical optimum independently for cell=1
    # (expose_z=True): argmax_a expected_reward[latent_state, a]
    # where expected_reward[s, a] = reward_probs[s, a, 1] - reward_probs[s, a, 0].
    inner = eval_env._env
    assert isinstance(inner, TabularSepsisEnv)
    expected = inner.reward_probs[:, :, 1] - inner.reward_probs[:, :, 0]
    latent_state = inner._latent_state.to(torch.long)
    expected_argmax = expected[latent_state].argmax(dim=-1)
    assert torch.equal(oracle_action, expected_argmax), (
        f"oracle action mismatch: got {oracle_action.tolist()}, "
        f"expected {expected_argmax.tolist()}"
    )


def test_bandit_oracle_marginalises_when_z_hidden(tmp_path: Path) -> None:
    """When Z is hidden (C3/C4), the analytical oracle marginalises the
    per-state expected reward uniformly over Z.  This means the oracle
    can produce a different argmax in C3 than in C1 even when the
    underlying ``_latent_state`` is identical — because the C3 oracle
    averages the two Z-slabs while the C1 oracle reads the slab the
    latent state actually points to.

    Assert the per-env action differs in at least one env.  If never:
    the marginalised branch of ``_build_tabular_oracle_policy`` is
    producing the same argmax as the conditional branch and something
    is wrong (e.g. the marginalisation collapsed to NaN, or the
    ``cat`` reshape broadcast incorrectly).
    """
    runner_c1 = _bandit_runner(tmp_path / "c1", cell=1)
    runner_c3 = _bandit_runner(tmp_path / "c3", cell=3)

    eval_c1 = runner_c1.env
    eval_c3 = runner_c3.env
    obs_c1, _ = eval_c1.reset(seed=7)
    obs_c3, _ = eval_c3.reset(seed=7)
    # Force identical latent states across cells so the only difference
    # is the oracle's marginalisation.
    inner_c1 = eval_c1._env
    inner_c3 = eval_c3._env
    assert isinstance(inner_c1, TabularSepsisEnv)
    assert isinstance(inner_c3, TabularSepsisEnv)
    # Construct latent states that span both Z slabs so the
    # marginalisation has something to average.  We pick a state in the
    # Z=1 slab so the C1 oracle reads ``reward_probs[OBS_STATES + s_obs]``
    # but the C3 oracle reads ``0.5 * (reward_probs[s_obs] +
    # reward_probs[OBS_STATES + s_obs])`` — the two argmaxes can differ.
    s_obs = torch.arange(eval_c1.n_envs) % OBS_STATES
    inner_c1._latent_state = (OBS_STATES + s_obs).to(inner_c1._latent_state.dtype)
    inner_c3._latent_state = (OBS_STATES + s_obs).to(inner_c3._latent_state.dtype)

    a_c1 = runner_c1._oracle_action(eval_c1, obs_c1).view(-1)
    a_c3 = runner_c3._oracle_action(eval_c3, obs_c3).view(-1)

    # Independently compute both analytical oracles.
    rp = inner_c1.reward_probs
    expected_full = rp[:, :, 1] - rp[:, :, 0]
    rp_marg = 0.5 * (rp[:OBS_STATES] + rp[OBS_STATES:])
    expected_marg = rp_marg[:, :, 1] - rp_marg[:, :, 0]
    latent = inner_c1._latent_state.to(torch.long)
    expected_a_c1 = expected_full[latent].argmax(dim=-1)
    expected_a_c3 = expected_marg[latent % OBS_STATES].argmax(dim=-1)

    assert torch.equal(a_c1, expected_a_c1), (
        f"C1 oracle should equal argmax over per-Z expected reward; "
        f"got {a_c1.tolist()}, expected {expected_a_c1.tolist()}"
    )
    assert torch.equal(a_c3, expected_a_c3), (
        f"C3 oracle should equal argmax over Z-marginalised expected "
        f"reward; got {a_c3.tolist()}, expected {expected_a_c3.tolist()}"
    )
    assert not torch.equal(a_c1, a_c3), (
        f"C1 and C3 oracles produced identical actions ({a_c1.tolist()}) "
        f"on every env — fixture should pick latent states where the "
        f"marginalised argmax differs from the conditional one"
    )
