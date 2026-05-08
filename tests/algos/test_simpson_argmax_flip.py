"""v25 gate test: on the simpson confounding profile, naive picks a
side arm and IPW recovers a middle (do-optimal under marginal P(Z))
arm.

This is the unit-level demonstration of the controlled-pair failure
case the bandit section of the paper hinges on.  Pre-v25 (smooth
profile) naive and IPW recover the same argmax — the v22 paper_bandit
offpolicy run produced bit-identical aggregate stats across the two
algos in C1/C3/C4 because TabularSepsis's smooth reward profile
doesn't admit argmax-flipping confounding.

The simpson profile (``_build_reward_simpson``) intentionally
constructs the Simpson's-paradox structure: side arms (a=0, a=N-1)
have high P_HIGH=0.85 reward under one Z and low P_LOW=0.10 under the
other, while middle arms have moderate P_MID=0.55 reward under both.
A behaviour policy that peeks at Z and prefers the per-Z best side
arm produces a confounded buffer where:
* naive sample-mean of arm 0 ≈ P_HIGH (only seen when Z=0)
* naive sample-mean of arm 7 ≈ P_HIGH (only seen when Z=1)
* naive sample-means of middle arms ≈ P_MID, but with high variance
  (sampled rarely)
Naive's argmax: arm 0 or arm 7.  IPW with true logprobs reweights
toward the marginal mean: arm 0 ≈ 0.475, arm 7 ≈ 0.475, middle arms
≈ 0.55.  IPW's argmax: a middle arm.

If this test fails, retune ``P_HIGH`` / ``P_LOW`` / ``P_MID`` /
``p_preferred`` in the env / fixture.  See iteration protocol in
``docs/v25_simpson_bandit.md``.
"""

from __future__ import annotations

import math

import torch

import causal_rl.envs.tabular_sepsis as tabular_sepsis
from causal_rl.algos.off_policy.offline_bandit_ipw import OfflineBanditIPW
from causal_rl.algos.off_policy.offline_bandit_naive import OfflineBanditNaive
from causal_rl.envs.tabular_sepsis import (
    N_ACTIONS,
    OBS_STATES,
    TabularSepsisEnv,
)


def _make_confounded_batch(
    env: TabularSepsisEnv,
    n_samples: int = 4000,
    p_preferred: float = 0.3,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Sample (action, reward, log π_b(a)) tuples from a Z-peeking
    behaviour policy.

    Behaviour policy: with probability ``p_preferred``, pick the
    per-Z preferred side arm (arm 0 if Z=0, arm N-1 if Z=1);
    otherwise pick uniformly over the remaining N-1 arms.  Every
    transition uses the env's current ``_latent_state`` so the
    reward distribution matches what the agent would experience.
    """
    torch.manual_seed(seed)

    # Build env with n_envs == n_samples so a single reset gives us
    # n_samples i.i.d. (z, state) draws.
    env.reset(seed=seed)
    z = (env._latent_state // OBS_STATES).long()  # (n_samples,)

    # Per-env action sampling under the Z-peeking behaviour.
    # P(a=preferred|z) = p_preferred; remaining mass split uniformly.
    p_other = (1.0 - p_preferred) / (N_ACTIONS - 1)
    n = z.numel()
    action_probs = torch.full((n, N_ACTIONS), float(p_other))
    preferred = torch.where(
        z == 0, torch.zeros(n, dtype=torch.long), torch.full((n,), N_ACTIONS - 1)
    )
    action_probs.scatter_(1, preferred.view(-1, 1), p_preferred)
    action = torch.multinomial(action_probs, 1).squeeze(-1)  # (n,)
    logp = torch.log(action_probs.gather(1, action.view(-1, 1)).squeeze(-1))

    # Sample rewards from the env's do_reward distribution at the
    # taken action (n_envs == n_samples, so vectorised).
    reward_probs = env.do_reward(action.view(-1, 1))  # (n, 2)
    bernoulli_pos = torch.bernoulli(reward_probs[:, 1])
    reward = torch.where(bernoulli_pos > 0, torch.ones(n), -torch.ones(n))

    return {
        "action": action.long(),
        "reward": reward.float(),
        "behaviour_logprob": logp.float(),
    }


def test_simpson_env_flips_naive_vs_ipw_argmax() -> None:
    """The headline contract: on the simpson profile + Z-peeking
    behaviour with ``p_preferred=0.9``, naive's argmax is a side arm
    and IPW's argmax is a middle arm.

    Pre-v25 (smooth profile) both estimators recover the same
    argmax; the test fails on the smooth profile, which is the
    behavioural regression target.
    """
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = {}

    n_samples = 4000
    env = TabularSepsisEnv(
        cell=2,
        n_envs=n_samples,
        alpha_conf=4.0,
        device="cpu",
        confounding_profile="simpson",
    )
    batch = _make_confounded_batch(env, n_samples=n_samples, seed=0)

    naive = OfflineBanditNaive(
        obs_dim=env.obs_shape[0], n_actions=N_ACTIONS,
        prior_count=0.0, prior_mean=0.0,
    )
    ipw = OfflineBanditIPW(
        obs_dim=env.obs_shape[0], n_actions=N_ACTIONS,
        prior_count=0.0, prior_mean=0.0, ipw_clip=10.0,
    )
    naive.update(batch)
    ipw.update(batch)

    naive_argmax = int(naive.mu_hat.argmax().item())
    ipw_argmax = int(ipw.mu_hat.argmax().item())

    fail_msg = (
        f"\nnaive mu_hat: {[round(x, 3) for x in naive.mu_hat.tolist()]}"
        f"\nipw   mu_hat: {[round(x, 3) for x in ipw.mu_hat.tolist()]}"
        f"\nnaive_argmax={naive_argmax}, ipw_argmax={ipw_argmax}"
    )
    assert naive_argmax in (0, N_ACTIONS - 1), (
        f"naive should pick a side arm under Z-peeking confounded data; "
        f"got {naive_argmax}.{fail_msg}"
    )
    assert ipw_argmax not in (0, N_ACTIONS - 1), (
        f"IPW with true behaviour logprobs should pick a middle arm "
        f"(do-optimal under marginal P(Z)); got {ipw_argmax}.{fail_msg}"
    )
    assert naive_argmax != ipw_argmax, fail_msg


def test_simpson_naive_picks_side_arm_with_biased_estimated_mean() -> None:
    """Defence-in-depth: not only should naive pick a side arm, its
    sample-mean estimate of that arm should be ~0.32 on [-1, +1]
    (the analytical conditional bias E[r | a=side]).  This confirms
    the bias mechanism rather than just the symptom.

    Math: with ``p_preferred=0.3`` and uniform P(Z),
    ``P(z=0 | a=0) = 0.3*0.5 / (0.3*0.5 + 0.1*0.5) = 0.75``.
    ``E[r | a=0] = 0.75 * 0.7 + 0.25 * -0.8 = 0.325``.
    Tolerance 0.07 covers Bernoulli noise on n=800 samples
    (σ ≈ √(1/800) ≈ 0.035, so 0.07 is ~2σ).
    """
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = {}

    n_samples = 4000
    env = TabularSepsisEnv(
        cell=2, n_envs=n_samples, alpha_conf=4.0, device="cpu",
        confounding_profile="simpson",
    )
    batch = _make_confounded_batch(env, n_samples=n_samples, seed=0)

    naive = OfflineBanditNaive(
        obs_dim=env.obs_shape[0], n_actions=N_ACTIONS,
        prior_count=0.0, prior_mean=0.0,
    )
    naive.update(batch)
    side_arm = int(naive.mu_hat.argmax().item())
    expected_naive_mean = 0.75 * 0.7 + 0.25 * -0.8
    gap = abs(float(naive.mu_hat[side_arm].item()) - expected_naive_mean)
    assert gap < 0.07, (
        f"naive sample-mean of side arm should be near analytical "
        f"conditional bias ({expected_naive_mean:.3f}); "
        f"got {naive.mu_hat[side_arm].item():.3f} (gap {gap:.3f}).  "
        f"This indicates the confounding mechanism is not behaving "
        f"as designed."
    )


def test_smooth_profile_does_not_flip_argmax() -> None:
    """Behavioural regression target for v25: under the smooth profile,
    the same fixture does NOT produce an argmax flip — naive and IPW
    recover the same argmax (or both side arms / both middle arms).

    This is what the v22 paper_bandit_offpolicy run observed
    empirically: bit-identical aggregate stats across the two algos
    in C1/C3/C4.  This test pins that pre-v25 behaviour so a future
    refactor of the smooth profile that *coincidentally* produces a
    flip would be flagged for review.
    """
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = {}

    n_samples = 4000
    env = TabularSepsisEnv(
        cell=2, n_envs=n_samples, alpha_conf=4.0, device="cpu",
        confounding_profile="smooth",
    )
    batch = _make_confounded_batch(env, n_samples=n_samples, seed=0)

    naive = OfflineBanditNaive(
        obs_dim=env.obs_shape[0], n_actions=N_ACTIONS,
        prior_count=0.0, prior_mean=0.0,
    )
    ipw = OfflineBanditIPW(
        obs_dim=env.obs_shape[0], n_actions=N_ACTIONS,
        prior_count=0.0, prior_mean=0.0, ipw_clip=10.0,
    )
    naive.update(batch)
    ipw.update(batch)
    naive_argmax = int(naive.mu_hat.argmax().item())
    ipw_argmax = int(ipw.mu_hat.argmax().item())
    assert naive_argmax == ipw_argmax, (
        f"smooth profile should produce no argmax flip — naive and IPW "
        f"both pick the same arm.  Got naive={naive_argmax}, "
        f"ipw={ipw_argmax}.  If this fires, the smooth reward shape "
        f"may have been altered (v25 invariant: smooth is byte-"
        f"identical to pre-v25), or the smooth reward distribution "
        f"now admits Simpson's-paradox confounding under the same "
        f"behaviour fixture."
    )
