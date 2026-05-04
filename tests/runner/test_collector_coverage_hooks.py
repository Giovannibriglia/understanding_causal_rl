"""Phase 1.2 acceptance: ``collect_offline_dataset`` honours coverage hooks.

* ``forbidden_actions`` blocks the listed action indices in the buffer
  and reassigns to a uniform draw over the allowed subset.
* ``n_action_restriction_steps`` gates the restriction to only the
  first ``k`` collected transitions.

The acceptance contract is that the *resulting buffer* contains the
expected action distribution, so that downstream propensity / coverage
diagnostics see a coverage-broken regime.
"""

from __future__ import annotations

from causal_rl.behaviour.uniform import UniformExplorer
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.runner.collector import collect_offline_dataset


def _collect(
    *,
    forbidden_actions: list[int] | None,
    n_action_restriction_steps: int | None,
    n_transitions: int = 200,
) -> tuple[set[int], int]:
    env = TabularSepsisEnv(cell=1, n_envs=8, device="cpu")
    behaviour = UniformExplorer(n_actions=8)
    buf = collect_offline_dataset(
        env=env,
        behaviour_policy=behaviour,
        n_transitions=n_transitions,
        expose_pi_b=True,
        expose_latent=False,
        seed=0,
        forbidden_actions=forbidden_actions,
        n_action_restriction_steps=n_action_restriction_steps,
    )
    used = set(int(a) for a in buf.action[: buf.size].view(-1).tolist())
    return used, int(buf.size)


def test_forbidden_actions_excluded_from_buffer() -> None:
    used, _ = _collect(forbidden_actions=[0, 1], n_action_restriction_steps=None)
    assert 0 not in used, f"action 0 should be forbidden; got buffer with actions {used}"
    assert 1 not in used, f"action 1 should be forbidden; got buffer with actions {used}"
    assert used.issubset({2, 3, 4, 5, 6, 7})
    # Sanity: at least some actions in the allowed subset *are* present.
    assert len(used) >= 1


def test_n_action_restriction_steps_gate_relaxes_restriction() -> None:
    """When restriction lifts after k steps, later transitions should
    contain forbidden actions again."""
    forbidden = [0, 1]
    n_total = 400
    restriction_steps = 80
    env = TabularSepsisEnv(cell=1, n_envs=8, device="cpu")
    behaviour = UniformExplorer(n_actions=8)
    buf = collect_offline_dataset(
        env=env,
        behaviour_policy=behaviour,
        n_transitions=n_total,
        expose_pi_b=True,
        expose_latent=False,
        seed=0,
        forbidden_actions=forbidden,
        n_action_restriction_steps=restriction_steps,
    )
    early = set(
        int(a)
        for a in buf.action[: restriction_steps].view(-1).tolist()
    )
    late = set(
        int(a)
        for a in buf.action[restriction_steps : buf.size].view(-1).tolist()
    )
    assert not (early & set(forbidden)), (
        f"forbidden actions {forbidden} appeared in restricted phase: {early}"
    )
    assert late & set(forbidden), (
        f"forbidden actions should reappear after restriction lifts; "
        f"late phase saw {late}"
    )


def test_no_forbidden_is_no_op() -> None:
    used, n = _collect(forbidden_actions=None, n_action_restriction_steps=None)
    assert n > 0
    # Without restriction every action should be reachable across 200 draws
    # over 8 envs with the uniform behaviour.
    assert len(used) >= 5
