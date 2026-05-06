"""v17: ``n_bootstrap`` and ``n_samples_gap`` propagate from the YAML
matrix config through the per-run ``run_single.py`` invocation.

Mirrors the existing ``n_eval_episodes`` wiring exactly.  The test
covers both layers:

* ``MatrixConfig`` (and ``RunConfigModel``) accept the two new keys
  with ``Field(..., ge=...)`` validation.
* ``scripts/run_full_matrix.py``'s command-builder appends
  ``--n-bootstrap`` and ``--n-samples-gap`` to the per-run command
  when the matrix overrides the defaults.

Both new fields default to 200 (matching the long-standing
``RunnerConfig`` defaults), so YAMLs without them behave exactly as
before.
"""

from __future__ import annotations

from pathlib import Path

from causal_rl.config.schemas import MatrixConfig, RunConfigModel


def test_run_config_model_accepts_diagnostic_budget_overrides() -> None:
    """``RunConfigModel`` accepts non-default values for the two new
    fields and validates the ``ge`` constraints (n_bootstrap >= 0,
    n_samples_gap >= 1)."""
    cfg = RunConfigModel(
        cell=1,
        env="tabular-sepsis-v0",
        algorithm="dqn",
        behaviour="uniform",
        seed=0,
        total_frames=2000,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output="/tmp/x",
        n_bootstrap=37,
        n_samples_gap=53,
    )
    assert cfg.n_bootstrap == 37
    assert cfg.n_samples_gap == 53


def test_run_config_model_defaults_are_200() -> None:
    """Backward-compatibility: configs that don't set the new keys
    fall back to the legacy ``RunnerConfig`` defaults."""
    cfg = RunConfigModel(
        cell=1,
        env="tabular-sepsis-v0",
        algorithm="dqn",
        behaviour="uniform",
        seed=0,
        total_frames=2000,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output="/tmp/x",
    )
    assert cfg.n_bootstrap == 200
    assert cfg.n_samples_gap == 200


def test_matrix_config_accepts_diagnostic_budget_overrides() -> None:
    """``MatrixConfig`` accepts the two new keys with the same
    validation pattern as ``RunConfigModel``."""
    cfg = MatrixConfig(
        cells=[1, 2, 3, 4],
        envs=["tabular-sepsis-v0"],
        algorithms=["dqn"],
        behaviours=["uniform"],
        seeds=[0],
        n_bootstrap=37,
        n_samples_gap=53,
    )
    assert cfg.n_bootstrap == 37
    assert cfg.n_samples_gap == 53


def test_matrix_config_defaults_are_200() -> None:
    cfg = MatrixConfig(
        cells=[1, 2, 3, 4],
        envs=["tabular-sepsis-v0"],
        algorithms=["dqn"],
        behaviours=["uniform"],
        seeds=[0],
    )
    assert cfg.n_bootstrap == 200
    assert cfg.n_samples_gap == 200


def test_run_full_matrix_cmd_includes_diagnostic_budget_flags(
    tmp_path: Path,
) -> None:
    """``scripts/run_full_matrix.py``'s ``_build_cmd`` appends
    ``--n-bootstrap`` and ``--n-samples-gap`` to the per-run command.

    Mirrors the existing ``--n-eval-episodes`` wiring.  We construct
    a tiny ``MatrixConfig`` with non-default values, call the helper,
    and assert the flags appear in the resulting argv list.
    """
    import sys

    project_root = Path(__file__).resolve().parents[2]
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_full_matrix import _build_cmd  # noqa: E402

    matrix = MatrixConfig(
        cells=[1],
        envs=["tabular-sepsis-v0"],
        algorithms=["dqn"],
        behaviours=["uniform"],
        seeds=[0],
        env_horizons={"tabular-sepsis-v0": 20},
        n_bootstrap=37,
        n_samples_gap=53,
    )
    cmd = _build_cmd(
        cell=1,
        env="tabular-sepsis-v0",
        algo="dqn",
        beh="uniform",
        seed=0,
        alpha_conf=0.0,
        bias_strength=1.0,
        horizon_override=None,
        offline_transitions_override=None,
        forbidden_actions=None,
        n_action_restriction_steps=None,
        matrix=matrix,
        run_dir=tmp_path / "run",
        device=None,
        quick=False,
    )
    # Flags appear adjacently as ["--n-bootstrap", "37"] etc.
    def _flag_index(flag: str) -> int:
        for i, token in enumerate(cmd):
            if token == flag:
                return i
        raise AssertionError(f"flag {flag!r} not in cmd: {cmd}")

    nb_idx = _flag_index("--n-bootstrap")
    assert cmd[nb_idx + 1] == "37", (
        f"--n-bootstrap value: expected '37', got {cmd[nb_idx + 1]!r}"
    )
    ng_idx = _flag_index("--n-samples-gap")
    assert cmd[ng_idx + 1] == "53", (
        f"--n-samples-gap value: expected '53', got {cmd[ng_idx + 1]!r}"
    )
