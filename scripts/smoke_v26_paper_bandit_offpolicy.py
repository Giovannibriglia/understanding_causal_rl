"""v26.1 production smoke: 1 cell × 1 alpha × 1 seed × 2 algos via the
actual ``run_full_matrix`` → ``run_single`` → ``BenchmarkRunner`` path
to exercise YAML loading + ``_make_behaviour_with_bias`` plumbing +
both off-policy bandit algos before kicking off the 3h sweep.

Usage::

    python scripts/smoke_v26_paper_bandit_offpolicy.py

Prints a per-algo summary including buffer Z-ratio and ``mu_hat``;
fails loudly if any check doesn't meet the v26.1 pass conditions.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.config.schemas import MatrixConfig  # noqa: E402
from causal_rl.envs.tabular_sepsis import N_ACTIONS, OBS_STATES  # noqa: E402
from causal_rl.runner import runner as runner_mod  # noqa: E402
from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig  # noqa: E402


def main() -> None:
    yaml_path = ROOT / "configs" / "paper_bandit_offpolicy.yaml"
    matrix = MatrixConfig.model_validate(yaml.safe_load(yaml_path.read_text()))

    out_root = ROOT / "results" / f"v26_smoke_{int(time.time())}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"smoke output root: {out_root}")
    print(f"loaded matrix: env={matrix.envs[0]}, behaviours={matrix.behaviours}")

    cell = 2
    alpha = 4.0
    seed = 0
    behaviour = "reward_aligned_z"  # v26 picks this from YAML
    if behaviour not in matrix.behaviours:
        msg = f"YAML must list 'reward_aligned_z'; got {matrix.behaviours}"
        raise RuntimeError(msg)

    horizon = matrix.env_horizons[matrix.envs[0]]
    # Reduce total_frames (online-rollout count) to keep wall short.
    # Keep offline_transitions / offline_updates / n_eval_episodes /
    # n_bootstrap / n_samples_gap at production values so the
    # buffer-construction and per-checkpoint code paths are
    # exercised at production scale.
    smoke_total_frames = 4000

    summaries: dict[str, dict] = {}
    for algo_name in ("offline_bandit_naive", "offline_bandit_ipw"):
        out_dir = out_root / f"{algo_name}_cell{cell}_alpha{alpha}_seed{seed}"
        cfg = RunnerConfig(
            cell=cell,
            env_name=matrix.envs[0],
            algorithm=algo_name,
            behaviour=behaviour,
            seed=seed,
            total_frames=smoke_total_frames,
            n_checkpoints_train=matrix.n_checkpoints_train,
            n_checkpoints_eval=matrix.n_checkpoints_eval,
            horizon=horizon,
            rollout_horizon=horizon,
            output_dir=out_dir,
            device="cpu",
            n_envs=matrix.n_envs,
            batch_size=matrix.n_envs,
            offline_transitions=matrix.offline_transitions,
            offline_updates=matrix.offline_updates,
            alpha_conf=alpha,
            bias_strength=1.0,
            n_eval_episodes=matrix.n_eval_episodes,
            n_bootstrap=matrix.n_bootstrap,
            n_samples_gap=matrix.n_samples_gap,
            eval_perturbations=matrix.eval_perturbations,
        )
        print(f"\n=== running {algo_name} ===")
        t0 = time.perf_counter()
        runner = BenchmarkRunner(cfg)
        # Confirm v26 plumbing reached the policy.
        bp = runner.behaviour_policy
        assert getattr(bp, "alpha_conf", None) == alpha, (
            f"alpha_conf didn't reach policy: bp.alpha_conf={getattr(bp, 'alpha_conf', None)}"
        )
        assert getattr(bp, "latent_source", None) == "latent_Z", (
            f"latent_source not 'latent_Z': {getattr(bp, 'latent_source', None)}"
        )

        # Spy on collect_offline_dataset to capture the buffer for
        # post-hoc Z-ratio analysis.  The runner doesn't retain the
        # buffer on self, so this is the cleanest way to inspect it
        # without changing production code.
        captured_buffer: dict[str, object] = {}
        real_collect = runner_mod.collect_offline_dataset

        def _spy(*args, **kwargs):
            buf = real_collect(*args, **kwargs)
            captured_buffer["buf"] = buf
            return buf

        runner_mod.collect_offline_dataset = _spy
        try:
            runner.run()
        finally:
            runner_mod.collect_offline_dataset = real_collect
        dt = time.perf_counter() - t0

        mu_hat = runner.algo.mu_hat.detach().cpu().numpy().tolist()
        buf = captured_buffer.get("buf")
        if buf is None:
            counts = None
        else:
            z = buf.latent[: buf.size].view(-1).long().clamp(0, 1)
            a = buf.action[: buf.size].view(-1).long()
            counts = np.zeros((2, N_ACTIONS), dtype=int)
            for zi, ai in zip(z.tolist(), a.tolist()):
                counts[zi, ai] += 1

        summaries[algo_name] = {
            "wall": dt,
            "mu_hat": mu_hat,
            "counts": counts,
        }
        print(f"  wall: {dt:.1f}s")
        print(f"  mu_hat: {[round(x, 3) for x in mu_hat]}")
        if counts is not None:
            print(f"  counts[Z=0]: {counts[0].tolist()}")
            print(f"  counts[Z=1]: {counts[1].tolist()}")

    print("\n=== v26.1 smoke verdict ===")
    # Pass condition 1: max per-action Z-ratio >= 5:1 in either buffer.
    max_ratio = 0.0
    max_arm = -1
    for algo_name, s in summaries.items():
        counts = s["counts"]
        if counts is None:
            continue
        for arm in range(N_ACTIONS):
            c0, c1 = int(counts[0, arm]), int(counts[1, arm])
            if c0 + c1 < 50:
                continue
            r = max(c0 / max(c1, 1), c1 / max(c0, 1))
            if r > max_ratio:
                max_ratio = r
                max_arm = arm
    print(f"max Z-ratio across buffers: {max_ratio:.2f} (arm {max_arm})")
    if max_ratio < 5.0:
        print("FAIL: confounding ratio below 5:1 — production path not "
              "producing confounded buffers; investigate before sweep.")
        sys.exit(1)

    # Pass condition 2: naive vs ipw mu_hat differ by more than noise floor.
    naive_mu = np.array(summaries["offline_bandit_naive"]["mu_hat"])
    ipw_mu = np.array(summaries["offline_bandit_ipw"]["mu_hat"])
    diff = float(np.abs(naive_mu - ipw_mu).max())
    print(f"max |naive_mu - ipw_mu|: {diff:.4f}")
    # Bernoulli noise floor on n=4000 samples per arm: σ ≈ √(1/4000) ≈ 0.016.
    # With both algos using the same buffer, divergence < 0.05 would
    # suggest the IPW reweighting isn't biting.
    if diff < 0.05:
        print("FAIL: naive and ipw mu_hat differ by < 0.05 — IPW "
              "reweighting not biting on the production buffer.")
        sys.exit(1)

    print("PASS: confounding ratio adequate AND algos diverge.  Sweep "
          "can proceed.")


if __name__ == "__main__":
    main()
