# Experiment Plan

This document maps the experiment tiers to paper figures and empirical claims.
It serves as the authoritative guide for reproducing results.

## Tiers Overview

| Tier | Config | Est. wall-clock | Purpose |
|------|--------|----------------|---------|
| 0 | `smoke_quick` | < 2 min (CPU) | Import/shape sanity check |
| 0b | `paper_bandit` | ~30 min (CPU) | Bandit family (UCB/UCB±/RCT) |
| 1 | `smoke` | 30–60 min (GPU) | Algorithm correctness |
| 1.5 | `horizon_sweep` | ~30 min (CPU) | Δ_TV is horizon-aware |
| 1.7 | `smoke_v7` | ~10 min (CPU) | v7 acceptance smoke (populates non_id stratum) |
| 1.8 | `smoke_v8` | ~25 min (CPU) | v8 acceptance smoke (perturbations on, wide bias_strengths) |
| 2 | `paper` | ~24 h (A100) | Headline figures |
| 3 | `bias_sweep` | ~8 h (A100) | Confounding sweep figures |
| 4 | `sample_sweep` | ~4 h (A100) | Sample-size figures |

Tier 0b (`paper_bandit`) and Tier 1 (`paper`/`smoke`) are independent and can
run in either order: the bandit family does not depend on the off-policy
infrastructure exercised by Tier 1.

---

## Tier 0 — Smoke Quick

**Config**: `configs/smoke_quick.yaml`
**Cells**: 1, 2 · **Envs**: tabular-sepsis-v0 · **Algos**: dqn, ppo · **Behaviours**: uniform · **Seeds**: 1 · **Frames**: 500

**Purpose**: Catches import errors, shape mismatches, and broken wiring before spending GPU hours.
Run this before every other tier.

```bash
python scripts/run_full_matrix.py --config smoke_quick --quick
```

**Pass criteria**: All runs exit with code 0; `train.csv` and `eval.csv` files are non-empty.

---

## Tier 1 — Smoke (Algorithm Correctness)

**Config**: `configs/smoke.yaml`
**Cells**: 1–4 · **Envs**: tabular-sepsis-v0, continuous-ward-v0 · **Algos**: all · **Seeds**: 1 · **Frames**: 5 000

**Purpose**: Verify all algorithm × environment × cell combinations run to completion,
that oracle return > random return, and that CSVs contain the expected columns.

```bash
python scripts/run_full_matrix.py --config smoke --device cuda --n-workers 8 --n-gpus 2
```

**Pass criteria**:
- `eval_oracle_return_mean > eval_return_mean` for on-policy algos (sanity)
- No NaN in `delta_tv` or `ece` columns
- `eval_perturbed.csv` exists for tabular runs

---

## Tier 1.5 — Horizon impact

**Config**: `configs/horizon_sweep.yaml`
**Cells**: 1–4 · **Algos**: dqn, cql, a2c · **Behaviours**: uniform, reward_aligned
**Seeds**: 3 · **horizon_sweep**: [5, 10, 20, 40, 80]
**total_frames_per_episode**: 200 (so episodes-per-run is roughly constant)

```bash
python scripts/run_full_matrix.py --config horizon_sweep --n-workers 4
```

**Claim**: in id cells, return-quality and Δ_TV are robust to episode length;
in partial_id cells, both degrade with horizon, with Δ_TV widening
systematically. This isolates the partial-observability contribution to the
metric from confounding contributions, and demonstrates that Δ_TV is a
horizon-aware quantity (not a property of a single (s, a) pair).

**Figure**: `horizon_panel.pdf` (left: normalised eval/oracle return vs
horizon; right: Δ_TV vs horizon; one line per cell).

---

## Tier 1.7 — v7 Acceptance Smoke

**Config**: `configs/smoke_v7.yaml`
**Cells**: 1–4 · **Algos**: dqn, cql · **Behaviours**: uniform, reward_aligned
**alpha_conf_sweep**: [0.0, 2.0] · **bias_strengths**: [0.5, 1.0]

```bash
python scripts/run_full_matrix.py --config smoke_v7 --n-workers 4
```

**Pass criteria**:
- `summary.json` shows `n_non_id > 0` (non_id stratum populated).
- `bound_width_mean` is finite for at least some configurations.
- `min_propensity` varies with behaviour in cells with `pi_b_known=False`.
- `propensity_calibration_ece` is finite-non-zero in cells 2/4.

---

## Tier 1.8 — v8 Acceptance Smoke

**Config**: `configs/smoke_v8.yaml`
**Cells**: 1–4 · **Algos**: dqn, cql · **Behaviours**: uniform, reward_aligned, reward_misaligned
**alpha_conf_sweep**: [0.0, 2.0] · **bias_strengths**: [0.0, 0.25, 0.5, 0.75, 1.0]
**eval_perturbations**: true (so D_env_KS / D_bisim populate the regression)

```bash
python scripts/run_full_matrix.py --config smoke_v8 --n-workers 4
```

Adds three things on top of `smoke_v7`:
1. Wide ``bias_strengths`` so the natural-bound width moves away from the
   uniform-policy floor of ``1 - 1/n_actions = 0.875`` and so the
   ``bias_sweep`` figure captures the predicted monotone trend.
2. ``reward_misaligned`` behaviour to generate non-uniform action
   distributions and meaningful per-arm `bound_lower_a*` / `bound_upper_a*`
   variation.
3. ``eval_perturbations: true`` so ``D_env_KS``, ``D_bisim``, and
   ``target_support_overlap`` are populated and the headline regression
   runs in *generalisation* mode rather than *in-domain* mode.

**Pass criteria** (in addition to v7's):
- ``bound_width_mean`` standard deviation across runs > 0.05 (was constant
  at 0.875 in v7).
- ``D_env_KS`` and ``D_bisim`` columns finite for the majority of perturbed
  rows.
- Headline regression mode = ``generalisation`` (recorded in
  ``headline_regression_meta.json``).

---

## Tier 2 — Paper Run (Headline Figures)

**Config**: `configs/paper.yaml`
**Cells**: 1–4 · **Envs**: both · **Algos**: ppo, a2c, dqn, ddpg, cql, confounded_dqn
**Behaviours**: uniform, reward_aligned, reward_misaligned, information, curiosity
**Seeds**: 10 (0–9) · **alpha_conf_sweep**: [0.0, 2.0] · **Frames**: 200 000

```bash
# Local (multi-GPU)
python scripts/run_full_matrix.py --config paper --device cuda --n-workers 8 --n-gpus 2

# SLURM array
python scripts/run_full_matrix_slurm.py --config paper --seeds 0 1 2 3 4 5 6 7 8 9 \
    > commands.txt
sbatch --array=1-$(wc -l < commands.txt) slurm_array.sh
```

### Figure–Claim Mapping

| Figure | Script | Claim | Source columns |
|--------|--------|-------|----------------|
| Fig 1: Learning curves | `make_all_figures.py::learning_curves` | On-policy algos (ppo/a2c) converge in cells 1–3; fail in cell 4 when non-identified | `train_return_mean` |
| Fig 2: Identifiability panel | `make_all_figures.py::identifiability_panel` | Final eval return stratified by `id_status` shows id > partial_id > non_id ordering | `eval_return_mean`, `id_status` |
| Fig 3: Headline regression | `make_all_figures.py::headline_regression` | G ≈ f(Δ_TV, D_env_KS, bound_width); pooled R² ∈ [0.4, 0.7]; id-stratum R² ∈ [0.6, 0.9] | `headline_stratified_r2.csv` |
| Fig 4: Perturbation curves | `make_all_figures.py::gap_curves` | Perturbed return degrades monotonically as eps_T, eps_R increase along TABULAR_DIAGONAL | `eval_perturbed_return_mean`, `eps_T`, `eps_R` |
| Fig 5: Calibration | `make_all_figures.py::calibration` | ECE < 0.10 for identified cells; ECE diverges in cell 4 | `ece` |
| Fig 6: Dual-policy Δ_TV | `make_all_figures.py::dual_tv` | delta_tv_beh > delta_tv (learned policy selects less-confounded actions over training) | `delta_tv`, `delta_tv_beh` |

---

## Tier 3 — Bias Sweep

**Config**: `configs/bias_sweep.yaml`
**Algos**: dqn, cql, bound_cql · **alpha_conf_sweep**: [0.0, 0.5, 1.0, 1.5, 2.0] · **Seeds**: 5 · **eval_perturbations**: off

```bash
python scripts/run_full_matrix.py --config bias_sweep --n-workers 4
```

### Figure–Claim Mapping

| Figure | Claim | Source columns |
|--------|-------|----------------|
| Fig 7: Bias sweep | As alpha_conf increases, bound_cql degrades less than cql (causal bounds absorb confounding) | `eval_return_mean` vs `alpha_conf` |
| Fig 8: n_informative_bounds | bound_cql with explicit bounds shows n_arms_with_informative_bound > 0 only when bounds are passed | `n_arms_with_informative_bound` |

---

## Tier 4 — Sample Sweep

**Config**: `configs/sample_sweep.yaml`
**Algos**: cql, dqn · **alpha_conf_sweep**: [0.0, 2.0] · **Seeds**: 3
**offline_transitions_sweep**: [500, 2000, 8000, 32000] · **eval_perturbations**: off

```bash
python scripts/run_full_matrix.py --config sample_sweep --n-workers 4
```

### Figure–Claim Mapping

| Figure | Claim | Source columns |
|--------|-------|----------------|
| Fig 9: Sample floor | CQL performance plateaus below ~5 000 offline transitions; bound_cql plateaus later | `eval_return_mean` vs `offline_transitions` |
| Fig 10: Interaction | Identifiability floor is higher at alpha_conf=2.0 than alpha_conf=0.0 for non-identified cells | `eval_return_mean`, `alpha_conf` |

---

## Quality Gates

Before submitting paper figures, the following assertions must hold
(automated in `tests/figures/test_quality_gates.py`):

1. **Tier 0 completion**: All smoke_quick runs exit 0
2. **Oracle dominance**: eval_oracle_return_mean > eval_return_mean + 1.0 (tabular cells 1–3)
3. **Identifiability ordering**: mean(id) > mean(partial_id) > mean(non_id)
4. **Headline R² bounds**: pooled R² ∈ [0.4, 0.7]; id-stratum R² ∈ [0.6, 0.9]; non_id-stratum R² < 0.3
5. **Perturbation monotonicity**: perturbed return non-increasing along TABULAR_DIAGONAL
6. **KS sensitivity**: D_env_KS > 0.05 at max perturbation (eps_T=0.5, eps_R=0.3)
7. **ECE sanity**: ECE < 0.30 for all identified cells
8. **Tier 1.5 horizon ratio**: at horizon=80, mean Δ_TV in partial_id cells / mean Δ_TV in id cells ≥ 1.5
9. **v7 acceptance**: in `smoke_v7.yaml`'s `summary.json`, `n_non_id > 0`
