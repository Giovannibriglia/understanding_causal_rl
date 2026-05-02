# Experiment Plan

This document maps the experiment tiers to paper figures and empirical claims.
It serves as the authoritative guide for reproducing results.

## Tiers Overview

| Tier | Config | Est. wall-clock | Purpose |
|------|--------|----------------|---------|
| 0 | `smoke_quick` | < 2 min (CPU) | Import/shape sanity check |
| 1 | `smoke` | 30–60 min (GPU) | Algorithm correctness |
| 2 | `paper` | ~24 h (A100) | Headline figures |
| 3 | `bias_sweep` | ~8 h (A100) | Confounding sweep figures |
| 4 | `sample_sweep` | ~4 h (A100) | Sample-size figures |

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
python scripts/run_full_matrix.py --config smoke
```

**Pass criteria**:
- `eval_oracle_return_mean > eval_return_mean` for on-policy algos (sanity)
- No NaN in `delta_tv` or `ece` columns
- `eval_perturbed.csv` exists for tabular runs

---

## Tier 2 — Paper Run (Headline Figures)

**Config**: `configs/paper.yaml`
**Cells**: 1–4 · **Envs**: both · **Algos**: ppo, a2c, dqn, ddpg, cql, confounded_dqn
**Behaviours**: uniform, reward_aligned, reward_misaligned, information, curiosity
**Seeds**: 10 (0–9) · **alpha_conf_sweep**: [0.0, 2.0] · **Frames**: 200 000

```bash
# Local (multi-GPU)
python scripts/run_full_matrix.py --config paper --n-workers 8 --n-gpus 2

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
**Algos**: cql, bound_cql · **alpha_conf_sweep**: [0.0, 2.0] · **Seeds**: 5 · **eval_perturbations**: off

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
