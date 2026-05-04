from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from causal_rl.algos.registry import ALGOS
from causal_rl.algos.registry import make as make_algo
from causal_rl.behaviour.registry import BEHAVIOURS
from causal_rl.behaviour.registry import make as make_behaviour
from causal_rl.envs.base import CausalEnv
from causal_rl.envs.bandit_view import BanditView
from causal_rl.envs.cell_config import CELL_CONFIGS
from causal_rl.envs.continuous_ward import ContinuousWardEnv
from causal_rl.envs.perturbations import (
    CONTINUOUS_DIAGONAL,
    TABULAR_DIAGONAL,
    PerturbationSpec,
    perturb_tabular_rewards,
    perturb_tabular_transition,
)
from causal_rl.envs.registry import ENVS
from causal_rl.envs.registry import make as make_env
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.identification.bounds import natural_bounds
from causal_rl.identification.id_oracle import get_id_status
from causal_rl.metrics.backdoor import backdoor_residual_summary
from causal_rl.metrics.calibration import expected_calibration_error
from causal_rl.metrics.coverage import (
    PropensityModel,
    effective_sample_size_ratio,
    expected_calibration_error as expected_calibration_error_2d,
    fit_propensity_model,
    min_propensity,
    pi_b_recovery_kl,
    support_overlap,
    tail_mass_top_q,
    target_support_overlap,
)
from causal_rl.metrics.info_bottleneck import conditional_mi_lower_bound
from causal_rl.metrics.d_env import d_env_ks
from causal_rl.metrics.runner_hooks import compute_gap_metrics
from causal_rl.runner.collector import collect_offline_dataset
from causal_rl.runner.csv_logger import CSVLogger
from causal_rl.runner.scheduling import checkpoint_ticks
from causal_rl.runner.schemas import EVAL_COLUMNS, PERTURBED_COLUMNS, TRAIN_COLUMNS
from causal_rl.utils.device import get_device
from causal_rl.utils.git import get_git_commit
from causal_rl.utils.seeding import set_seed


@dataclass(frozen=True)
class RunnerConfig:
    cell: int
    env_name: str
    algorithm: str
    behaviour: str
    seed: int
    total_frames: int
    n_checkpoints_train: int
    n_checkpoints_eval: int
    output_dir: Path
    horizon: int | None = None
    rollout_horizon: int | None = None
    device: str | None = None
    n_envs: int = 64
    batch_size: int = 64
    offline_transitions: int = 50_000
    offline_updates: int = 2_000
    alpha_conf: float = 0.0
    bias_strength: float = 1.0
    eval_n_envs: int | None = None
    n_bootstrap: int = 200
    n_samples_gap: int = 200
    n_eval_episodes: int = 3
    # Optional smoke-only perturbation-grid trim; None or <=0 ⇒ full grid.
    perturbation_grid_size: int | None = None
    eval_perturbations: bool = True
    # Oracle choice for evaluation: 'auto'|'dp'|'cem'|'grid'|'algo'
    oracle: str = "auto"


class BenchmarkRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.device = get_device(config.device)
        set_seed(config.seed)
        self.cell_cfg = CELL_CONFIGS[config.cell]
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_logger = CSVLogger(self.output_dir / "train.csv", TRAIN_COLUMNS)
        self.eval_logger = CSVLogger(self.output_dir / "eval.csv", EVAL_COLUMNS)
        self.perturbed_logger = CSVLogger(self.output_dir / "eval_perturbed.csv", PERTURBED_COLUMNS)

        self.env = self._make_env_instance()
        self.env_spec = ENVS[config.env_name]
        beh_depends_on_u = BEHAVIOURS[config.behaviour].depends_on_u
        self.id_status = get_id_status(
            config.env_name,
            config.cell,
            alpha_conf=config.alpha_conf,
            beh_depends_on_u=beh_depends_on_u,
        )

        obs_dim = self.env.obs_shape[0]
        if self.env.is_discrete_action:
            n_actions = 8
            algo_kwargs: dict[str, object] = {"obs_dim": obs_dim, "n_actions": n_actions}
            if config.algorithm in {"ucb_plus", "ucb_minus"} and isinstance(self.env, BanditView):
                actions_obs, rewards_obs = self.env.collect_observational_data(
                    n_samples=5000,
                    alpha_conf=config.alpha_conf,
                    seed=config.seed,
                )
                if config.algorithm == "ucb_minus":
                    means: list[float] = []
                    for a in range(n_actions):
                        mask = actions_obs == a
                        means.append(
                            float(rewards_obs[mask].mean().item()) if bool(mask.any()) else 0.5
                        )
                    algo_kwargs["observational_means"] = means
                else:  # ucb_plus
                    bounds = natural_bounds(
                        actions_obs, rewards_obs, n_actions=n_actions, n_bootstrap=200
                    )
                    algo_kwargs["lower_bounds"] = [bounds[a].lower for a in range(n_actions)]
                    algo_kwargs["upper_bounds"] = [bounds[a].upper for a in range(n_actions)]
            self.algo = make_algo(config.algorithm, **algo_kwargs)
            self.behaviour_policy = self._make_behaviour_with_bias(
                config.behaviour,
                n_actions=n_actions,
                requires_latent=beh_depends_on_u,
            )
        else:
            act_dim = self.env.act_shape[0]
            self.algo = make_algo(config.algorithm, obs_dim=obs_dim, act_dim=act_dim)
            self.behaviour_policy = self._make_behaviour_with_bias(
                config.behaviour,
                act_dim=act_dim,
                requires_latent=beh_depends_on_u,
            )

        self._move_algo_to_device()
        # Sanity: warn if bootstrap memory would be excessive.
        _mem = self.config.n_envs * self.config.n_bootstrap * self.config.n_samples_gap
        if _mem > 5e8:
            import warnings

            self._effective_n_bootstrap = max(
                1, int(5e8 / (self.config.n_envs * self.config.n_samples_gap))
            )
            warnings.warn(
                f"n_envs={self.config.n_envs} * n_bootstrap={self.config.n_bootstrap} * "
                f"n_samples_gap={self.config.n_samples_gap} = {_mem:.0f} > 5e8. "
                f"Clamping n_bootstrap to {self._effective_n_bootstrap}.",
                stacklevel=2,
            )
        else:
            self._effective_n_bootstrap = self.config.n_bootstrap
        if self.config.total_frames <= 0:
            msg = "total_frames must be > 0."
            raise ValueError(msg)
        if self.env.horizon <= 0:
            msg = "environment horizon must be > 0."
            raise ValueError(msg)
        if self.config.total_frames % self.env.horizon != 0:
            msg = (
                f"total_frames={self.config.total_frames} must be divisible by "
                f"episode_horizon_length={self.env.horizon}."
            )
            raise ValueError(msg)
        self._write_meta()
        self._oracle_policy_by_steps_left: torch.Tensor | None = None
        # Cache for the latest gap metric so ConfoundedDQN can use it.
        self._last_gap_metrics: dict[str, float] = {"delta_tv": 0.0}
        self._coverage_metrics: dict[str, float] = {}
        self._bound_metrics: dict[str, float] = {}
        self._propensity_calibration_ece: float = float("nan")
        self._pi_b_recovery_kl: float = float("nan")
        self._propensity_model: PropensityModel | None = None
        self._extra_metrics: dict[str, float] = {
            "backdoor_residual_mean": float("nan"),
            "backdoor_residual_max": float("nan"),
            "target_support_overlap": float("nan"),
            "cond_mi_r_z_given_sa": float("nan"),
        }

    def _make_behaviour_with_bias(self, name: str, **kwargs: object) -> object:
        # Pass bias_strength only when the behaviour constructor accepts it.
        try:
            return make_behaviour(name, bias_strength=self.config.bias_strength, **kwargs)
        except TypeError:
            return make_behaviour(name, **kwargs)

    def _move_algo_to_device(self) -> None:
        for value in vars(self.algo).values():
            if isinstance(value, nn.Module):
                value.to(self.device)

    @property
    def _eval_n_envs(self) -> int:
        return (
            self.config.eval_n_envs if self.config.eval_n_envs is not None else self.config.n_envs
        )

    def _make_env_instance(self) -> CausalEnv:
        return self._make_env_with_n(self.config.n_envs)

    def _make_eval_env_instance(self) -> CausalEnv:
        return self._make_env_with_n(self._eval_n_envs)

    def _make_env_with_n(self, n_envs: int) -> CausalEnv:
        if self.config.horizon is None:
            return make_env(
                self.config.env_name,
                cell=self.config.cell,
                n_envs=n_envs,
                alpha_conf=self.config.alpha_conf,
                device=str(self.device),
            )
        return make_env(
            self.config.env_name,
            cell=self.config.cell,
            n_envs=n_envs,
            alpha_conf=self.config.alpha_conf,
            horizon=self.config.horizon,
            device=str(self.device),
        )

    def _write_meta(self) -> None:
        # total_frames is the number of batched ticks (each tick = n_envs parallel steps).
        # total_transitions is the actual number of (s,a,s',r) tuples collected.
        meta = {
            "seed": self.config.seed,
            "cell": self.config.cell,
            "env": self.config.env_name,
            "algorithm": self.config.algorithm,
            "behaviour": self.config.behaviour,
            "device": str(self.device),
            "horizon": self.env.horizon,
            "rollout_horizon": self.config.rollout_horizon or self.env.horizon,
            "n_envs": self.config.n_envs,
            "total_frames": self.config.total_frames,
            "total_transitions": self.config.total_frames * self.config.n_envs,
            "n_checkpoints_train": self.config.n_checkpoints_train,
            "n_checkpoints_eval": self.config.n_checkpoints_eval,
            "id_status": self.id_status,
            "cli": " ".join(sys.argv),
            "git_commit": get_git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        }
        with (self.output_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _log_train(
        self, step: int, episode: int, returns: torch.Tensor, metrics: dict[str, float]
    ) -> None:
        row: dict[str, object] = {
            "step": step,
            "episode": episode,
            "wall_time": time.time(),
            "cell": self.config.cell,
            "env_name": self.config.env_name,
            "env_fidelity": self.env_spec.fidelity,
            "algorithm": self.config.algorithm,
            "behaviour_policy": self.config.behaviour,
            "seed": self.config.seed,
            "device": str(self.device),
            "train_return_mean": float(returns.mean().item()),
            "train_return_std": float(returns.std().item()),
            "policy_loss": metrics.get("policy_loss", 0.0),
            "value_loss": metrics.get("value_loss", 0.0),
            "td_loss": metrics.get("td_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "kl": metrics.get("kl", 0.0),
            "lr": metrics.get("lr", 0.0),
            "grad_norm": metrics.get("grad_norm", 0.0),
            "id_status": self.id_status,
            "alpha_conf": self.config.alpha_conf,
            "bias_strength": getattr(self.behaviour_policy, "bias_strength", 1.0),
            "expose_z": self.cell_cfg.expose_z,
            "pi_b_known": self.cell_cfg.pi_b_known,
            "beh_depends_on_u": self.behaviour_policy.depends_on_u,
        }
        gap_keys = [
            "delta_tv",
            "delta_tv_ci_lo",
            "delta_tv_ci_hi",
            "delta_kl",
            "delta_chi2",
            "delta_sup",
            "delta_mmd2",
            "delta_ks",
        ]
        row.update({k: metrics.get(k, 0.0) for k in gap_keys})
        # Distinguish marginal (before IPW) from conditional (after IPW) gaps.
        row["delta_tv_marginal"] = metrics.get("delta_tv_marginal", 0.0)
        row["delta_tv_conditional"] = metrics.get(
            "delta_tv_conditional", metrics.get("delta_tv", 0.0)
        )
        row["min_propensity"] = self._coverage_metrics.get("min_propensity", float("nan"))
        row["ess_ratio"] = self._coverage_metrics.get("ess_ratio", float("nan"))
        row["overlap_at_1e-2"] = self._coverage_metrics.get("overlap_at_1e-2", float("nan"))
        row["tail_mass_top10pct"] = self._coverage_metrics.get("tail_mass_top10pct", float("nan"))
        row["propensity_calibration_ece"] = self._propensity_calibration_ece
        row["pi_b_recovery_kl"] = self._pi_b_recovery_kl
        row["bound_width_mean"] = self._bound_metrics.get("bound_width_mean", float("nan"))
        row["bound_width_max"] = self._bound_metrics.get("bound_width_max", float("nan"))
        row["n_arms_with_informative_bound"] = self._bound_metrics.get(
            "n_arms_with_informative_bound", self.algo.n_informative_bounds()
        )
        for a in range(8):
            row[f"bound_lower_a{a}"] = self._bound_metrics.get(
                f"bound_lower_a{a}", float("nan")
            )
            row[f"bound_upper_a{a}"] = self._bound_metrics.get(
                f"bound_upper_a{a}", float("nan")
            )
            row[f"obs_arm_count_a{a}"] = self._bound_metrics.get(
                f"obs_arm_count_a{a}", 0
            )
        for k in (
            "backdoor_residual_mean",
            "backdoor_residual_max",
            "target_support_overlap",
            "cond_mi_r_z_given_sa",
        ):
            row[k] = self._extra_metrics.get(k, float("nan"))
        self.train_logger.write(row)

    def _log_eval(
        self,
        step: int,
        episode: int,
        eval_returns: torch.Tensor,
        oracle_returns: torch.Tensor,
        metrics: dict[str, float],
    ) -> None:
        row: dict[str, object] = {
            "step": step,
            "episode": episode,
            "wall_time": time.time(),
            "cell": self.config.cell,
            "env_name": self.config.env_name,
            "env_fidelity": self.env_spec.fidelity,
            "algorithm": self.config.algorithm,
            "behaviour_policy": self.config.behaviour,
            "seed": self.config.seed,
            "eval_return_mean": float(eval_returns.mean().item()),
            "eval_return_std": float(eval_returns.std().item()),
            "delta_tv": metrics.get("delta_tv", 0.0),
            "delta_tv_ci_lo": metrics.get("delta_tv_ci_lo", 0.0),
            "delta_tv_ci_hi": metrics.get("delta_tv_ci_hi", 0.0),
            "delta_kl": metrics.get("delta_kl", 0.0),
            "delta_chi2": metrics.get("delta_chi2", 0.0),
            "delta_sup": metrics.get("delta_sup", 0.0),
            "delta_mmd2": metrics.get("delta_mmd2", 0.0),
            "delta_ks": metrics.get("delta_ks", 0.0),
            "id_status": self.id_status,
            "alpha_conf": self.config.alpha_conf,
            "bias_strength": getattr(self.behaviour_policy, "bias_strength", 1.0),
            "expose_z": self.cell_cfg.expose_z,
            "pi_b_known": self.cell_cfg.pi_b_known,
            "beh_depends_on_u": self.behaviour_policy.depends_on_u,
            "eval_oracle_return_mean": float(oracle_returns.mean().item()),
            "eval_oracle_return_std": float(oracle_returns.std().item()),
            "delta_tv_beh": metrics.get("delta_tv_beh", 0.0),
            "ece": metrics.get("ece", 0.0),
        }
        row["min_propensity"] = self._coverage_metrics.get("min_propensity", float("nan"))
        row["ess_ratio"] = self._coverage_metrics.get("ess_ratio", float("nan"))
        row["overlap_at_1e-2"] = self._coverage_metrics.get("overlap_at_1e-2", float("nan"))
        row["tail_mass_top10pct"] = self._coverage_metrics.get("tail_mass_top10pct", float("nan"))
        row["propensity_calibration_ece"] = self._propensity_calibration_ece
        row["pi_b_recovery_kl"] = self._pi_b_recovery_kl
        row["bound_width_mean"] = self._bound_metrics.get("bound_width_mean", float("nan"))
        row["bound_width_max"] = self._bound_metrics.get("bound_width_max", float("nan"))
        row["n_arms_with_informative_bound"] = self._bound_metrics.get(
            "n_arms_with_informative_bound", float("nan")
        )
        for a in range(8):
            row[f"bound_lower_a{a}"] = self._bound_metrics.get(
                f"bound_lower_a{a}", float("nan")
            )
            row[f"bound_upper_a{a}"] = self._bound_metrics.get(
                f"bound_upper_a{a}", float("nan")
            )
            row[f"obs_arm_count_a{a}"] = self._bound_metrics.get(
                f"obs_arm_count_a{a}", 0
            )
        for k in (
            "backdoor_residual_mean",
            "backdoor_residual_max",
            "target_support_overlap",
            "cond_mi_r_z_given_sa",
        ):
            row[k] = self._extra_metrics.get(k, float("nan"))
        self.eval_logger.write(row)

    def _compute_coverage_metrics(self, log_probs: torch.Tensor) -> None:
        self._coverage_metrics = {
            "min_propensity": float(min_propensity(log_probs).item()),
            "ess_ratio": float(effective_sample_size_ratio(log_probs).item()),
            "overlap_at_1e-2": float(support_overlap(log_probs, tau=1e-2).item()),
            "tail_mass_top10pct": float(tail_mass_top_q(log_probs, q=0.1).item()),
        }

    def _compute_bound_metrics(self, actions: torch.Tensor, rewards: torch.Tensor) -> None:
        # Compute Bareinboim natural-bound metrics for any discrete-action run.
        # Emits per-arm lower/upper/count so downstream figures (bound_width
        # panel) can scatter per-arm points instead of a degenerate per-run
        # mean.  bound_cql and ucb_plus also consume the aggregates.
        n_actions = 8
        nan_per_arm: dict[str, float] = {
            **{f"bound_lower_a{a}": float("nan") for a in range(n_actions)},
            **{f"bound_upper_a{a}": float("nan") for a in range(n_actions)},
            **{f"obs_arm_count_a{a}": 0 for a in range(n_actions)},
        }
        if not self.env.is_discrete_action or actions.numel() == 0:
            self._bound_metrics = {
                "bound_width_mean": float("nan"),
                "bound_width_max": float("nan"),
                "n_arms_with_informative_bound": float("nan"),
                **nan_per_arm,
            }
            return
        acts = actions.view(-1).long()
        rewards_01 = ((rewards.view(-1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
        bounds = natural_bounds(acts, rewards_01, n_actions=n_actions, n_bootstrap=200)
        widths = [r.upper - r.lower for r in bounds.values()]
        # Bareinboim's "informative" criterion: an arm's upper bound lies below
        # the best lower bound across arms (so the arm can be excluded as
        # globally suboptimal).  The previous mu_lower/p_a denominator collapsed
        # in the limit of small p_a and produced an almost-always-zero indicator.
        mu_lower_max = max(r.lower for r in bounds.values())
        n_inf = sum(1 for r in bounds.values() if r.upper < mu_lower_max)
        per_arm: dict[str, float] = {}
        for a in range(n_actions):
            r = bounds[a]
            per_arm[f"bound_lower_a{a}"] = float(r.lower)
            per_arm[f"bound_upper_a{a}"] = float(r.upper)
            per_arm[f"obs_arm_count_a{a}"] = int((acts == a).sum().item())
        self._bound_metrics = {
            "bound_width_mean": float(sum(widths) / len(widths)),
            "bound_width_max": float(max(widths)),
            "n_arms_with_informative_bound": float(n_inf),
            **per_arm,
        }

    def _compute_extra_metrics(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        latent_z: torch.Tensor | None,
    ) -> None:
        """Compute the v8 metrics: backdoor residual, target support overlap,
        and ``I(R; Z | S, A)`` lower bound.

        These all run on the offline buffer or the most recent rollout,
        with quiet NaN fallbacks when the inputs aren't available."""
        if not self.env.is_discrete_action:
            return
        n_actions = 8
        # 5.1 backdoor residual.  Z is observable in cells with expose_z=True;
        # we use the env-supplied latent_Z when available and treat hidden Z
        # as a NaN signal.  Only use latent_z when its leading dim matches
        # ``actions`` — the online path concatenates rollouts, in which case
        # the per-step ``info`` snapshot is too small.
        z_obs = None
        if self.cell_cfg.expose_z and latent_z is not None:
            if latent_z.shape[0] == actions.view(-1).shape[0]:
                z_obs = latent_z
        try:
            self._extra_metrics.update(
                backdoor_residual_summary(actions, rewards, z_obs, n_actions=n_actions)
            )
        except Exception:  # noqa: BLE001
            pass
        # 5.2 target_support_overlap: needs a propensity model and target-policy
        # actions on a held-out batch of observations.  Fitted on the same
        # buffer that produced the coverage metrics; reused via _propensity_model.
        try:
            if self._propensity_model is not None:
                with torch.no_grad():
                    behaviour_lp = self._propensity_model(obs)  # (N, n_actions)
                    target_actions, _ = self.algo.select_action(obs, deterministic=True)
                tso = target_support_overlap(
                    target_actions.view(-1), behaviour_lp, tau=1e-2
                )
                self._extra_metrics["target_support_overlap"] = float(tso.item())
        except Exception:  # noqa: BLE001
            pass
        # 5.3 conditional MI: only meaningful when latent_z is exposed and
        # aligned with the action/reward batch.
        if (
            latent_z is not None
            and latent_z.shape[0] == actions.view(-1).shape[0]
            and obs.shape[0] >= 64
        ):
            try:
                # Cap sample size for runtime predictability.
                n = min(int(obs.shape[0]), 1024)
                mi = conditional_mi_lower_bound(
                    obs[:n].detach(),
                    actions[:n].detach(),
                    rewards[:n].detach(),
                    latent_z[:n].detach(),
                    n_epochs=40,
                )
                self._extra_metrics["cond_mi_r_z_given_sa"] = float(mi)
            except Exception:  # noqa: BLE001
                pass

    def _evaluate_policy(self, seed: int) -> torch.Tensor:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        eval_env = self._make_eval_env_instance()
        try:
            with torch.no_grad():
                obs, _ = eval_env.reset(seed=seed)
                returns = torch.zeros((self._eval_n_envs,), device=self.device)
                for _ in range(eval_env.horizon):
                    action, _ = self.algo.select_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    returns = returns + reward
                    if bool(torch.all(terminated | truncated)):
                        break
            return returns
        finally:
            eval_env.close()
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def _build_tabular_oracle_policy(self, eval_env: TabularSepsisEnv) -> torch.Tensor:
        if self._oracle_policy_by_steps_left is not None:
            return self._oracle_policy_by_steps_left
        transition = eval_env.transition
        reward_probs = eval_env.reward_probs
        n_states, n_actions, _ = transition.shape
        horizon = int(eval_env.horizon)
        expected_reward = reward_probs[:, :, 1] - reward_probs[:, :, 0]
        values = torch.zeros((horizon + 1, n_states), device=transition.device)
        policy = torch.zeros((horizon + 1, n_states), dtype=torch.long, device=transition.device)
        for steps_left in range(1, horizon + 1):
            next_v = values[steps_left - 1]
            q = expected_reward + eval_env.gamma * torch.einsum("sak,k->sa", transition, next_v)
            values[steps_left], policy[steps_left] = torch.max(q, dim=1)
        self._oracle_policy_by_steps_left = policy
        return policy

    def _oracle_action(self, eval_env: CausalEnv, obs: torch.Tensor) -> torch.Tensor:
        # Discrete / tabular oracle (exact DP)
        if eval_env.is_discrete_action:
            if isinstance(eval_env, TabularSepsisEnv):
                policy = self._build_tabular_oracle_policy(eval_env)
                latent_state = eval_env._latent_state.to(torch.long)
                step_count = eval_env._step_count.to(torch.long)
                steps_left = (eval_env.horizon - step_count).clamp(min=1, max=eval_env.horizon)
                action = policy[steps_left, latent_state]
                return action.unsqueeze(-1)
            # fallback to learned policy for other discrete envs
            action, _ = self.algo.select_action(obs, deterministic=True)
            return action

        # Non-continuous fallback: use algorithm
        if not isinstance(eval_env, ContinuousWardEnv):
            action, _ = self.algo.select_action(obs, deterministic=True)
            return action

        # For ContinuousWard: support multiple oracle types
        oracle_choice = getattr(self.config, "oracle", "auto")
        if oracle_choice in ("auto", "cem"):
            try:
                return self._cem_oracle(eval_env, obs)
            except Exception:
                oracle_choice = "grid"
        if oracle_choice == "grid":
            act_dim = int(eval_env.act_shape[0])
            candidates_1d = torch.tensor([0.0, 0.5, 1.0], device=obs.device)
            mesh = torch.cartesian_prod(*([candidates_1d] * act_dim))
            best_score = torch.full((obs.shape[0],), -1e9, device=obs.device)
            best_action = torch.zeros((obs.shape[0], act_dim), device=obs.device)
            for i in range(mesh.shape[0]):
                candidate = mesh[i].unsqueeze(0).expand(obs.shape[0], -1)
                expected = eval_env.do_reward(candidate).mean(dim=1)
                better = expected > best_score
                best_score = torch.where(better, expected, best_score)
                best_action = torch.where(better.unsqueeze(-1), candidate, best_action)
            return best_action
        if oracle_choice == "algo":
            action, _ = self.algo.select_action(obs, deterministic=True)
            return action
        raise ValueError(f"Unknown oracle choice: {oracle_choice}")

    def _cem_oracle(self, eval_env: ContinuousWardEnv, obs: torch.Tensor) -> torch.Tensor:
        """Cross-Entropy Method (CEM) planning oracle for ContinuousWardEnv."""
        device = eval_env.state.device
        n_envs = int(eval_env.n_envs)
        act_dim = int(eval_env.act_shape[0])
        state_dim = int(eval_env.state.shape[-1])

        pop_size = 256
        num_iters = 4
        elite_frac = 0.1
        cem_horizon = min(5, int(eval_env.horizon))
        num_elite = max(1, int(pop_size * elite_frac))
        smoothing = 0.1

        means = torch.full((n_envs, cem_horizon, act_dim), 0.5, device=device)
        stds = torch.full((n_envs, cem_horizon, act_dim), 0.25, device=device)

        base_state = eval_env.state.detach()
        base_z = eval_env.latent_z.detach()
        base_u = eval_env.latent_u.detach()

        for _it in range(num_iters):
            samples = means.unsqueeze(1) + stds.unsqueeze(1) * torch.randn(
                (n_envs, pop_size, cem_horizon, act_dim), device=device
            )
            samples = samples.clamp(0.0, 1.0)

            returns = torch.zeros((n_envs, pop_size), device=device)
            discount = 1.0
            state = base_state.unsqueeze(1).expand(-1, pop_size, -1).reshape(-1, state_dim)
            z = base_z.unsqueeze(1).expand(-1, pop_size, -1).reshape(-1, base_z.shape[-1])
            u = base_u.unsqueeze(1).expand(-1, pop_size, -1).reshape(-1, base_u.shape[-1])

            for t in range(cem_horizon):
                action_t = samples[:, :, t, :].reshape(-1, act_dim)
                drift = eval_env._drift(state, action_t, z, u)
                next_state = (state + eval_env.dt * drift).clamp(0.0, 1.0)
                rew = eval_env._reward(next_state, action_t).reshape(n_envs, pop_size)
                returns = returns + discount * rew
                discount = discount * float(eval_env.gamma)
                state = next_state
            topk = torch.topk(returns, k=num_elite, dim=1).indices
            new_means = torch.zeros_like(means)
            new_stds = torch.zeros_like(stds)
            for i in range(n_envs):
                elites = samples[i, topk[i]]
                new_means[i] = elites.mean(dim=0)
                new_stds[i] = elites.std(dim=0) + 1e-6
            means = smoothing * new_means + (1.0 - smoothing) * means
            stds = smoothing * new_stds + (1.0 - smoothing) * stds

        actions = means[:, 0, :].clamp(0.0, 1.0)
        return actions

    def _evaluate_oracle(self, seed: int) -> torch.Tensor:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        eval_env = self._make_eval_env_instance()
        try:
            with torch.no_grad():
                obs, _ = eval_env.reset(seed=seed)
                returns = torch.zeros((self._eval_n_envs,), device=self.device)
                for _ in range(eval_env.horizon):
                    action = self._oracle_action(eval_env, obs)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    returns = returns + reward
                    if bool(torch.all(terminated | truncated)):
                        break
            return returns
        finally:
            eval_env.close()
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def _evaluate_gap_metrics(self, seed: int) -> dict[str, float]:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        eval_env = self._make_eval_env_instance()
        try:
            totals: dict[str, float] = {}
            beh_tv_total = 0.0
            n_steps = 0
            conf_buf: list[float] = []
            outcome_buf: list[float] = []
            with torch.no_grad():
                obs, info = eval_env.reset(seed=seed)
                for _ in range(eval_env.horizon):
                    action, log_prob = self.algo.select_action(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, info = eval_env.step(action)
                    step_metrics = compute_gap_metrics(
                        eval_env,
                        obs,
                        action,
                        observed_reward=reward,
                        pi_b_logprob=None,
                        n_samples=self.config.n_samples_gap,
                        n_bootstrap=self._effective_n_bootstrap,
                    )
                    for key, value in step_metrics.items():
                        totals[key] = totals.get(key, 0.0) + float(value)

                    # Dual-policy delta_tv: also measure under behavior policy action
                    latent = info.get("latent_U") if isinstance(info, dict) else None
                    beh_action, _ = self.behaviour_policy.select_action(obs, latent=latent)
                    beh_step_metrics = compute_gap_metrics(
                        eval_env,
                        obs,
                        beh_action,
                        observed_reward=None,
                        pi_b_logprob=None,
                        n_samples=self.config.n_samples_gap,
                        n_bootstrap=0,
                    )
                    beh_tv_total += float(beh_step_metrics.get("delta_tv", 0.0))

                    # Collect (confidence, binary_outcome) for ECE (tabular only)
                    if eval_env.is_discrete_action:
                        conf_buf.extend(torch.exp(log_prob).tolist())
                        outcome_buf.extend((reward > 0.0).float().tolist())

                    n_steps += 1
                    obs = next_obs
                    if bool(torch.all(terminated | truncated)):
                        break
            if n_steps == 0:
                return {
                    "delta_tv": 0.0,
                    "delta_tv_ci_lo": 0.0,
                    "delta_tv_ci_hi": 0.0,
                    "delta_kl": 0.0,
                    "delta_chi2": 0.0,
                    "delta_sup": 0.0,
                    "delta_mmd2": 0.0,
                    "delta_ks": 0.0,
                    "delta_tv_marginal": 0.0,
                    "delta_tv_conditional": 0.0,
                    "delta_tv_beh": 0.0,
                    "ece": 0.0,
                }
            result = {k: v / float(n_steps) for k, v in totals.items()}
            result["delta_tv_beh"] = beh_tv_total / float(n_steps)
            if conf_buf:
                c = torch.tensor(conf_buf, device=self.device)
                o = torch.tensor(outcome_buf, device=self.device)
                result["ece"] = expected_calibration_error(c, o)
            else:
                result["ece"] = 0.0
            return result
        finally:
            eval_env.close()
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def _make_perturbed_env(self, spec: PerturbationSpec) -> CausalEnv:
        """Return a copy of the eval env with transition/rewards perturbed by spec."""
        env = self._make_eval_env_instance()
        if isinstance(env, TabularSepsisEnv):
            env.transition = perturb_tabular_transition(env.transition, spec.eps_T)
            env.reward_probs = perturb_tabular_rewards(env.reward_probs, spec.eps_R)
        return env

    def _evaluate_perturbations(self, step: int, episode: int, seed: int) -> None:
        """Run closed-loop evaluation across TABULAR_DIAGONAL perturbation levels.

        Optimised in v10:
        - the base env is constructed once and reused across every spec;
        - the perturbed env for each spec is constructed once and reused
          for the rollout, the KS computation, and the bisim computation
          (was 3 separate constructions);
        - the bisim distance is computed from already-built env tensors,
          not by re-allocating yet another perturbed env.
        """
        if not self.config.eval_perturbations:
            return
        if not isinstance(self.env, TabularSepsisEnv):
            return

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        # Optional smoke knob: trim the perturbation grid to a smaller
        # number of specs so smoke configs don't pay for the full 5-spec
        # sweep.  Defaults to the full grid when the field is None.
        grid_size = getattr(self.config, "perturbation_grid_size", None)
        specs = TABULAR_DIAGONAL
        if grid_size is not None and 0 < int(grid_size) < len(TABULAR_DIAGONAL):
            specs = TABULAR_DIAGONAL[: int(grid_size)]

        base_env = self._make_eval_env_instance()
        try:
            for spec in specs:
                perturbed_env = self._make_perturbed_env(spec)
                try:
                    # (a) Closed-loop rollout on the perturbed env.
                    all_returns: list[float] = []
                    for ep in range(self.config.n_eval_episodes):
                        ep_seed = seed + ep * 1000
                        with torch.no_grad():
                            obs, _ = perturbed_env.reset(seed=ep_seed)
                            rets = torch.zeros((self._eval_n_envs,), device=self.device)
                            for _ in range(perturbed_env.horizon):
                                act, _ = self.algo.select_action(obs, deterministic=True)
                                obs, reward, terminated, truncated, _ = perturbed_env.step(
                                    act
                                )
                                rets += reward
                                if bool(torch.all(terminated | truncated)):
                                    break
                            all_returns.extend(rets.tolist())
                    ret_tensor = torch.tensor(all_returns)

                    # (b) D_env_KS — uses the already-built base_env and
                    # perturbed_env, no extra construction.
                    _tenv = base_env

                    def oracle_fn(obs: torch.Tensor, _e: CausalEnv = _tenv) -> torch.Tensor:
                        return self._oracle_action(_e, obs)

                    ks_val = d_env_ks(
                        base_env,
                        perturbed_env,
                        oracle_fn,
                        n_states=self._eval_n_envs,
                        n_samples=50,
                    )

                    # (c) D_bisim — operates directly on the env tensors of
                    # the already-built envs.  Avoids the previous double
                    # construction inside _compute_bisim_distance.
                    d_bisim_val = self._compute_bisim_distance_from_envs(
                        base_env, perturbed_env
                    )

                    self.perturbed_logger.write(
                        {
                            "step": step,
                            "episode": episode,
                            "wall_time": time.time(),
                            "cell": self.config.cell,
                            "env_name": self.config.env_name,
                            "algorithm": self.config.algorithm,
                            "behaviour_policy": self.config.behaviour,
                            "seed": self.config.seed,
                            "eps_T": spec.eps_T,
                            "eps_R": spec.eps_R,
                            "eval_perturbed_return_mean": float(ret_tensor.mean().item()),
                            "eval_perturbed_return_std": float(ret_tensor.std().item()),
                            "D_env_KS": ks_val,
                            "D_bisim": d_bisim_val,
                        }
                    )
                finally:
                    perturbed_env.close()
        finally:
            try:
                base_env.close()
            except Exception:  # noqa: BLE001
                pass
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def _compute_bisim_distance_from_envs(
        self, base_env: CausalEnv, target_env: CausalEnv
    ) -> float:
        """Average bisim distance between two already-built tabular MDPs.

        Sampling a 24-state random subset bounds the runtime on the 1440-
        state sepsis MDP.  The full bisim is computed by ``bisimulation_distance``
        on the projected transitions and rewards; the scalar returned is the
        mean diagonal of the pairwise metric on the subset.

        Returns ``NaN`` when either env isn't a tabular sepsis instance
        (continuous-ward bisim needs a samples-based approximation).
        """
        from causal_rl.metrics.bisimulation import bisimulation_distance

        if not isinstance(base_env, TabularSepsisEnv) or not isinstance(
            target_env, TabularSepsisEnv
        ):
            return float("nan")
        try:
            n_subset = 24  # bound runtime: 24x24 fixed-point.
            n_states = int(base_env.transition.shape[0])
            torch.manual_seed(0)
            idx = torch.randperm(n_states)[:n_subset]
            # Project transitions onto the subset and renormalise per row.
            base_t = base_env.transition[idx][:, :, idx]
            tgt_t = target_env.transition[idx][:, :, idx]
            base_t = base_t / (base_t.sum(dim=-1, keepdim=True).clamp(min=1e-9))
            tgt_t = tgt_t / (tgt_t.sum(dim=-1, keepdim=True).clamp(min=1e-9))
            # Expected reward = P(R=+1) - P(R=-1).
            base_r = base_env.reward_probs[idx, :, 1] - base_env.reward_probs[idx, :, 0]
            tgt_r = target_env.reward_probs[idx, :, 1] - target_env.reward_probs[idx, :, 0]
            d = bisimulation_distance(
                base_r,
                base_t,
                tgt_r,
                tgt_t,
                gamma=float(base_env.gamma) if float(base_env.gamma) < 1.0 else 0.99,
                n_iters=15,
            )
            return float(torch.diagonal(d).mean().item())
        except Exception:  # noqa: BLE001
            return float("nan")

    def run(self) -> None:
        if ALGOS[self.config.algorithm].kind == "off_policy":
            self._run_offline()
        else:
            self._run_online()

    def _discounted_returns(self, rewards: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        t_steps = rewards.shape[0]
        out = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for t in range(t_steps - 1, -1, -1):
            running = rewards[t] + self.env.gamma * running * (1.0 - done[t])
            out[t] = running
        return out

    def _run_online(self) -> None:
        obs, info = self.env.reset(seed=self.config.seed)
        episode = 0
        ep_return = torch.zeros((self.config.n_envs,), device=self.device)
        rollout_horizon = self.config.rollout_horizon or self.env.horizon
        train_ticks = checkpoint_ticks(
            self.config.total_frames, self.config.n_checkpoints_train
        )
        eval_ticks = checkpoint_ticks(
            self.config.total_frames, self.config.n_checkpoints_eval
        )
        latest_update_metrics: dict[str, float] = {}
        obs_buf: list[torch.Tensor] = []
        act_buf: list[torch.Tensor] = []
        rew_buf: list[torch.Tensor] = []
        next_obs_buf: list[torch.Tensor] = []
        done_buf: list[torch.Tensor] = []
        logp_buf: list[torch.Tensor] = []
        for tick in range(1, self.config.total_frames + 1):
            action, log_prob = self.algo.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = (terminated | truncated).float()
            ep_return += reward
            returns_snapshot = ep_return.clone()
            obs_buf.append(obs)
            act_buf.append(action.detach())
            rew_buf.append(reward.detach())
            next_obs_buf.append(next_obs)
            done_buf.append(done.detach())
            logp_buf.append(log_prob.detach())

            should_update = (len(obs_buf) >= rollout_horizon) or (tick == self.config.total_frames)
            if should_update:
                rewards_t = torch.stack(rew_buf, dim=0)
                done_t = torch.stack(done_buf, dim=0)
                returns_t = self._discounted_returns(rewards_t, done_t)
                obs_batch = torch.cat(obs_buf, dim=0)
                action_batch = torch.cat(act_buf, dim=0)
                reward_batch = torch.cat(rew_buf, dim=0)
                next_obs_batch = torch.cat(next_obs_buf, dim=0)
                done_batch = torch.cat(done_buf, dim=0)
                logp_batch = torch.cat(logp_buf, dim=0)
                self._compute_coverage_metrics(logp_batch.view(-1))
                self._compute_bound_metrics(action_batch, reward_batch)
                # v8: extra metrics on the rollout buffer.  Latent_z is read
                # from the most recent ``info`` dict when expose_z=True.
                latent_z = None
                if isinstance(info, dict):
                    latent_z = info.get("latent_Z")
                try:
                    self._compute_extra_metrics(
                        obs_batch, action_batch, reward_batch, latent_z
                    )
                except Exception:  # noqa: BLE001
                    pass
                returns = returns_t.reshape(-1)
                value_old = torch.zeros_like(returns)
                adv = returns.clone()
                policy = getattr(self.algo, "policy", None)
                if policy is not None and hasattr(policy, "value"):
                    with torch.no_grad():
                        value_old = policy.value(obs_batch).detach()
                    adv = returns - value_old
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                batch = {
                    "obs": obs_batch,
                    "action": action_batch,
                    "reward": reward_batch,
                    "next_obs": next_obs_batch,
                    "done": done_batch,
                    "returns": returns,
                    "adv": adv,
                    "value": value_old,
                    "log_prob": logp_batch,
                }
                metrics = self.algo.update(batch)
                latest_update_metrics = metrics
                obs_buf.clear()
                act_buf.clear()
                rew_buf.clear()
                next_obs_buf.clear()
                done_buf.clear()
                logp_buf.clear()

            obs = next_obs
            if bool(torch.any(terminated | truncated)):
                episode += 1
                ep_return = torch.zeros_like(ep_return)
                obs, info = self.env.reset()
            if tick in train_ticks:
                gap_metrics = self._evaluate_gap_metrics(seed=self.config.seed + tick + 40_000)
                self._last_gap_metrics = gap_metrics
                metrics_for_log = {**latest_update_metrics, **gap_metrics}
                self._log_train(
                    step=tick, episode=episode, returns=returns_snapshot, metrics=metrics_for_log
                )
            if tick in eval_ticks:
                eval_returns = self._evaluate_policy(seed=self.config.seed + tick + 10_000)
                oracle_returns = self._evaluate_oracle(seed=self.config.seed + tick + 30_000)
                gap_metrics = self._evaluate_gap_metrics(seed=self.config.seed + tick + 50_000)
                self._log_eval(
                    step=tick,
                    episode=episode,
                    eval_returns=eval_returns,
                    oracle_returns=oracle_returns,
                    metrics=gap_metrics,
                )
        # Perturbation eval is policy-after-training analysis; the per-spec
        # bisim/KS quantities depend only on the env-pair, so running this
        # at every checkpoint just multiplied wall time by ``n_checkpoints_eval``
        # without adding new information.  Single end-of-training call.
        self._evaluate_perturbations(
            step=self.config.total_frames,
            episode=episode,
            seed=self.config.seed + self.config.total_frames + 60_000,
        )

    def _run_offline(self) -> None:
        obs, info = self.env.reset(seed=self.config.seed)
        del obs, info
        buffer = collect_offline_dataset(
            env=self.env,
            behaviour_policy=self.behaviour_policy,
            n_transitions=self.config.offline_transitions,
            expose_pi_b=self.cell_cfg.pi_b_known,
            expose_latent=self.behaviour_policy.depends_on_u,
            seed=self.config.seed,
        )
        self._compute_bound_metrics(buffer.action[: buffer.size], buffer.reward[: buffer.size])
        if (not self.cell_cfg.pi_b_known) and self.env.is_discrete_action and buffer.size > 0:
            # π_b is hidden — fit a propensity model and use its log-prob of
            # the action TAKEN to seed coverage metrics.  Without this, the
            # buffer's behaviour_logprob is all-zeros and min_propensity
            # collapses to exp(0) = 1.0 (degenerate).
            model: PropensityModel = fit_propensity_model(
                buffer.obs[: buffer.size], buffer.action[: buffer.size], n_actions=8
            )
            self._propensity_model = model
            with torch.no_grad():
                pred_lp_per_action = model(buffer.obs[: buffer.size])  # (N, 8)
            actions_taken = buffer.action[: buffer.size].view(-1, 1).long()
            learned_log_probs = pred_lp_per_action.gather(1, actions_taken).squeeze(-1)
            self._compute_coverage_metrics(learned_log_probs)
            # ECE on the (N, n_actions) log-prob matrix vs the integer actions
            # that were *actually taken* — i.e. how well-calibrated is the
            # model's "what action did the behaviour pick?" classifier.
            ece = expected_calibration_error_2d(
                pred_lp_per_action,
                buffer.action[: buffer.size].view(-1).long(),
                n_bins=10,
            )
            self._propensity_calibration_ece = (
                float(ece.item()) if hasattr(ece, "item") else float(ece)
            )
            # The collector records the *true* π_b log-prob of the taken
            # action in a separate channel that's always populated, even
            # when ``expose_pi_b=False`` hides it from the algorithm.  Use
            # that channel to compute KL(true ‖ predicted) on the taken
            # actions — a one-sample plug-in estimator of the per-state KL.
            true_lp_buf = getattr(buffer, "true_behaviour_logprob", None)
            if true_lp_buf is not None and buffer.size > 0:
                true_lp_taken = true_lp_buf[: buffer.size]
                finite_mask = ~torch.isnan(true_lp_taken)
                if bool(finite_mask.any()):
                    diff = (
                        true_lp_taken[finite_mask]
                        - learned_log_probs.detach()[finite_mask]
                    )
                    self._pi_b_recovery_kl = max(0.0, float(diff.mean().item()))
                else:
                    self._pi_b_recovery_kl = float("nan")
            else:
                self._pi_b_recovery_kl = float("nan")
        elif buffer.behaviour_logprob is not None and buffer.size > 0:
            self._compute_coverage_metrics(buffer.behaviour_logprob[: buffer.size])
            self._propensity_calibration_ece = float("nan")
            # When π_b is fully known the recovery KL is by definition zero.
            self._pi_b_recovery_kl = 0.0
        # v8 metrics — backdoor residual, target overlap, conditional MI.
        if buffer.size > 0:
            try:
                self._compute_extra_metrics(
                    buffer.obs[: buffer.size],
                    buffer.action[: buffer.size],
                    buffer.reward[: buffer.size],
                    buffer.latent[: buffer.size]
                    if buffer.latent is not None
                    else None,
                )
            except Exception:  # noqa: BLE001
                # Never fail a run because of a metric error.
                pass
        episode = 0
        total_updates = self.config.total_frames
        train_ticks = checkpoint_ticks(total_updates, self.config.n_checkpoints_train)
        eval_ticks = checkpoint_ticks(total_updates, self.config.n_checkpoints_eval)
        for step in range(1, total_updates + 1):
            batch_obj = buffer.sample(self.config.batch_size)
            # Use the most recently computed gap metric as the sensitivity signal
            # for ConfoundedDQN (the true Δ_φ from the eval env, not latent magnitude).
            delta_tv_val = float(self._last_gap_metrics.get("delta_tv", 0.0))
            batch: dict[str, object] = {
                "obs": batch_obj.obs,
                "action": batch_obj.action,
                "reward": batch_obj.reward,
                "next_obs": batch_obj.next_obs,
                "done": batch_obj.done,
                "delta_tv": delta_tv_val,
            }
            metrics = self.algo.update(batch)  # type: ignore[arg-type]
            if step in train_ticks:
                gap_metrics = self._evaluate_gap_metrics(seed=self.config.seed + step + 40_000)
                self._last_gap_metrics = gap_metrics
                if self.cell_cfg.pi_b_known:
                    # IPW-corrected gap estimate using batch behaviour log-probs.
                    ipw_batch_metrics = compute_gap_metrics(
                        self.env,
                        batch_obj.obs,
                        batch_obj.action,
                        observed_reward=batch_obj.reward,
                        pi_b_logprob=batch_obj.behaviour_logprob,
                    )
                    # Conditional (IPW-corrected) gap is more interpretable in known-π_b cells.
                    gap_metrics["delta_tv_conditional"] = ipw_batch_metrics.get(
                        "delta_tv_conditional", gap_metrics.get("delta_tv_conditional", 0.0)
                    )
                    gap_metrics["delta_tv_marginal"] = gap_metrics.get("delta_tv", 0.0)
                metrics_for_log = {**metrics, **gap_metrics}
                self._log_train(
                    step=step,
                    episode=episode,
                    returns=batch_obj.reward,
                    metrics=metrics_for_log,
                )
            if step in eval_ticks:
                eval_returns = self._evaluate_policy(seed=self.config.seed + step + 10_000)
                oracle_returns = self._evaluate_oracle(seed=self.config.seed + step + 30_000)
                gap_metrics = self._evaluate_gap_metrics(seed=self.config.seed + step + 50_000)
                self._log_eval(
                    step=step,
                    episode=episode,
                    eval_returns=eval_returns,
                    oracle_returns=oracle_returns,
                    metrics=gap_metrics,
                )
        # Single end-of-training perturbation eval (see _run_online for the
        # rationale).
        self._evaluate_perturbations(
            step=total_updates,
            episode=episode,
            seed=self.config.seed + total_updates + 60_000,
        )
