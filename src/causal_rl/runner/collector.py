from __future__ import annotations

import torch

from causal_rl.behaviour.base import BiasedExplorer
from causal_rl.envs.base import CausalEnv
from causal_rl.replay.buffer import ReplayBuffer


def collect_offline_dataset(
    env: CausalEnv,
    behaviour_policy: BiasedExplorer,
    n_transitions: int,
    expose_pi_b: bool,
    expose_latent: bool,
    seed: int,
) -> ReplayBuffer:
    torch.manual_seed(seed)
    obs, info = env.reset(seed=seed)
    obs_dim = obs.shape[-1]
    act_dim = 1 if env.is_discrete_action else env.act_shape[0]
    buffer = ReplayBuffer(
        capacity=n_transitions,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=obs.device,
        discrete_action=env.is_discrete_action,
    )
    done = torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
    while buffer.size < n_transitions:
        latent = info.get("latent_U")
        action, logprob = behaviour_policy.select_action(
            obs, latent=latent if expose_latent else None
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        latent_logged = info.get("latent_U")
        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done.float(),
            behaviour_logprob=logprob if expose_pi_b else None,
            latent=latent_logged if expose_latent else None,
        )
        obs = next_obs
        if torch.any(done):
            obs, info = env.reset()
    return buffer
