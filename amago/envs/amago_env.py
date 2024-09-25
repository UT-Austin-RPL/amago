from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import gym as og_gym
import gymnasium as gym
import torch
import torch.nn.functional as F

from amago.envs.env_utils import (
    ContinuousActionWrapper,
    DiscreteActionWrapper,
    MultiBinaryActionWrapper,
    space_convert,
)
from amago.hindsight import Timestep
from amago.utils import unstack_dict


class AMAGOEnv(gym.Wrapper):
    def __init__(
        self, env: gym.Env, env_name: Optional[str] = None, batched_envs: int = 1
    ):
        super().__init__(env)
        self.batched_envs = batched_envs
        self._env_name = env_name

        # action space conversion
        self.discrete = isinstance(space_convert(env.action_space), gym.spaces.Discrete)
        self.multibinary = isinstance(
            space_convert(env.action_space), gym.spaces.MultiBinary
        )
        if self.discrete:
            self.env = DiscreteActionWrapper(self.env)
            self.action_size = self.action_space.n
        elif self.multibinary:
            self.env = MultiBinaryActionWrapper(self.env)
            self.action_size = self.action_space.n
        else:
            self.env = ContinuousActionWrapper(self.env)
            self.action_size = self.action_space.shape[-1]
        self.action_space = space_convert(self.env.action_space)
        self._batch_idxs = np.arange(self.batched_envs)

        # observation space conversion (defaults to dict)
        obs_space = self.env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict | og_gym.spaces.Dict):
            obs_space = gym.spaces.Dict({"observation": space_convert(obs_space)})
        self.observation_space = gym.spaces.Dict(
            {k: space_convert(v) for k, v in obs_space.items()}
        )

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    @property
    def env_name(self):
        """
        Dynamically change the name of the current environment in a multi-task setting
        """
        if self._env_name is None:
            raise ValueError(
                "AMAGOEnv env_name is not set. Pass `env_name` on init or override `env_name` property."
            )
        return self._env_name

    @property
    def blank_action(self):
        if self.discrete:
            action = np.ones((self.batched_envs, self.action_size), dtype=np.int8)
        elif self.multibinary:
            action = np.zeros((self.batched_envs, self.action_size), dtype=np.int8)
        else:
            action = np.full((self.batched_envs, self.action_size), -2.0)
        return action

    def make_action_rep(self, action) -> np.ndarray:
        if self.discrete:
            action_rep = np.zeros((self.batched_envs, self.action_size), dtype=np.uint8)
            action_rep[self._batch_idxs, action[..., 0]] = 1
        else:
            action_rep = action.copy()
            if self.batched_envs == 1 and action_rep.ndim == 1:
                action_rep = action_rep[np.newaxis, ...]
        return action_rep

    def inner_reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def reset(self, seed=None, options=None) -> Timestep:
        self.step_count = np.zeros((self.batched_envs, 1), dtype=np.int64)
        obs, info = self.inner_reset(seed=seed, options=options)
        if not isinstance(obs, dict):
            obs = {"observation": obs}

        if self.batched_envs == 1:
            obs = [obs]
            prev_actions = [self.blank_action[0, ...]]
        else:
            obs = unstack_dict(obs)
            prev_actions = np.unstack(self.blank_action, axis=0)

        timesteps = []
        for idx in range(self.batched_envs):
            timesteps.append(
                Timestep(
                    obs=obs[idx],
                    prev_action=prev_actions[idx],
                    reward=0.0,
                    terminal=False,
                    time_idx=self.step_count[idx].item(),
                ),
            )
        return timesteps, info

    def inner_step(self, action):
        return self.env.step(action)

    def step(self, action: np.ndarray) -> tuple[Timestep, float, bool, bool, dict]:
        # take environment step
        self.step_count += 1
        obs, rewards, terminateds, truncateds, infos = self.inner_step(action)
        if not isinstance(obs, dict):
            # force dict obs
            obs = {"observation": obs}

        if self.batched_envs == 1:
            # "batch" an unbatched env so the wrappers above know what shape to expect
            dones = terminateds or truncateds
            prev_actions = self.make_action_rep(action[np.newaxis, ...])
            timesteps = [
                Timestep(
                    obs=obs,
                    prev_action=prev_actions[0],
                    reward=rewards,
                    terminal=dones,
                    time_idx=self.step_count[0].item(),
                )
            ]
            rewards = np.array([rewards], dtype=np.float32)
            terminateds = np.array([terminateds], dtype=bool)
            truncateds = np.array([truncateds], dtype=bool)
        else:
            dones = np.logical_or(terminateds, truncateds)
            prev_actions = self.make_action_rep(action)
            # unstack to avoid indexing arrays during `Timestep` creation
            _dones = np.unstack(dones, axis=0)
            _obs = unstack_dict(obs)
            _rewards = np.unstack(rewards, axis=0)
            _prev_actions = np.unstack(prev_actions, axis=0)
            _time_idxs = np.unstack(self.step_count, axis=0)
            while dones.ndim < 2:
                dones = dones[..., np.newaxis]
            timesteps = []
            for idx in range(self.batched_envs):
                # we can end up with random jax/np datatypes here...
                timesteps.append(
                    Timestep(
                        obs=_obs[idx],
                        prev_action=_prev_actions[idx],
                        reward=_rewards[idx],
                        terminal=_dones[idx],
                        time_idx=_time_idxs[idx],
                    )
                )

        # likely redundant step count reset... just in case the environment
        # is not automatically reset by the parallel wrapper at the top of the stack.
        self.step_count *= ~dones
        return timesteps, rewards, terminateds, truncateds, infos
