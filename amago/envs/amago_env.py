from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import gym as og_gym
import gymnasium as gym

from amago.envs.env_utils import (
    ContinuousActionWrapper,
    DiscreteActionWrapper,
    MultiBinaryActionWrapper,
    space_convert,
)
from amago.hindsight import Timestep


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
            action = [i for i in range(self.action_size)]
        elif self.multibinary:
            action = np.zeros((self.action_size,), dtype=np.int8)
        else:
            action = np.full((self.action_size,), -2.0)
        return action

    def make_action_rep(self, action) -> np.ndarray:
        if self.discrete:
            action_rep = np.zeros((self.action_size,))
            action_rep[action] = 1.0
        else:
            action_rep = action.copy()
        return action_rep

    def inner_reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def reset(self, seed=None, options=None) -> Timestep:
        self.step_count = np.zeros((self.batched_envs,), dtype=np.int32)
        obs, info = self.inner_reset(seed=seed, options=options)
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        if self.batched_envs == 1:
            obs = {k: [v] for k, v in obs.items()}
        timesteps = []
        for idx in range(self.batched_envs):
            timesteps.append(
                Timestep(
                    obs={k: v[idx] for k, v in obs.items()},
                    prev_action=self.make_action_rep(self.blank_action),
                    reward=0.0,
                    terminal=False,
                    time_idx=0,
                ),
            )
        return timesteps, info

    def inner_step(self, action):
        return self.env.step(action)

    def step(self, action: np.ndarray) -> tuple[Timestep, float, bool, bool, dict]:
        # take environment step
        obs, rewards, terminated, truncated, info = self.inner_step(action)
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        if self.batched_envs == 1:
            action = action[np.newaxis, :]
            rewards = [rewards]
            terminated = [terminated]
            truncated = [truncated]
            obs = {k: [v] for k, v in obs.items()}

        timesteps = []
        for idx in range(self.batched_envs):
            self.step_count[idx] += 1
            done = terminated[idx] or truncated[idx]
            timesteps.append(
                Timestep(
                    obs={k: v[idx] for k, v in obs.items()},
                    prev_action=self.make_action_rep(action[idx]),
                    reward=rewards[idx],
                    terminal=done,
                    time_idx=self.step_count[idx],
                )
            )
            if done:
                self.step_count[idx] = 0

        return (
            timesteps,
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            info,
        )
