import random
from abc import ABC, abstractmethod
import copy

import numpy as np
import gymnasium as gym

from amago.envs.env_utils import (
    ContinuousActionWrapper,
    DiscreteActionWrapper,
    space_convert,
)
from amago.hindsight import Trajectory, GoalSeq, Timestep


class AMAGOEnv(gym.Wrapper, ABC):
    def __init__(self, env: gym.Env, horizon: int, start: int = 0):
        super().__init__(env)

        self.horizon = horizon
        self.start = start
        # action space conversion
        self.discrete = isinstance(space_convert(env.action_space), gym.spaces.Discrete)
        if self.discrete:
            self.env = DiscreteActionWrapper(self.env)
            self.action_size = self.action_space.n
        else:
            self.env = ContinuousActionWrapper(self.env)
            self.action_size = self.action_space.shape[-1]
        self.action_space = space_convert(self.env.action_space)
        # observation space conversion (defaults to dict)
        obs_space = self.env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            obs_space = gym.spaces.Dict({"observation": space_convert(obs_space)})
        self.observation_space = gym.spaces.Dict(
            {k: space_convert(v) for k, v in obs_space.items()}
        )

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    @property
    @abstractmethod
    def env_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def achieved_goal(self) -> list[np.ndarray]:
        raise NotImplementedError

    @property
    @abstractmethod
    def kgoal_space(self) -> gym.spaces.Box:
        raise NotImplementedError

    @property
    @abstractmethod
    def goal_sequence(self) -> GoalSeq:
        raise NotImplementedError

    @property
    def max_goal_seq_length(self):
        return self.kgoal_space.shape[0]

    @property
    def blank_action(self):
        if self.discrete:
            action = [i for i in range(self.action_size)]
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
        self.step_count = 0
        obs, _ = self.inner_reset(seed=seed, options=options)
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        timestep = Timestep(
            obs=obs,
            prev_action=self.make_action_rep(self.blank_action),
            achieved_goal=self.achieved_goal,
            goal_seq=self.goal_sequence,
            time=0.0,
            reset=True,
            real_reward=None,
            terminal=False,
            raw_time_idx=0,
        )
        return timestep

    def inner_step(self, action):
        return self.env.step(action)

    def step(
        self,
        action,
        normal_rl_reward: bool = False,
        normal_rl_reset: bool = False,
        soft_reset_kwargs: dict = {},
    ) -> tuple[Timestep, float, bool, bool, dict]:
        assert isinstance(action, np.ndarray)
        if self.discrete:
            assert action.dtype == np.uint8
        else:
            assert action.dtype == np.float32

        # freeze goal sequence before this timestep
        goal_seq = copy.deepcopy(self.goal_sequence)

        # take environment step
        obs, real_reward, terminated, truncated, info = self.inner_step(action)
        self.step_count += 1

        if not normal_rl_reward:
            real_reward = None
        elif self.step_count < self.start:
            real_reward = 0.0

        soft_done = terminated or truncated
        if soft_done and not normal_rl_reset:
            # roll through episode resets
            obs, info = self.env.reset(**soft_reset_kwargs)

        if not isinstance(obs, dict):
            obs = {"observation": obs}
        # create Timestep. goal_seq holds desired goal before this timestep,
        # achieved_goal holds goal we actually achieved this timestep.
        timestep = Timestep(
            obs=obs,
            prev_action=self.make_action_rep(action),
            achieved_goal=copy.deepcopy(self.achieved_goal),
            goal_seq=goal_seq,
            time=float(self.step_count) / self.horizon,
            reset=soft_done,
            real_reward=real_reward,
            raw_time_idx=self.step_count,
        )

        # meta-termination (triggers resets of this environment,
        # and used during optimization)
        outer_terminated = (
            normal_rl_reset and terminated
        ) or timestep.all_goals_completed
        outer_truncated = self.step_count >= self.horizon or (
            truncated and normal_rl_reset
        )
        if outer_terminated or outer_truncated:
            timestep.terminal = True

        return timestep, timestep.reward, outer_terminated, outer_truncated, info
