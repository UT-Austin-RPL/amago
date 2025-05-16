import os
import random
import warnings
import time
from uuid import uuid4
from typing import Optional, Any, Iterable, Callable, Type
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import gym as og_gym
import gymnasium as gym
import torch

from .env_utils import (
    ContinuousActionWrapper,
    DiscreteActionWrapper,
    MultiBinaryActionWrapper,
    space_convert,
)
from .exploration import ExplorationWrapper
from amago.hindsight import Timestep, Trajectory, split_batched_timestep
from amago.loading import get_path_to_trajs


class AMAGOEnv(gym.Wrapper):
    def __init__(
        self, env: gym.Env, env_name: Optional[str] = None, batched_envs: int = 1
    ):
        super().__init__(env)
        self.batched_envs = batched_envs
        self._env_name = env_name.replace("/", "_") if env_name is not None else None

        # action space conversion
        self.discrete = isinstance(space_convert(env.action_space), gym.spaces.Discrete)
        self.multibinary = isinstance(
            space_convert(env.action_space), gym.spaces.MultiBinary
        )
        if self.discrete:
            self.env = DiscreteActionWrapper(self.env)
            self.action_size = self.action_space.n
            self.action_dtype = np.uint8
        elif self.multibinary:
            self.env = MultiBinaryActionWrapper(self.env)
            self.action_size = self.action_space.n
            self.action_dtype = np.uint8
        else:
            self.env = ContinuousActionWrapper(self.env)
            self.action_size = self.action_space.shape[-1]
            self.action_dtype = np.float32
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
        (for logging and saving .traj files)
        """
        if self._env_name is None:
            raise ValueError(
                "AMAGOEnv env_name is not set. Pass `env_name` on init or override `env_name` property."
            )
        return self._env_name

    def make_action_rep(self, action) -> np.ndarray:
        if self.discrete:
            # action as one-hot
            action_rep = np.zeros((self.batched_envs, self.action_size), dtype=np.uint8)
            action_rep[self._batch_idxs, action[..., 0]] = 1
        else:
            action_rep = action.copy()
            if self.batched_envs == 1 and action_rep.ndim == 1:
                action_rep = np.expand_dims(action_rep, axis=0)
        return action_rep.astype(self.action_dtype)

    def inner_reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def reset(self, seed=None, options=None) -> Timestep:
        self.step_count = np.zeros((self.batched_envs,), dtype=np.int64)
        obs, info = self.inner_reset(seed=seed, options=options)
        if not isinstance(obs, dict):
            # force dict obs
            obs = {"observation": obs}
        if self.batched_envs == 1:
            # force batch dim
            obs = {k: v[np.newaxis, ...] for k, v in obs.items()}
        timestep = Timestep(
            obs=obs,
            prev_action=np.zeros(
                (self.batched_envs, self.action_size), dtype=self.action_dtype
            ),
            reward=np.zeros((self.batched_envs,), dtype=np.float32),
            terminal=np.zeros((self.batched_envs,), dtype=bool),
            time_idx=self.step_count.copy(),
            batched_envs=self.batched_envs,
        )
        return timestep, info

    def inner_step(self, action):
        return self.env.step(action)

    def step(
        self, action: np.ndarray
    ) -> tuple[Timestep, np.ndarray, np.ndarray, np.ndarray, dict]:
        # take environment step
        obs, rewards, terminateds, truncateds, infos = self.inner_step(action)
        self.step_count += 1
        if not isinstance(obs, dict):
            # force dict obs
            obs = {"observation": obs}
        if self.batched_envs == 1:
            # force batch dim
            obs = {k: v[np.newaxis, ...] for k, v in obs.items()}
            rewards = np.array([rewards], dtype=np.float32)
            terminateds = np.array([terminateds], dtype=bool)
            truncateds = np.array([truncateds], dtype=bool)
        prev_actions = self.make_action_rep(action)
        timestep = Timestep(
            obs=obs,
            prev_action=prev_actions,
            reward=rewards,
            terminal=terminateds,
            time_idx=self.step_count.copy(),
            batched_envs=self.batched_envs,
        )
        # reset the step count here in case the env is going to auto-reset itself
        self.step_count *= ~np.logical_or(terminateds, truncateds)
        return timestep, rewards, terminateds, truncateds, infos


class ReturnHistory:
    def __init__(self, env_name):
        self.data = {}

    def add_score(self, env_name, score):
        if env_name in self.data:
            self.data[env_name].append(score)
        else:
            self.data[env_name] = [score]


class SpecialMetricHistory:
    log_prefix = "AMAGO_LOG_METRIC"

    def __init__(self, env_name):
        self.data = {}

    def add_score(self, env_name: str, key: str, value: Any):
        if key.startswith(self.log_prefix):
            # remove logging tag
            key = key[len(self.log_prefix) :].strip()
        if env_name not in self.data:
            self.data[env_name] = {}
        if isinstance(value, torch.Tensor | np.ndarray):
            # for batched env stats
            value = value.flatten().tolist()
        # flatten
        if not isinstance(value, Iterable):
            value = [value]
        if key not in self.data[env_name]:
            self.data[env_name][key] = value
        else:
            self.data[env_name][key].extend(value)


AMAGO_ENV_LOG_PREFIX = SpecialMetricHistory.log_prefix


class SequenceWrapper(gym.Wrapper):
    """
    A wrapper that saves trajectory files to disk and logs rollout metrics.
    Automatically logs total return in all envs.

    We also log any metric from the gym env's `info` dict that begins with "AMAGO_LOG_METRIC"
    (`amago.envs.env_utils.AMAGO_ENV_LOG_PREFIX`).
    """

    def __init__(
        self,
        env: gym.Env,
        save_trajs_to: Optional[str],
        save_every: Optional[tuple[int, int]] = None,
        save_trajs_as: str = "npz",
    ):
        super().__init__(env)

        self.batched_envs = env.batched_envs
        self.dset_write_dir = save_trajs_to
        self.saving = self.dset_write_dir is not None
        if self.saving:
            os.makedirs(self.dset_write_dir, exist_ok=True)
        self.save_every = save_every
        self.save_trajs_as = save_trajs_as
        self._total_frames = 0
        self._total_frames_by_env_name = defaultdict(int)
        self.finished_trajs = []
        self.reset_stats()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_shape = self.env.action_space.n
        else:
            action_shape = self.env.action_space.shape[-1]
        rl2_shape = action_shape + 1  # action + reward
        self.rl2_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "rl2": gym.spaces.Box(
                    shape=(rl2_shape,),
                    dtype=np.float32,
                    low=float("-inf"),
                    high=float("inf"),
                ),
            }
        )

    @property
    def step_count(self):
        return self.env.step_count

    def reset_stats(self):
        # stores all of the success/return histories
        self.return_history = ReturnHistory(self.env_name)
        self.special_history = SpecialMetricHistory(self.env_name)

    def random_traj_length(self):
        return random.randint(*self.save_every) if self.save_every else None

    def reset(self, seed=None) -> Timestep:
        timestep, info = self.env.reset(seed=seed)
        assert timestep.batched_envs == self.batched_envs
        self._current_timestep = timestep.as_input()
        self.active_trajs = [
            Trajectory(timesteps=[t]) for t in split_batched_timestep(timestep)
        ]
        self.since_last_save = [0 for _ in range(self.batched_envs)]
        self.save_this_time = [
            self.random_traj_length() for _ in range(self.batched_envs)
        ]
        self.total_return = np.zeros(self.batched_envs, dtype=np.float64)
        return timestep.obs, info

    def step(self, action):
        timestep, reward, terminated, truncated, info = self.env.step(action)
        assert terminated.shape[0] == self.batched_envs
        assert truncated.shape[0] == self.batched_envs
        assert reward.shape[0] == self.batched_envs

        self.total_return += reward
        done = np.logical_or(terminated, truncated)
        for idx, split_timestep in enumerate(split_batched_timestep(timestep)):
            self.active_trajs[idx].add_timestep(split_timestep)
            self.since_last_save[idx] += 1
            if done[idx]:
                # we're going to handle this as if the env auto-resets such that the last timestep of this trajectory
                # is the same as the first timestep of the next trajectory. However, if the env is not vectorized, this
                # will all be overriden by the `reset` call that is passed down by AsyncVectorEnv as soon as it sees a `done`.
                self.return_history.add_score(self.env.env_name, self.total_return[idx])
                self.total_return[idx] = 0
                # the observation is correct for the new traj but the step counter, terminals, etc. of the `Timestep` are not
                split_timestep = split_timestep.create_reset_version(np.array([True]))
                self.finish_active_traj(idx=idx, new_init_timestep=split_timestep)
            elif (
                self.save_every is not None
                and self.since_last_save[idx] > self.save_this_time[idx]
            ):
                # initial timestep of new trajectory is the exact same as the last timestep of the previous trajectory
                self.finish_active_traj(idx=idx, new_init_timestep=split_timestep)

        for info_key, info_val in info.items():
            if info_key.startswith(self.special_history.log_prefix):
                self.special_history.add_score(self.env.env_name, info_key, info_val)

        self._total_frames += timestep.batched_envs
        self._total_frames_by_env_name[self.env.env_name] += timestep.batched_envs

        if done.any():  # avoid a deepcopy if we can
            timestep = timestep.create_reset_version(done)

        # - if the env is running in a pool of async/sync envs, terminated/truncated will trigger a `reset`, redoing the reset logic already done here
        # - if the env is vectorized, dones will never be checked for a reset but *will* be used to reset the agent's hidden state
        self._current_timestep = timestep.as_input()
        return (
            timestep.obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def finish_active_traj(self, idx: int, new_init_timestep: Timestep):
        if self.saving:
            # environment name, a random id, and a timestamp .traj
            traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{time.time()}"
            path = os.path.join(self.dset_write_dir, traj_name)
            # put path and trajectory in the queue to be saved
            self.finished_trajs.append((path, self.active_trajs[idx]))
            self.since_last_save[idx] = 0
            self.save_this_time[idx] = self.random_traj_length()
        # these values will be overriden if we can call `reset`
        self.active_trajs[idx] = Trajectory(timesteps=[new_init_timestep])

    def save_finished_trajs(self):
        # save all the trajectories we've finished to disk
        while len(self.finished_trajs) > 0:
            path, traj = self.finished_trajs.pop()
            traj.save_to_disk(path, save_as=self.save_trajs_as)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def total_frames_by_env_name(self) -> dict[str, int]:
        return self._total_frames_by_env_name

    @property
    def current_timestep(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._current_timestep

    @property
    def env_name(self):
        return self.env.env_name


@dataclass
class EnvCreator:
    make_env: Callable
    exploration_wrapper_type: Type[ExplorationWrapper]
    save_trajs_to: Optional[str]
    save_every_low: int
    save_every_high: int
    save_trajs_as: str

    def __post_init__(self):
        self.rl2_space = None
        self.already_vectorized = False

    def __call__(self):
        env = self.make_env()
        self.already_vectorized = env.batched_envs > 1
        if self.exploration_wrapper_type is not None:
            env = self.exploration_wrapper_type(env)
        env = SequenceWrapper(
            env,
            save_every=(self.save_every_low, self.save_every_high),
            save_trajs_to=self.save_trajs_to,
            save_trajs_as=self.save_trajs_as,
        )
        self.rl2_space = env.rl2_space
        return env
