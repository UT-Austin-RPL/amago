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
from amago.hindsight import Timestep, Trajectory
from amago.utils import unstack_dict, stack_list_array_dicts


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
            action = np.ones((self.batched_envs, self.action_size), dtype=np.uint8)
        elif self.multibinary:
            action = np.zeros((self.batched_envs, self.action_size), dtype=np.uint8)
        else:
            action = np.full((self.batched_envs, self.action_size), -2.0)
        return action

    def make_action_rep(self, action) -> np.ndarray:
        if self.discrete:
            action_rep = np.zeros((self.batched_envs, self.action_size), dtype=np.uint8)
            action_rep[self._batch_idxs, action[..., 0]] = 1
        else:
            action_rep = action.copy().astype(
                np.uint8 if self.multibinary else np.float32
            )
            if self.batched_envs == 1 and action_rep.ndim == 1:
                action_rep = np.expand_dims(action_rep, axis=0)
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
                    time_idx=0,
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
            prev_actions = self.make_action_rep(action[np.newaxis, ...])
            timesteps = [
                Timestep(
                    obs=obs,
                    prev_action=prev_actions[0],
                    reward=rewards,
                    terminal=terminateds,
                    time_idx=self.step_count[0].item(),
                )
            ]
            rewards = np.array([rewards], dtype=np.float32)
            terminateds = np.array([terminateds], dtype=bool)
            truncateds = np.array([truncateds], dtype=bool)
        else:
            prev_actions = self.make_action_rep(action)
            # unstack to avoid indexing arrays during `Timestep` creation
            _terminals = np.unstack(terminateds, axis=0)
            _obs = unstack_dict(obs)
            _rewards = np.unstack(rewards, axis=0)
            _prev_actions = np.unstack(prev_actions, axis=0)
            _time_idxs = np.unstack(self.step_count, axis=0)
            timesteps = []
            for idx in range(self.batched_envs):
                # we can end up with random jax/np datatypes here...
                timesteps.append(
                    Timestep(
                        obs=_obs[idx],
                        prev_action=_prev_actions[idx],
                        reward=_rewards[idx],
                        terminal=_terminals[idx],
                        time_idx=_time_idxs[idx],
                    )
                )

        return timesteps, rewards, terminateds, truncateds, infos


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
        env,
        save_every: tuple[int, int] | None = None,
        make_dset: bool = False,
        dset_root: str = None,
        dset_name: str = None,
        dset_split: str = None,
        save_trajs_as: str = "trajectory",
    ):
        super().__init__(env)

        self.batched_envs = env.batched_envs
        self.make_dset = make_dset
        if make_dset:
            assert dset_root is not None
            assert dset_name is not None
            assert dset_split in ["train", "val", "test"]
            self.dset_write_dir = os.path.join(dset_root, dset_name, dset_split)
            if not os.path.exists(self.dset_write_dir):
                os.makedirs(self.dset_write_dir)
        else:
            self.dset_write_dir = None
        self.dset_root = dset_root
        self.dset_name = dset_name
        self.dset_split = dset_split
        self.save_every = save_every
        self.since_last_save = 0
        self.save_trajs_as = save_trajs_as
        self._total_frames = 0
        self._total_frames_by_env_name = defaultdict(int)
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

    def reset(self, seed=None) -> Timestep:
        timestep, info = self.env.reset(seed=seed)
        assert len(timestep) == self.batched_envs
        self.active_trajs = [Trajectory(timesteps=[t]) for t in timestep]
        self.since_last_save = [0 for _ in range(self.batched_envs)]
        self.save_this_time = [
            random.randint(*self.save_every) if self.save_every else None
            for _ in range(self.batched_envs)
        ]
        self.total_return = np.zeros(self.batched_envs, dtype=np.float64)
        self._current_timestep = [
            t.make_sequence(last_only=True) for t in self.active_trajs
        ]
        return stack_list_array_dicts([t.obs for t in timestep]), info

    def step(self, action):
        timestep, reward, terminated, truncated, info = self.env.step(action)
        assert len(timestep) == self.batched_envs
        assert terminated.shape[0] == self.batched_envs
        assert truncated.shape[0] == self.batched_envs
        assert reward.shape[0] == self.batched_envs

        self.total_return += reward
        done = np.logical_or(terminated, truncated)
        for idx in range(self.batched_envs):
            self.active_trajs[idx].add_timestep(timestep[idx])
            self.since_last_save[idx] += 1
            if done[idx]:
                self.return_history.add_score(self.env.env_name, self.total_return[idx])
            save = (
                self.save_every is not None
                and self.since_last_save[idx] > self.save_this_time[idx]
            )
            if (done[idx] or save) and self.make_dset:
                self.log_to_disk(idx=idx)
                self.active_trajs[idx] = Trajectory(timesteps=[timestep[idx]])

        for info_key, info_val in info.items():
            if info_key.startswith(self.special_history.log_prefix):
                self.special_history.add_score(self.env.env_name, info_key, info_val)

        self._current_timestep = [
            t.make_sequence(last_only=True) for t in self.active_trajs
        ]
        self._total_frames += len(timestep)
        self._total_frames_by_env_name[self.env.env_name] += len(timestep)
        return (
            stack_list_array_dicts([t.obs for t in timestep]),
            reward,
            terminated,
            truncated,
            info,
        )

    def log_to_disk(self, idx: int):
        traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{time.time()}"
        path = os.path.join(self.dset_write_dir, traj_name)
        self.active_trajs[idx].save_to_disk(path, save_as=self.save_trajs_as)
        self.since_last_save[idx] = 0

    def sequence(self):
        return [t.make_sequence() for t in self.active_trajs]

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def total_frames_by_env_name(self) -> dict[str, int]:
        return self._total_frames_by_env_name

    @property
    def current_timestep(self) -> tuple[np.ndarray, np.ndarray]:
        return self._current_timestep


@dataclass
class EnvCreator:
    make_env: Callable
    exploration_wrapper_Cls: Type[ExplorationWrapper]
    make_dset: bool
    dset_root: str
    dset_name: str
    dset_split: str
    save_trajs_as: str
    traj_save_len: int
    max_seq_len: int
    stagger_traj_file_lengths: bool

    def __post_init__(self):
        if self.max_seq_len < self.traj_save_len and self.stagger_traj_file_lengths:
            self.save_every_low = self.traj_save_len - self.max_seq_len
            self.save_every_high = self.traj_save_len + self.max_seq_len
            warnings.warn(
                f"Note: Partial Context Mode. Randomizing trajectory file lengths in [{self.save_every_low}, {self.save_every_high}]"
            )
        else:
            self.save_every_low = self.save_every_high = self.traj_save_len
        self.rl2_space = None
        self.already_vectorized = False

    def __call__(self):
        env = self.make_env()
        self.already_vectorized = env.batched_envs > 1
        if self.exploration_wrapper_Cls is not None:
            env = self.exploration_wrapper_Cls(env)
        env = SequenceWrapper(
            env,
            save_every=(self.save_every_low, self.save_every_high),
            make_dset=self.make_dset,
            dset_root=self.dset_root,
            dset_name=self.dset_name,
            dset_split=self.dset_split,
            save_trajs_as=self.save_trajs_as,
        )
        self.rl2_space = env.rl2_space
        return env
