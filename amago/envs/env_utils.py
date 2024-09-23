import os
import time
import random
import warnings
from abc import ABC, abstractmethod
from uuid import uuid4
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Type, Callable, Iterable, Any

import gymnasium as gym
import numpy as np
import torch
import gin
from einops import rearrange

from amago.loading import MAGIC_PAD_VAL
from amago.hindsight import Timestep, Trajectory
from amago.utils import amago_warning


class DummyAsyncVectorEnv(gym.Env):
    def __init__(self, env_funcs):
        self.envs = [e() for e in env_funcs]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.single_action_space = self.action_space
        self._call_buffer = None

    def reset(self, *args, **kwargs):
        outs = [e.reset() for e in self.envs]
        return np.stack([o[0] for o in outs], axis=0), [o[1] for o in outs]

    def call_async(self, prop):
        try:
            self._call_buffer = [eval(f"e.{prop}()") for e in self.envs]
        except:
            self._call_buffer = [eval(f"e.{prop}") for e in self.envs]

    def call_wait(self):
        return self._call_buffer

    def render(self):
        return self.envs[0].render()

    def step(self, action):
        assert action.shape[0] == len(self.envs)
        outs = []
        for i in range(len(self.envs)):
            outs.append(self.envs[i].step(action[i]))
        states = [o[0] for o in outs]
        rewards = np.stack([o[1] for o in outs], axis=0)
        te = [o[2] for o in outs]
        tr = [o[3] for o in outs]
        info = [o[4] for o in outs]

        for i, (terminal, truncated) in enumerate(zip(te, tr)):
            if terminal or truncated:
                states[i], info[i] = self.envs[i].reset()
        te = np.stack(te, axis=0)
        tr = np.stack(tr, axis=0)
        states = np.stack(states, axis=0)

        return states, rewards, te, tr, info


def space_convert(gym_space):
    import gym as og_gym

    if isinstance(gym_space, og_gym.spaces.Box):
        return gym.spaces.Box(
            shape=gym_space.shape, low=gym_space.low, high=gym_space.high
        )
    elif isinstance(gym_space, og_gym.spaces.Discrete):
        return gym.spaces.Discrete(gym_space.n)
    elif isinstance(gym_space, gym.spaces.Space):
        return gym_space
    else:
        raise TypeError(f"Unsupported original gym space `{type(gym_space)}`")


class DiscreteActionWrapper(gym.ActionWrapper):
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def action(self, action):
        if isinstance(action, int):
            return action
        if len(action.shape) > 0:
            action = action[0]
        action = int(action)
        return action


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Normalize continuous action spaces [-1, 1]
    """

    def __init__(self, env):
        super().__init__(env)
        self._true_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32,
        )

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def action(self, action):
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action


class MultiBinaryActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action.astype(np.int8)


class GPUSequenceBuffer:
    def __init__(self, device, max_len: int, num_parallel: int):
        self.device = device
        self.max_len = max_len
        self.num_parallel = num_parallel
        self.buffers = [None for _ in range(num_parallel)]
        self.cur_idxs = [0 for _ in range(num_parallel)]
        self.time_start = [0 for _ in range(num_parallel)]
        self.time_end = [1 for _ in range(num_parallel)]
        self._time_idx_buffer = torch.zeros(
            (num_parallel, max_len), device=self.device, dtype=torch.long
        )

    def _make_blank_buffer(self, array: np.ndarray):
        shape = (self.max_len,) + array.shape[1:]
        return torch.full(shape, MAGIC_PAD_VAL).to(
            dtype=array.dtype, device=self.device
        )

    def add_timestep(self, arrays: np.ndarray | dict[np.ndarray], dones=None):
        if not isinstance(arrays, dict):
            arrays = {"_": arrays}

        for k in arrays.keys():
            v = torch.from_numpy(arrays[k]).to(self.device)
            assert v.shape[0] == self.num_parallel
            assert v.shape[1] == 1
            arrays[k] = v

        if dones is None:
            dones = [False for _ in range(self.num_parallel)]

        for i in range(self.num_parallel):
            self.time_end[i] += 1
            if dones[i] or self.buffers[i] is None:
                self.buffers[i] = {
                    k: self._make_blank_buffer(arrays[k][i]) for k in arrays
                }
                self.cur_idxs[i] = 0
                self.time_start[i] = 0
                self.time_end[i] = 1
            if self.cur_idxs[i] < self.max_len:
                for k in arrays.keys():
                    self.buffers[i][k][self.cur_idxs[i]] = arrays[k][i]
                self.cur_idxs[i] += 1
            else:
                self.time_start[i] += 1
                for k in arrays.keys():
                    self.buffers[i][k] = torch.cat(
                        (self.buffers[i][k], arrays[k][i]), axis=0
                    )[-self.max_len :]

    @property
    def sequences(self):
        longest = max(self.cur_idxs)
        out = {}
        for k in self.buffers[0].keys():
            out[k] = torch.stack([buffer[k] for buffer in self.buffers], axis=0)[
                :, :longest
            ]
        return out

    @property
    def time_idxs(self):
        longest = max(self.cur_idxs)
        time_intervals = zip(self.time_start, self.time_end)
        for i, interval in enumerate(time_intervals):
            arange = torch.arange(*interval)
            self._time_idx_buffer[i, : len(arange)] = arange
        out = self._time_idx_buffer[:, :longest]
        return out

    @property
    def sequence_lengths(self):
        # shaped for Batch, Length, Actions
        return torch.Tensor(self.cur_idxs).to(self.device).view(-1, 1, 1).long()


class ExplorationWrapper(ABC, gym.ActionWrapper):

    def __init__(self, amago_env):
        super().__init__(amago_env)
        self.batched_envs = amago_env.batched_envs

    @abstractmethod
    def add_exploration_noise(self, action: np.ndarray, local_step: int):
        raise NotImplementedError

    def action(self, a: np.ndarray):
        assert a.shape[0] == self.batched_envs
        action = self.add_exploration_noise(a, self.env.step_count)
        return action

    @property
    def return_history(self):
        return self.env.return_history

    @property
    def success_history(self):
        return self.env.success_history


@gin.configurable
class BilevelEpsilonGreedy(ExplorationWrapper):
    """
    Implements the bi-level epsilon greedy exploration strategy visualized in Figure 13
    of the AMAGO paper. Exploration noise decays both over the course of training and
    throughout each rollout. This more closely resembles the online exploration/exploitation
    strategy of a meta-RL agent in a new environment.
    """

    def __init__(
        self,
        amago_env,
        rollout_horizon: int,
        eps_start_start: float = 1.0,  # start of training, start of rollout
        eps_start_end: float = 0.05,  # end of training, start of rollout
        eps_end_start: float = 0.8,  # start of training, end of rollout
        eps_end_end: float = 0.01,  # end of training, end of rollout
        steps_anneal: int = 1_000_000,
    ):
        super().__init__(amago_env)
        self.eps_start_start = eps_start_start
        self.eps_start_end = eps_start_end
        self.eps_end_start = eps_end_start
        self.eps_end_end = eps_end_end
        self.rollout_horizon = rollout_horizon

        self.start_global_slope = (eps_start_start - eps_start_end) / steps_anneal
        self.end_global_slope = (eps_end_start - eps_end_end) / steps_anneal
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.global_step = 0
        self.global_multiplier = 1.0

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        np.random.seed(random.randint(0, 1e6))
        # self.global_multiplier = np.random.rand(self.parallel_envs)
        self.global_multiplier = np.ones(self.batched_envs)
        return out

    def current_eps(self, local_step: np.ndarray):
        ep_start = max(
            self.eps_start_start - self.start_global_slope * self.global_step,
            self.eps_start_end,
        )
        ep_end = max(
            self.eps_end_start - self.end_global_slope * self.global_step,
            self.eps_end_end,
        )
        local_progress = local_step / self.rollout_horizon
        current = self.global_multiplier * (
            ep_start - ((ep_start - ep_end) * local_progress)
        )
        return current

    def add_exploration_noise(self, action: np.ndarray, local_step: np.ndarray):
        assert action.shape[0] == self.batched_envs
        assert local_step.shape[0] == self.batched_envs

        noise = self.current_eps(local_step)
        if self.discrete:
            # epsilon greedy (DQN-style)
            num_actions = self.env.action_space.n
            random_action = np.random.randint(0, num_actions, size=(self.batched_envs,))
            use_random = np.random.rand(self.batched_envs) <= noise
            if use_random:
                expl_action = np.full_like(action, random_action)
            else:
                expl_action = action
            assert expl_action.dtype == np.uint8
        else:
            # random noise (TD3-style)
            expl_action = action + noise * np.random.randn(*action.shape)
            expl_action = np.clip(expl_action, -1.0, 1.0).astype(np.float32)
            assert expl_action.dtype == np.float32
        self.global_step += 1

        assert expl_action.shape[0] == self.batched_envs
        return expl_action


@gin.configurable
class EpsilonGreedy(BilevelEpsilonGreedy):
    """
    Sets the parameters of the BilevelEpsilonGreedy wrapper to be equivalent to standard epsilon-greedy.
    """

    def __init__(
        self,
        amago_env,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        steps_anneal: int = 1_000_000,
    ):
        super().__init__(
            amago_env,
            rollout_horizon=float("inf"),
            eps_start_start=eps_start,
            eps_start_end=eps_end,
            eps_end_start=eps_start,
            eps_end_end=eps_end,
            steps_anneal=steps_anneal,
        )


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
            key = key[len(self.log_prefix) :].strip()
        if env_name not in self.data:
            self.data[env_name] = {}
        if key not in self.data[env_name]:
            self.data[env_name][key] = [value]
        else:
            self.data[env_name][key].append(value)


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
        self.active_traj = Trajectory(timesteps=[timestep])
        self.since_last_save = 0
        self.save_this_time = (
            random.randint(*self.save_every) if self.save_every else None
        )
        self.total_return = 0.0
        self._current_timestep = self.active_traj.make_sequence(last_only=True)
        return timestep.obs, info

    def step(self, action):
        timestep, reward, terminated, truncated, info = self.env.step(action)
        self.total_return += reward
        self.active_traj.add_timestep(timestep)
        self.since_last_save += 1
        for info_key, info_val in info.items():
            if info_key.startswith(self.special_history.log_prefix):
                self.special_history.add_score(self.env.env_name, info_key, info_val)
        if timestep.terminal:
            self.return_history.add_score(self.env.env_name, self.total_return)

        save = (
            self.save_every is not None and self.since_last_save > self.save_this_time
        )
        if (timestep.terminal or save) and self.make_dset:
            self.log_to_disk()
            self.active_traj = Trajectory(timesteps=[timestep])

        self._current_timestep = self.active_traj.make_sequence(last_only=True)
        self._total_frames += 1
        self._total_frames_by_env_name[self.env.env_name] += 1
        return timestep.obs, reward, terminated, truncated, info

    def log_to_disk(self):
        traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{time.time()}"
        path = os.path.join(self.dset_write_dir, traj_name)
        self.active_traj.save_to_disk(
            path,
            save_as=self.save_trajs_as,
        )
        self.since_last_save = 0

    def sequence(self):
        seq = self.active_traj.make_sequence()
        return seq

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def total_frames_by_env_name(self) -> dict[str, int]:
        return self._total_frames_by_env_name

    @property
    def current_timestep(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._current_timestep


class AlreadyVectorizedSequenceWrapper(SequenceWrapper):
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
        super().__init__(
            env,
            save_every=save_every,
            make_dset=make_dset,
            dset_root=dset_root,
            dset_name=dset_name,
            dset_split=dset_split,
            save_trajs_as=save_trajs_as,
        )
        self.batched_envs = env.batched_envs

    def split_timesteps(self, timestep: Timestep, reset: bool) -> tuple[Timestep]:
        breakpoint()
        timesteps = []
        for idx in range(self.batched_envs):
            if reset:
                action = timestep.prev_action
                reward = timestep.reward
            else:
                action = timestep.prev_action[idx]
                reward = timestep.reward[idx]
            timesteps = Timestep(
                obs={k: v[idx] for k, v in timestep.obs.items()},
                prev_action=action,
                reward=reward,
                reset=reset,
                real_reward=timestep.real_reward[idx],
            )

    def split_info(self, info: dict) -> tuple[dict]:
        if info:
            breakpoint()

    def log_to_disk(self):
        traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{time.time()}"
        path = os.path.join(self.dset_write_dir, traj_name)
        self.active_traj.save_to_disk(
            path,
            save_as=self.save_trajs_as,
        )
        self.since_last_save = 0

    def reset(self, seed=None) -> tuple[np.ndarray, list]:
        _timestep, _info = self.env.reset(seed=seed)
        timesteps = self.split_timesteps(_timstep, reset=True)
        infos = self.split_info(_info)
        assert len(timesteps) == self.batched_envs
        self.active_trajs = [
            Trajectory(max_goals=self.env.max_goal_seq_length, timesteps=[t])
            for t in timesteps
        ]
        self.since_last_save = [0 for _ in range(self.batched_envs)]
        self.save_this_time = [
            random.randint(*self.save_every) if self.save_every else None
            for _ in range(self.batched_envs)
        ]
        self.total_return = np.zeros(self.env.batched_envs)
        self._current_timesteps = [
            traj.make_sequence(last_only=True) for traj in self.active_trajs
        ]
        return [tstep.obs for tstep in timesteps], infos

    def sequence(self):
        return [traj.make_sequence() for traj in self.active_trajs]

    def step(self, action):
        breakpoint()
        _timestep, reward, terminated, truncated, _info = self.env.step(action)
        timesteps = self.split_timesteps(_timestep)
        infos = self.split_info(_info)

        self.total_return += reward

        self._current_timesteps = []
        obs = []
        assert len(timesteps) == self.batched_envs
        for idx in range(self.batched_envs):
            self.active_trajs[idx].add_timestep(timesteps[idx])
            self.since_last_save[idx] += 1
            self.total_return[idx] += reward[idx]
            for info_key, info_val in infos[idx].items():
                if info_key.startswith(self.special_history.log_prefix):
                    self.special_history.add_score(
                        self.env.env_name, info_key, info_val
                    )

            if timesteps[idx].terminal:
                self.return_history.add_score(self.env.env_name, self.total_return[idx])
                success = (
                    self.active_trajs[idx].is_success
                    if "success" not in infos[idx]
                    else infos[idx]["success"]
                )
                self.success_history.add_score(self.env.env_name, success)

            save = (
                self.save_every is not None
                and self.since_last_save[idx] > self.save_this_time[idx]
            )
            if timesteps[idx].terminal or save:
                self.log_to_disk(idx=idx)
                self.active_trajs[idx] = Trajectory(
                    max_goals=self.env.max_goal_seq_length, timesteps=[timesteps[idx]]
                )
            self._current_timesteps.append(
                self.active_trajs[idx].make_sequence(last_only=True)
            )
            obs.append(timesteps[idx].obs)
        self._total_frames += self.batched_envs
        self._total_frames_by_env_name[self.env.env_name] += self.batched_envs
        return obs, reward, terminated, truncated, infos

    def log_to_disk(self, idx: int):
        traj_name = f"{self.env.env_name.strip().replace('_', '')}_{uuid4().hex[:8]}_{time.time()}"
        path = os.path.join(self.dset_write_dir, traj_name)
        self.active_trajs[idx].save_to_disk(path, save_as=self.save_trajs_as)
        self.since_last_save[idx] = 0


@dataclass
class EnvCreator:
    make_env: Callable
    make_dset: bool
    dset_root: str
    dset_name: str
    dset_split: str
    save_trajs_as: str
    traj_save_len: int
    max_seq_len: int
    stagger_traj_file_lengths: bool
    exploration_wrapper_Cls: Type[ExplorationWrapper]

    def __post_init__(self):
        if self.max_seq_len < self.traj_save_len and self.stagger_traj_file_lengths:
            self.save_every_low = self.traj_save_len - self.max_seq_len
            self.save_every_high = self.traj_save_len + self.max_seq_len
            warnings.warn(
                f"Note: Partial Context Mode. Randomizing trajectory file lengths in [{self.save_every_low}, {self.save_every_high}]"
            )
        else:
            self.save_every_low = self.save_every_high = self.traj_save_len
        self.horizon = -float("inf")
        self.rl2_space = None

    def __call__(self):
        env = self.make_env()
        already_vectorized = env.batched_envs > 1
        if self.exploration_wrapper_Cls is not None:
            env = self.exploration_wrapper_Cls(env)
        SeqWrapper = (
            AlreadyVectorizedSequenceWrapper if already_vectorized else SequenceWrapper
        )
        env = SeqWrapper(
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
