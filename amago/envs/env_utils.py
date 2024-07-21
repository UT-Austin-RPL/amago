import os
import time
import random
import warnings
from uuid import uuid4
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Type, Callable, Iterable

import gymnasium as gym
import numpy as np
import torch
import gin
from einops import rearrange

from amago.loading import MAGIC_PAD_VAL
from amago.hindsight import Timestep, Trajectory, GoalSeq
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


@gin.configurable
class ExplorationWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        eps_start_start: float = 1.0,
        eps_start_end: float = 0.05,
        eps_end_start: float = 0.8,
        eps_end_end: float = 0.01,
        steps_anneal: int = 1_000_000,
    ):
        super().__init__(env)

        self.eps_start_start = eps_start_start
        self.eps_start_end = eps_start_end
        self.eps_end_start = eps_end_start
        self.eps_end_end = eps_end_end
        self.global_slope = (eps_start_start - eps_start_end) / steps_anneal
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.multibinary = isinstance(self.env.action_space, gym.spaces.MultiBinary)
        self.global_step = 0
        self.disabled = False

    def turn_off_exploration(self):
        self.disabled = True

    def turn_on_exploration(self):
        self.disabled = False

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.global_multiplier = random.random()
        np.random.seed(random.randint(0, 1e6))
        return out

    def current_eps(self, local_step: int, horizon: int):
        ep_start = max(
            self.eps_start_start - self.global_slope * self.global_step,
            self.eps_start_end,
        )
        ep_end = max(
            self.eps_start_end - self.global_slope * self.global_step, self.eps_end_end
        )
        local_progress = float(local_step) / horizon
        current = self.global_multiplier * (
            ep_start - ((ep_start - ep_end) * local_progress)
        )
        return current

    def action(self, a):
        noise = (
            self.current_eps(self.env.step_count, self.env.horizon)
            if not self.disabled
            else 0.0
        )
        if self.discrete:
            # epsilon greedy (DQN-style)
            num_actions = self.env.action_space.n
            random_action = random.randrange(0, num_actions)
            use_random = random.random() <= noise
            if use_random:
                expl_action = np.full_like(a, random_action)
            else:
                expl_action = a
            assert expl_action.dtype == np.uint8
        elif self.multibinary:
            random_actions = np.random.randn(*a.shape) < 0.0
            use_random = np.random.random(*a.shape) <= noise
            expl_action = (1 - use_random) * a + use_random * random_actions
            expl_action = expl_action.astype(np.int8)
        else:
            # random noise (TD3-style)
            expl_action = a + noise * np.random.randn(*a.shape)
            expl_action = np.clip(expl_action, -1.0, 1.0).astype(np.float32)
            assert expl_action.dtype == np.float32

        if not self.disabled:
            self.global_step += 1
        return expl_action

    @property
    def return_history(self):
        return self.env.return_history

    @property
    def success_history(self):
        return self.env.success_history


class ReturnHistory:
    def __init__(self, env_name):
        self.data = {}

    def add_score(self, env_name, score):
        if env_name in self.data:
            self.data[env_name].append(score)
        else:
            self.data[env_name] = [score]


SuccessHistory = ReturnHistory


class SequenceWrapper(gym.Wrapper):
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
        rl2_shape = action_shape + 3  # action + reward + done + time
        self.gcrl2_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "goal": self.env.kgoal_space,
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

    @property
    def horizon(self):
        return self.env.horizon

    def reset_stats(self):
        # stores all of the success/return histories
        self.return_history = ReturnHistory(self.env_name)
        self.success_history = SuccessHistory(self.env_name)

    def reset(self, seed=None) -> Timestep:
        timestep = self.env.reset(seed=seed)
        self.active_traj = Trajectory(
            max_goals=self.env.max_goal_seq_length, timesteps=[timestep]
        )
        self.since_last_save = 0
        self.save_this_time = (
            random.randint(*self.save_every) if self.save_every else None
        )
        self.total_return = 0.0
        self._current_timestep = self.active_traj.make_sequence(last_only=True)
        return timestep.obs, {}

    def step(self, action):
        timestep, reward, terminated, truncated, info = self.env.step(action)
        self.total_return += reward
        self.active_traj.add_timestep(timestep)
        self.since_last_save += 1
        if timestep.terminal:
            self.return_history.add_score(self.env.env_name, self.total_return)
            success = (
                self.active_traj.is_success
                if "success" not in info
                else info["success"]
            )
            self.success_history.add_score(self.env.env_name, success)
        save = (
            self.save_every is not None and self.since_last_save > self.save_this_time
        )
        if (timestep.terminal or save) and self.make_dset:
            self.log_to_disk()
            self.active_traj = Trajectory(
                max_goals=self.env.max_goal_seq_length, timesteps=[timestep]
            )
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

    def turn_off_exploration(self):
        if isinstance(self.env, ExplorationWrapper):
            self.env.turn_off_exploration()

    def turn_on_exploration(self):
        if isinstance(self.env, ExplorationWrapper):
            self.env.turn_on_exploration()

    @property
    def current_timestep(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._current_timestep


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
        self.gcrl2_space = None

    def __call__(self):
        env = self.make_env()
        self.horizon = max(self.horizon, env.horizon)
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
        self.gcrl2_space = env.gcrl2_space
        return env
