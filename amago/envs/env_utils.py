import gymnasium as gym
import numpy as np
import torch

from amago.loading import MAGIC_PAD_VAL


class AlreadyVectorizedEnv(gym.Env):
    """
    Thin wrapper imitating the Async calls of a single environment
    that already has a batch dimension.

    Important: assumes the vectorized environment is handling
    automatic resets.
    """

    def __init__(self, env_funcs):
        assert len(env_funcs) == 1
        self.env = env_funcs[0]()
        self.batched_envs = self.env.batched_envs
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.single_action_space = self.action_space
        self._call_buffer = None

    def reset(self, *args, **kwargs):
        outs = self.env.reset()
        return outs

    def call_async(self, prop):
        try:
            result = eval(f"self.env.{prop}()")
        except:
            result = eval(f"self.env.{prop}")
        if isinstance(result, list | tuple | np.ndarray):
            # imitate `batched_envs` envs each with a batch dim of 1
            self._call_buffer = [[r] for r in result]
        else:
            self._call_buffer = [result]

    def call_wait(self):
        return self._call_buffer

    def render(self):
        return self.env.render()

    def step(self, action):
        assert action.shape[0] == self.batched_envs
        obs, rewards, te, tr, info = self.env.step(action)
        obs = np.stack(obs, axis=0)
        return obs, rewards, te, tr, info


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
        elif isinstance(action, np.ndarray) and action.ndim == 1:
            return int(action[0])
        elif isinstance(action, np.ndarray) and action.ndim == 2:
            return action[:, 0]
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
        l = max(self.cur_idxs)
        out = {}
        for k in self.buffers[0].keys():
            out[k] = torch.stack([buffer[k] for buffer in self.buffers], axis=0)[:, :l]
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
