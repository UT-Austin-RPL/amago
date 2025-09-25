"""
Miscellaneous environment wrappers and utilities.
"""

import gymnasium as gym
import numpy as np


def extend_box_obs_space_by(
    space: gym.spaces.Box, by: int, low: float, high: float
) -> gym.spaces.Box:
    """Utility for adding dimensions to gym.spaces.Box spaces when concatenating extra info like terminal signals and time remaining."""
    assert isinstance(space, gym.spaces.Box)
    return gym.spaces.Box(
        shape=(space.shape[0] + by,),
        low=np.concatenate((space.low, (low,) * by), axis=-1),
        high=np.concatenate((space.high, (high,) * by), axis=-1),
    )


class AlreadyVectorizedEnv(gym.Env):
    """Imitates Async calls of a single environment that already has a batch dimension.

    Important: assumes the vectorized environment is handling automatic resets.
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
    """Imitates Async calls of synchronous parallel environments."""

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
    """Converts original `gym` action spaces to their `gymnasium` equivalents so that they pass type checks.

    Args:
        gym_space: The original `gym` space to convert.

    Returns:
        The converted `gymnasium` space.
    """
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
    """Messy numpy/int squeeze/unsqueeze compatability

    So that the Agent can output discrete actions in any reasonable format.
    AMAGOEnv automatically adds this wrapper to Discrete action spaces.
    """

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def action(self, action):
        if isinstance(action, int):
            return action
        elif isinstance(action, np.ndarray) and action.ndim == 1:
            assert action.shape[0] == 1 and action.dtype == np.uint8
            return action[0].item()
        elif isinstance(action, np.ndarray) and action.ndim == 2:
            assert action.shape[1] == 1
            return np.squeeze(action, axis=1)
        return action


class ContinuousActionWrapper(gym.ActionWrapper):
    """Normalize continuous action spaces [-1, 1]

    AMAGOEnv automatically adds this wrapper to continuous/Box action spaces.
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
        assert abs(action).max() <= 1.0, "Continuous action out of [-1, 1] bound"
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action


class MultiBinaryActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action.astype(np.int8)
    
class MultiDiscreteActionWrapper(gym.ActionWrapper):
    def action(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 2:
            return np.squeeze(action, axis=0) if action.shape[0] == 1 else action
        return action.astype(np.int32)
