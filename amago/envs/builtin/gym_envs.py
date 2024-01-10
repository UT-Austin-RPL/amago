import warnings
import random

import gymnasium as gym
import numpy as np

from amago.envs import AMAGOEnv
from amago.hindsight import GoalSeq


class GymEnv(AMAGOEnv):
    def __init__(
        self,
        gym_env: gym.Env,
        env_name: str,
        horizon: int,
        start: int = 0,
        zero_shot: bool = True,
        convert_from_old_gym: bool = False,
        soft_reset_kwargs={},
    ):
        if convert_from_old_gym:
            gym_env = gym.wrappers.EnvCompatibility(gym_env)

        super().__init__(gym_env, horizon=horizon, start=start)
        self.zero_shot = zero_shot
        self._env_name = env_name
        self.soft_reset_kwargs = soft_reset_kwargs

    @property
    def env_name(self):
        return self._env_name

    @property
    def blank_goal(self):
        return np.zeros((1,), dtype=np.float32)

    @property
    def achieved_goal(self) -> np.ndarray:
        return [self.blank_goal + 1]

    @property
    def kgoal_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=0.0, shape=(1, 1))

    @property
    def goal_sequence(self) -> GoalSeq:
        goal_seq = [self.blank_goal]
        return GoalSeq(seq=goal_seq, active_idx=0)

    def step(self, action):
        return super().step(
            action,
            normal_rl_reward=True,
            normal_rl_reset=self.zero_shot,
            soft_reset_kwargs=self.soft_reset_kwargs,
        )


class _DiscreteToBox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n,), dtype=np.int64
        )

    def observation(self, obs):
        arr = np.zeros((self.n,), dtype=np.int64)
        arr[obs] = 1
        return arr


class _MultiDiscreteToBox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
        self.max_options = env.observation_space.nvec.max()
        self.categories = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.categories * self.max_options,), dtype=np.int64
        )

    def observation(self, obs):
        arr = np.zeros((self.categories, self.max_options), dtype=np.int64)
        for i, o in enumerate(obs):
            arr[i, o] = 1
        return arr.flatten()


class POPGymEnv(GymEnv):
    def __init__(self, env_name: str):
        try:
            import popgym
            from popgym.wrappers import Flatten, DiscreteAction
        except ImportError:
            msg = "Missing POPGym Install: `pip install amago[envs]` or `pip install popgym`"
            print(msg)
            exit()

        str_to_cls = {v["id"]: k for k, v in popgym.envs.ALL.items()}
        env = str_to_cls[env_name]()
        env = Flatten(env)
        if isinstance(env.action_space, gym.spaces.Discrete | gym.spaces.MultiDiscrete):
            env = DiscreteAction(env)
        if isinstance(env.observation_space, gym.spaces.Discrete):
            env = _DiscreteToBox(env)
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            env = _MultiDiscreteToBox(env)
        super().__init__(
            env, env_name=env_name, horizon=10_000, start=0, zero_shot=True
        )


class RandomLunar(gym.Env):
    def __init__(
        self,
        k_shots=2,
    ):
        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.k_shots = k_shots

    def reset(self, *args, **kwargs):
        self.current_gravity = random.uniform(-3.0, -0.1)
        self.current_wind = random.uniform(0.0, 20.0)
        self.current_turbulence = random.uniform(0.0, 2.0)
        self.env = gym.make(
            "LunarLander-v2",
            continuous=True,
            gravity=self.current_gravity,
            enable_wind=True,
            wind_power=self.current_wind,
            turbulence_power=self.current_turbulence,
        )
        self.current_k = 0
        return self.env.reset()

    def step(self, action):
        done = False
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            next_state, info = self.env.reset()
            self.current_k += 1
        if self.current_k >= self.k_shots:
            done = True
        return next_state, reward, done, False, info


from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class RandomFrozenLake(gym.Env):
    def __init__(self, k_shots: int = 10, size: int = 4):
        self.size = size
        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            shape=(size**2 + 1,), low=0.0, high=1.0
        )
        self.k_shots = k_shots

    def reset(self, *args, **kwargs):
        self.env = _DiscreteToBox(
            gym.make(
                "FrozenLake-v1",
                desc=generate_random_map(size=self.size),
                is_slippery=False,
            )
        )
        self.current_k = 0
        state, info = self.env.reset()
        return self.add_reset(state, 1), info

    def add_reset(self, obs: np.ndarray, reset_signal: int):
        return np.concatenate((obs, np.array([reset_signal])), dtype=obs.dtype)

    def step(self, action):
        done = False
        reset_signal = 0
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            next_state, info = self.env.reset()
            reset_signal = 1
            self.current_k += 1
        if self.current_k >= self.k_shots:
            done = True
        return self.add_reset(next_state, reset_signal), reward, done, False, info


if __name__ == "__main__":
    env = POPgymEnv("popgym-ConcentrationEasy-v0")
    env.reset()
