import warnings

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


if __name__ == "__main__":
    env = POPgymEnv("popgym-ConcentrationEasy-v0")
    env.reset()
