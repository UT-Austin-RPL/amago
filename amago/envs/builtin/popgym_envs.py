import gymnasium as gym
import numpy as np

from amago.envs.builtin.gym_envs import GymEnv


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
