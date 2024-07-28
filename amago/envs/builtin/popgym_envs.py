import random

import gymnasium as gym
import numpy as np
import popgym
from popgym.wrappers import Flatten, DiscreteAction

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


class POPGymAMAGO(GymEnv):
    def __init__(self, env_name: str):
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


class MultiDomainPOPGym(gym.Env):
    mt_names = [
        # env name, observation dim, action dim
        "AutoencodeEasy",  # 4, 6
        "AutoencodeMedium",  # 4, 6
        "AutoencodeHard",  # 4, 6
        "RepeatPreviousEasy",  # 4, 4
        "RepeatPreviousMedium",  # 4, 4
        "RepeatPreviousHard",  # 4, 4
        "RepeatFirstEasy",  # 4, 4
        "RepeatFirstMedium",  # 4, 4
        "RepeatFirstHard",  # 4, 4
        "CountRecallEasy",  # 26, 4
        "CountRecallMedium",  # 26, 8
        "CountRecallHard",  # 16, 26
        "PositionOnlyCartPoleEasy",  # 2, 2
        "PositionOnlyCartPoleMedium",  # 2, 2
        "PositionOnlyCartPoleHard",  # 2, 2
        "VelocityOnlyCartpoleEasy",  # 2, 2
        "VelocityOnlyCartpoleMedium",  # 2, 2
        "VelocityOnlyCartpoleHard",  # 2, 2
        "NoisyPositionOnlyCartPoleEasy",  # 2, 2
        "NoisyPositionOnlyCartPoleMedium",  # 2, 2
        "NoisyPositionOnlyCartPoleHard",  # 2, 2
        "MultiarmedBanditEasy",  # 10, 2
        "MultiarmedBanditMedium",  # 20, 2
        "HigherLowerEasy",  # 2, 13
        "HigherLowerMedium",  # 2, 13
        "HigherLowerHard",  # 2, 13
        "MineSweeperEasy",  # 16, 3
    ]

    def __init__(self, warmup_episodes: int = 1):
        self.warmup_episodes = warmup_episodes

        str_to_cls = {v["id"]: k for k, v in popgym.envs.ALL.items()}
        self.name_to_env = {}
        for cls, id_dict in popgym.envs.ALL.items():
            raw_name = id_dict["id"].split("-")[1]
            if raw_name in self.mt_names:
                env = cls()
                env = Flatten(env)
                if isinstance(
                    env.action_space, gym.spaces.Discrete | gym.spaces.MultiDiscrete
                ):
                    env = DiscreteAction(env)
                if isinstance(env.observation_space, gym.spaces.Discrete):
                    env = _DiscreteToBox(env)
                elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
                    env = _MultiDiscreteToBox(env)
                self.name_to_env[raw_name] = env
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(26 + 3,))
        self.action_space = gym.spaces.Discrete(26)
        self.reset()

    def reset(self, *args, **kwargs):
        env_name = random.choice(self.mt_names)
        self.current_env_name = env_name
        self.current_env = self.name_to_env[env_name]
        self.current_action_size = self.current_env.action_space.n
        self.current_episode = 0
        self.reset_next_step = False
        raw_obs, info = self.current_env.reset()
        obs = self.make_obs(raw_obs, valid_action=True, reward=0.0)
        return obs, info

    def make_obs(self, raw_obs, valid_action: bool, reward: float):
        obs = np.zeros((29,), dtype=np.float32)
        obs[: len(raw_obs)] = raw_obs
        obs[-3] = self.current_episode / self.warmup_episodes
        obs[-2] = valid_action
        obs[-1] = reward
        return obs

    def step(self, action):
        valid_action = action < self.current_action_size
        if not valid_action:
            action = random.randrange(0, self.current_action_size)

        if self.reset_next_step:
            next_raw_obs, info = self.current_env.reset()
            self.reset_next_step = False
            self.current_episode += 1
            reward = 0.0
        else:
            next_raw_obs, reward, terminal, truncated, info = self.current_env.step(
                action
            )
            done = terminal | truncated
            if done:
                self.reset_next_step = True

        next_obs = self.make_obs(next_raw_obs, valid_action=valid_action, reward=reward)
        shown_done = self.current_episode > self.warmup_episodes
        shown_reward = 0.0 if self.current_episode < self.warmup_episodes else reward
        return next_obs, shown_reward, shown_done, False, info


class MultiDomainPOPGymAMAGO(GymEnv):
    def __init__(self, warmup_episodes: int = 1):
        env = MultiDomainPOPGym(warmup_episodes)
        super().__init__(env, env_name="TODO", horizon=10_000, start=0, zero_shot=True)

    @property
    def env_name(self):
        return f"MultiDomainPOPGym-{self.env.current_env_name}"
