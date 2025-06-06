"""
Procgen wrapper
"""

import random

import procgen
import gym as og_gym
import numpy as np
import gymnasium as gym

from amago.envs import AMAGOEnv


class ProcgenAMAGO(AMAGOEnv):
    """AMAGOEnv for TwoAttemptMTProcgen that logs metrics for each game separately."""

    def __init__(self, env):
        super().__init__(
            env=env,
            env_name="Procgen",
        )

    @property
    def env_name(self):
        return self.env.current_game


ALL_PROCGEN_GAMES = [
    "dodgeball",
    "caveflyer",
    "heist",
    "jumper",
    "maze",
    "miner",
    "fruitbot",
    "plunder",
    "chaser",
    "leaper",
    "bigfish",
    "starpilot",
    "bossfight",
    "ninja",
    "coinrun",
    "climber",
]


class TwoAttemptMTProcgen(gym.Env):
    """A Multi-Task Procgen environment that gives two attempts at each
    level.

    Args:
        games: A list of Procgen game names to include (e.g. ["coinrun",
            "dodgeball"]).
        distribution_mode: The distribution mode to use for the environment.
            Options are:
            - "easy": Standard procgen easy mode for every game.
            - "hard": Standard procgen hard mode for every game.
            - "memory-hard": Memory mode in games it is available, hard mode
              otherwise.
        reward_scales: A dictionary mapping game names to multipliers that
            scale their rewards (e.g., {"coinrun": 10.0, "dodgeball": 0.5}).
        seed_range: A tuple of integers representing the range of seeds to use
            for the environment. For train/test splits.
    """

    def __init__(
        self,
        games: list[str],
        distribution_mode: str,
        reward_scales: dict[str, float] = {},
        seed_range: tuple[int, int] = (0, 2000),
    ):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(15)
        self.seed_range = seed_range
        assert distribution_mode in ["easy", "hard", "memory-hard"]
        self.distribution_mode = distribution_mode
        for game in games:
            assert (
                game in ALL_PROCGEN_GAMES
            ), f"Invalid Procgen game `{game}`. Options are: {ALL_PROCGEN_GAMES}"
        self.games = games
        self.reward_scales = reward_scales
        self.env = None
        self.reset()

    def frame(self, frame):
        if self._current_episode > 0:
            # paint a small box on the screen to indicate the last episode.
            # means that RL^2 reset flag is unnecessary and resolves value
            # ambiguity w/ short context lengths.
            frame[1:5, 1:5, :] = 0
        return frame

    def _reset_current_env(self):
        if self.env is not None:
            self.env.close()

        if self.distribution_mode == "memory-hard":
            if self.current_game in [
                "dodgeball",
                "caveflyer",
                "heist",
                "jumper",
                "maze",
                "miner",
            ]:
                distribution = "memory"
            else:
                distribution = "hard"
        else:
            distribution = self.distribution_mode

        self.env = og_gym.make(
            f"procgen:procgen-{self.current_game}-v0",
            num_levels=1,
            distribution_mode=distribution,
            use_sequential_levels=False,
            start_level=self.current_level,
        )
        self._reset_next_step = False

    def reset(self, *args, **kwargs):
        self._current_episode = 0
        self.current_game = random.choice(self.games)
        self.current_level = random.randint(*self.seed_range)
        self._reset_current_env()
        obs = self.env.reset()
        return self.frame(obs), {"game": self.current_game, "level": self.current_level}

    def step(self, action):
        if self._reset_next_step:
            self._reset_current_env()
            next_obs = self.env.reset()
            reward, done, info = 0.0, False, {"soft_reset": True}
        else:
            next_obs, reward, done, info = self.env.step(action)

        actually_done = False
        if done:
            self._current_episode += 1
            actually_done = self._current_episode >= 2
            if not actually_done:
                self._reset_next_step = True

        if self.current_game in self.reward_scales:
            reward = self.reward_scales[self.current_game] * reward

        return self.frame(next_obs), reward, actually_done, actually_done, info
