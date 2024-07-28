import warnings
import copy
import random
from typing import Iterable, Tuple, List

import minigrid
import numpy as np
import gymnasium as gym
import cv2

from amago.envs import AMAGOEnv
from amago.hindsight import GoalSeq
from amago.envs.env_utils import space_convert, DiscreteActionWrapper


BANNED_BABYAI_TASKS = [
    "GoToSeq",
    "OpenRedBlueDoorsDebug",
    "OpenTwoDoors",
    "OpenDoorsOrderN4Debug",
    "OpenDoorsOrder",
    "PutNextLocal",
    "PutNext",
    "MiniBossLevel",
    "BossLevel",
    "BossLevelNoUnlock",
    "SynthSeq",
    "SynthLoc",
    "Synth",
    "MoveTwoAcross",
]

ALL_BABYAI_TASKS = [
    env_name
    for env_name in gym.envs.registry.keys()
    if (
        env_name.startswith("BabyAI-")
        and all(banned not in env_name for banned in BANNED_BABYAI_TASKS)
    )
]


class MultitaskMetaBabyAI(gym.Env):
    def __init__(
        self,
        task_names: List[str],
        k_episodes: int = 1,
        seed_range: Tuple[int, int] = (0, 1_000_000),
        observation_type: str = "partial-grid",
    ):
        self.task_names = [task.replace("BabyAI-", "") for task in task_names]
        self.k_episodes = k_episodes
        self.observation_type = observation_type
        self.seed_range = seed_range

        self.action_space = gym.spaces.Discrete(7)
        if observation_type == "partial-grid":
            img_space = gym.spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.int32)
        elif observation_type == "full-grid":
            img_space = gym.spaces.Box(
                low=0, high=255, shape=(22, 22, 3), dtype=np.int32
            )
        elif observation_type == "partial-image":
            img_space = gym.spaces.Box(
                low=0, high=255, shape=(63, 63, 3), dtype=np.uint8
            )
        elif observation_type == "full-image":
            img_space = gym.spaces.Box(
                low=0, high=255, shape=(63, 63, 3), dtype=np.uint8
            )
        else:
            raise ValueError(
                f"Observation type {observation_type} not supported. Supported types are 'partial-grid', 'full-grid', 'partial-image', 'full-image'."
            )
        mission_space = gym.spaces.Box(
            low=0, high=len(self.WORDS) + 2, shape=(9,), dtype=np.int32
        )
        extra_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "image": img_space,
                "mission": mission_space,
                "extra": extra_space,
            }
        )

    def sample_new_env(self, env_name: str):
        env = gym.make(f"BabyAI-{env_name}")
        if self.observation_type == "full-grid":
            env = minigrid.wrappers.FullyObsWrapper(env)
        elif self.observation_type == "partial-image":
            # tile size of 9 to keep image roughly the same size as we'll resize
            # the fully observed version (and so both are close to Atari)
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=9)
        elif self.observation_type == "full-image":
            env = minigrid.wrappers.RGBImgObsWrapper(env)
        # use minigrid wrapper to fix random seed
        seed = random.randint(*self.seed_range)
        env = minigrid.wrappers.ReseedWrapper(env, seeds=(seed,))
        return env

    WORDS = [
        "a",
        "ball",
        "behind",
        "blue",
        "box",
        "door",
        "door,",
        "front",
        "go",
        "green",
        "grey",
        "in",
        "key",
        "left",
        "object",
        "of",
        "on",
        "open",
        "pick",
        "purple",
        "red",
        "right",
        "the",
        "then",
        "to",
        "up",
        "yellow",
        "you",
        "your",
    ]

    def create_observation(self, raw_obs: dict):
        raw = raw_obs["image"]
        if self.observation_type == "partial-grid":
            # default
            img = raw
        elif self.observation_type == "full-grid":
            # pad to 22x22
            x, y, _ = raw.shape
            img = np.zeros((22, 22, 3), dtype=np.int32)
            img[:x, :y] = raw
        elif self.observation_type == "partial-image":
            img = raw
        elif self.observation_type == "full-image":
            img = cv2.resize(raw, (63, 63))

        mission_ids = np.zeros(9, dtype=np.int32)  # 0 means blank
        mission = raw_obs["mission"]
        for idx, word in enumerate(mission.split()):
            if word not in self.WORDS:
                mission_ids[idx] = 1  # 1 means unknown
            else:
                mission_ids[idx] = self.WORDS.index(word) + 2

        extra = np.zeros(5, dtype=np.float32)
        extra[raw_obs["direction"]] = 1.0
        extra[-1] = self.current_episode / self.k_episodes

        return {
            "image": img,
            "mission": mission_ids,
            "extra": extra,
        }

    def reset(self, *args, **kwargs):
        self.current_episode = 1
        self._reset_next_step = False
        self.current_task = random.choice(self.task_names)
        self.env = self.sample_new_env(self.current_task)
        obs, info = self.env.reset()
        return self.create_observation(obs), info

    def step(self, action):
        if self._reset_next_step:
            self._reset_next_step = False
            next_obs, info = self.env.reset()
            return self.create_observation(next_obs), 0, False, False, info

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = False
        if terminated or truncated:
            self.current_episode += 1
            if self.current_episode > self.k_episodes:
                done = True
            else:
                self._reset_next_step = True
        return self.create_observation(next_obs), reward, done, False, info


if __name__ == "__main__":
    env = MultitaskMetaBabyAI(ALL_BABYAI_TASKS)
    b = float("inf")
    for _ in range(100):
        obs, _ = env.reset()
        print(obs["image"].dtype)
        b = min(b, obs["image"].min())
    print(b)
