import random
import warnings
import math

from amago.utils import amago_warning

try:
    import retro
except ImportError:
    amago_warning(
        "Missing stable-retro Install: `https://stable-retro.farama.org/getting_started/"
    )
try:
    import cv2
except ImportError:
    amago_warning("Missing cv2 Install: `pip install opencv-python`")
import gymnasium as gym
import numpy as np
from einops import rearrange

from amago.envs import AMAGOEnv


class AtariAMAGOWrapper(AMAGOEnv):
    def __init__(self, env: gym.Env):
        assert isinstance(env, AtariGame | ALE)
        super().__init__(
            env=env,
            env_name="Atari",
        )

    @property
    def env_name(self):
        return self.env.rom_name


class ALEAction:
    str2action = {
        "": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
        "upright": 6,
        "upleft": 7,
        "downright": 8,
        "downleft": 9,
        "upfire": 10,
        "rightfire": 11,
        "leftfire": 12,
        "downfire": 13,
        "uprightfire": 14,
        "upleftfire": 15,
        "downrightfire": 16,
        "downleftfire": 17,
    }

    def __init__(self, up, down, left, right, fire):
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.fire = fire

    def to_discrete(self):
        updown = ""
        if self.up or self.down:
            if self.up and self.down:
                pass
            else:
                updown = "up" if self.up else "down"

        leftright = ""
        if self.left or self.right:
            if self.left and self.right:
                pass
            else:
                leftright = "left" if self.left else "right"

        fire = "fire" if self.fire else ""
        action_str = f"{updown}{leftright}{fire}"
        action = self.str2action[action_str]
        return action


class AtariGame(gym.Env):
    def __init__(
        self,
        game: str,
        resolution: tuple[int, int] = (84, 84),
        time_limit: int = 108_000,
        frame_skip: int = 4,
        channels_last: bool = False,
        use_discrete_actions: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.time_limit = time_limit
        self.frame_skip = frame_skip
        self._env = gym.make(
            game,
            frameskip=frame_skip,
            repeat_action_probability=0.25,
            obs_type="rgb",
            full_action_space=True,
        )
        self.channels_last = channels_last
        self.rom_name = game
        obs_shape = (
            self.resolution + (3,) if self.channels_last else (3,) + self.resolution
        )
        self.use_discrete_actions = use_discrete_actions
        if use_discrete_actions:
            self.action_space = self._env.action_space
        else:
            self.action_space = gym.spaces.MultiBinary(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def screen(self, frame: np.ndarray) -> np.ndarray:
        obs = cv2.resize(
            frame, tuple(reversed(self.resolution)), interpolation=cv2.INTER_AREA
        )
        if not self.channels_last:
            obs = rearrange(obs, "h w c -> c h w")

        return obs

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        self._time = 0
        obs, info = self._env.reset()
        return self.screen(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.use_discrete_actions:
            action = ALEAction(*action.tolist()).to_discrete()
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        self._time += self.frame_skip  # matches frame counter used in standard ALE
        truncated = truncated or self._time >= self.time_limit
        return self.screen(next_obs), reward, terminated, truncated, info


class ALE(gym.Env):
    def __init__(self, games: list[str], use_discrete_actions: bool):
        super().__init__()
        self.games = games
        self.use_discrete_actions = use_discrete_actions
        self.pick_new_game()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def pick_new_game(self):
        game = random.choice(self.games)
        self._env = AtariGame(
            f"ALE/{game}-v5", use_discrete_actions=self.use_discrete_actions
        )
        self.time_limit = self._env.time_limit
        return game

    def reset(self, *args, **kwargs):
        self.rom_name = self.pick_new_game()
        return self._env.reset()

    def step(self, action):
        next_state, reward, terminated, truncated, info = self._env.step(action)
        return next_state, reward, terminated, truncated, info


class RetroAMAGOWrapper(AMAGOEnv):
    def __init__(self, env: gym.Env):
        super().__init__(
            env=env,
            env_name="RetroArcade-placeholder",
        )

    @property
    def env_name(self):
        return self.env.rom_name


class RetroArcade(gym.Env):
    console_cabinets = {
        "Snes": [18, 196, 65],
        "Nes": [191, 40, 17],
        "PCEngine": [232, 211, 19],
        "GameGear": [25, 230, 199],
        "Genesis": [8, 8, 201],
        "GbColor": [124, 49, 189],
        "GbAdvance": [182, 49, 189],
        "GameBoy": [59, 58, 58],
        "Gb": [59, 58, 58],
        "Sms": [255, 255, 255],
        "Atari2600": [219, 157, 147],
    }

    discrete_action_n = {
        "Nes": 36,
        "Snes": 468,
        "GameBoy": 72,
        "Genesis": 126,
        "Sms": 36,
        "Atari2600": 18,
        "Gb": 72,
    }

    def __init__(
        self,
        game_start_dict: dict[str, list[str]],
        resolution: tuple[int, int] = (84, 84),
        time_limit_minutes: int = 20,
        frame_skip: int = 8,
        channels_last: bool = False,
        use_discrete_actions: bool = False,
    ):
        super().__init__()
        self.game_start_dict = game_start_dict
        self.game_start_flat = []
        for game, levels in self.game_start_dict.items():
            for level in levels:
                self.game_start_flat.append((game, level))
        self.resolution = resolution
        self.time_limit = math.ceil(time_limit_minutes * 60.0 * 60 / frame_skip)
        self.frame_skip = frame_skip
        self._env = None
        self.channels_last = channels_last
        obs_shape = (
            self.resolution + (3,) if self.channels_last else (3,) + self.resolution
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.use_discrete_actions = use_discrete_actions
        if not use_discrete_actions:
            self.action_space = gym.spaces.MultiBinary(12)
        else:
            warnings.warn(
                "MultiBinary actions reccomended! Discrete spaces are too large and will break when using multiple consoles",
                UserWarning,
            )
            console = self.game_start_flat[0][0].split("-")[-1]
            self.action_space = gym.spaces.Discrete(self.discrete_action_n[console])
        self.reset()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def screen(self, frame: np.ndarray) -> np.ndarray:
        obs = cv2.resize(
            frame, tuple(reversed(self.resolution)), interpolation=cv2.INTER_AREA
        )
        obs = (obs * self._cabinet_mask) + self._cabinet
        if not self.channels_last:
            obs = rearrange(obs, "h w c -> c h w")
        return obs

    def close(self):
        if self._env is not None:
            self._env.close()

    def reset(self, *args, **kwargs):
        game, start = random.choice(self.game_start_flat)
        self.rom_name = f"{game}_{start.replace('.state', '')}"
        if self._env is not None:
            self._env.close()
        actions = (
            retro.Actions.DISCRETE
            if self.use_discrete_actions
            else retro.Actions.FILTERED
        )
        self._env = retro.make(game=game, state=start, use_restricted_actions=actions)
        self._console = self._env.system
        self._time = 0
        obs, info = self._env.reset()
        self._cabinet_mask = np.zeros(self.resolution + (3,), dtype=np.uint8)
        self._cabinet_mask[2:-2, 2:-2, :] = 1
        self._cabinet = (1 - self._cabinet_mask) * np.array(
            self.console_cabinets[self._console], dtype=np.uint8
        )
        return self.screen(obs), info | {"game": game}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.use_discrete_actions:
            # trim action to size of this console, rest of the values are redundant
            action = action[: self._env.action_space.shape[0]]
        reward = 0.0
        for _ in range(self.frame_skip):
            next_obs, rew_i, terminated, truncated, info = self._env.step(action)
            reward += rew_i
            if terminated or truncated:
                break
        self._time += 1
        truncated = self._time >= self.time_limit
        return self.screen(next_obs), reward, terminated, truncated, info
