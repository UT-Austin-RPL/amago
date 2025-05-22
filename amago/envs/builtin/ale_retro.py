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
import ale_py  # import to register when using mismatched gymnasium/ale versioning for continuous actions
import gymnasium as gym
import numpy as np
from einops import rearrange
import gin

from amago.envs import AMAGOEnv


class _MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    # From https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv
    # Later added to gymnasium 1.0
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        assert (
            env.observation_space.dtype is not None
        ), "No dtype specified for the observation space"
        assert (
            env.observation_space.shape is not None
        ), "No shape defined for the observation space"
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


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


@gin.configurable
class AtariGame(gym.Env):
    """Play a single Atari Game

    Args:
        game: The name of the game to play.

    Keyword Args:
        resolution: image resolution. Default is (84, 84).
        grayscale: Whether to use grayscale frames. Default is
            False.
        time_limit: The time limit of the game in frames. Default
            is 108,000 frames (30 minutes).
        frame_skip: The number of frames to skip between
            observations. Default is 4.
        channels_last: Format images as (H, W, C) if True, or (C,
            H, W) if False. Default is False.
        action_space: The action space to use. One of "discrete",
            "multibinary", or "continuous". Default is "discrete".
        sticky_action_prob: The probability of repeating the
            previous action. Default is 0.25.
        terminal_on_life_loss: Whether to terminate the episode on
            life loss. Default is False.
        version: The version of gymnasium ALE environment to use.
            Default is "v5".
        continuous_action_threshold: A threshold parameter for
            continuous action values to trigger their corresponding
            discrete action. Default is 0.5.
        clip_rewards: Whether to clip rewards to [-1, 1]. Default
            is False.
    """

    def __init__(
        self,
        game: str,
        resolution: tuple[int, int] = (84, 84),
        grayscale: bool = False,
        time_limit: int = 108_000,
        frame_skip: int = 4,
        channels_last: bool = False,
        action_space: str = "discrete",
        sticky_action_prob: float = 0.25,
        terminal_on_life_loss: bool = False,
        version: str = "v5",
        continuous_action_threshold: float = 0.5,
        clip_rewards: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.time_limit = time_limit
        self.frame_skip = frame_skip
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale = grayscale
        self.channels_last = channels_last
        self.clip_rewards = clip_rewards
        self.rom_name = game

        # create environment
        env_kwargs = dict(
            # more standard approach is to grayscale and then let a wrapper handle frame skipping.
            # for color i'm not sure this works and revert to the default from our previous experiments.
            frameskip=1 if grayscale else frame_skip,
            repeat_action_probability=sticky_action_prob,
            obs_type="rgb" if not grayscale else "grayscale",
            full_action_space=True,
        )
        if action_space == "continuous":
            assert (
                ale_py.__version__ >= "0.10"
            ), "pip install --upgrade ale_py==0.10 for continuous actions"
            env_kwargs["continuous"] = True
            env_kwargs["continuous_action_threshold"] = continuous_action_threshold
        self._env = gym.make(f"ALE/{game}-{version}", **env_kwargs)
        if grayscale:
            # grayscaled frameskips can use the MaxAndSkipEnv trick
            self._env = _MaxAndSkipEnv(self._env, skip=frame_skip)

        # set observation space
        channels = 1 if grayscale else 3
        obs_shape = (
            self.resolution + (channels,)
            if channels_last
            else (channels,) + self.resolution
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # set action space
        self.action_space_type = action_space
        if self.action_space_type == "discrete":
            self.action_space = self._env.action_space
        elif self.action_space_type == "multibinary":
            self.action_space = gym.spaces.MultiBinary(5)
        elif self.action_space_type == "continuous":
            self.action_space = self._env.action_space
        else:
            raise ValueError(f"Invalid action space: {self.action_space_type}")

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def screen(self, frame: np.ndarray) -> np.ndarray:
        obs = cv2.resize(
            frame, tuple(reversed(self.resolution)), interpolation=cv2.INTER_AREA
        )
        if self.grayscale:
            obs = np.expand_dims(obs, axis=-1)
        if not self.channels_last:
            obs = rearrange(obs, "h w c -> c h w")
        return obs

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        self._time = 0
        obs, info = self._env.reset()
        self.lives = info["lives"]
        return self.screen(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.action_space_type == "multibinary":
            action = ALEAction(*action.tolist()).to_discrete()
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        self._time += self.frame_skip  # matches frame counter used in standard ALE
        truncated = truncated or self._time >= self.time_limit
        if self.terminal_on_life_loss:
            terminated = terminated or info["lives"] < self.lives
        if self.clip_rewards:
            reward = max(min(reward, 1.0), -1.0)
        return self.screen(next_obs), reward, terminated, truncated, info


class ALE(gym.Env):
    """Play a collection of Atari games in a multi-task setting.

    Creates `AtariGame` envs with the default kwargs. Use `gin` to customize them.

    Args:
        game_names: A list of game names to play (e.g. ["Breakout", "Pong", "Qbert"]).
    """

    def __init__(self, game_names: list[str]):
        super().__init__()
        self.games = game_names
        self.pick_new_game()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def pick_new_game(self):
        game = random.choice(self.games)
        self._env = AtariGame(game)
        self.time_limit = self._env.time_limit
        self.rom_name = self._env.rom_name
        return game

    def reset(self, *args, **kwargs):
        self.rom_name = self.pick_new_game()
        return self._env.reset()

    def step(self, action):
        next_state, reward, terminated, truncated, info = self._env.step(action)
        return next_state, reward, terminated, truncated, info


class AtariAMAGOWrapper(AMAGOEnv):
    """Wraps an AtariGame or ALE environment to be used as an AMAGOEnv.

    Args:
        env: The environment to wrap.
    """

    def __init__(self, env: AtariGame | ALE):
        assert isinstance(env, AtariGame | ALE)
        super().__init__(
            env=env,
            env_name="Atari-placeholder",
        )

    @property
    def env_name(self):
        return self.env.rom_name


class RetroArcade(gym.Env):
    """A multi-task Gym Retro environment.

    On each reset, creates a new Retro game from the level start settings like
    {game : [level1, level2, ...], game2 : [level1, level2, ...], ...}. Each game
    is set up similarly to an Atari environment. Multi-console action spaces are
    padded to a max length. We add a colored border to the edge of the screen
    that distinguishes between different consoles to identify the action space.

    Args:
        game_start_dict: A dictionary mapping game names to lists of level start states.
            Options will depend on the ROMs installed with stable-retro.

    Keyword Args:
        resolution: The resolution of the screen. Default is (84, 84).
        time_limit_minutes: The time limit of the game in minutes of game time.
            Default is 20.
        frame_skip: The number of frames to skip between observations. Default is 8.
        channels_last: Whether to use channels last image format. Default is False.
        use_discrete_actions: Whether to use discrete actions. If True, the env will break
            when using multiple consoles. If False, use a multi-binary action space padded
            to the largest console. Default is False.
    """

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


class RetroAMAGOWrapper(AMAGOEnv):
    """Wraps a RetroArcade environment to be used as an AMAGOEnv.

    Args:
        env: The environment to wrap.
    """

    def __init__(self, env: RetroArcade):
        super().__init__(
            env=env,
            env_name="RetroArcade-placeholder",
        )

    @property
    def env_name(self):
        return self.env.rom_name
