import random
import warnings


import gymnasium as gym

try:
    import dm_env
    from dm_env import specs
    import dm_alchemy
    from dm_alchemy import symbolic_alchemy
except ImportError:
    warnings.warn("Missing dm_env / dm_alchemy Install: `pip install amago[envs]`")


class GymFromDMEnv(gym.Env):
    """
    This code is from:
    https://github.com/google-deepmind/bsuite/blob/main/bsuite/utils/gym_wrapper.py
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env: dm_env.Environment, obs_key):
        self._env = env  # type: dm_env.Environment
        self._last_observation = None  # type: Optional[np.ndarray]
        self.viewer = None
        self.game_over = False  # Needed for Dopamine agents.
        self.obs_key = obs_key

    def step(self, action):
        timestep = self._env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.0
        terminated = truncated = timestep.last()
        if terminated or truncated:
            self.game_over = True
        return timestep.observation[self.obs_key], reward, terminated, truncated, {}

    def reset(self, *args, **kwargs):
        self.game_over = False
        timestep = self._env.reset()
        self._last_observation = timestep.observation
        return timestep.observation[self.obs_key], {}

    def render(self, mode: str = "rgb_array"):
        if self._last_observation is None:
            raise ValueError("Environment not ready to render. Call reset() first.")

        if mode == "rgb_array":
            return self._last_observation

        if mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._last_observation)
            return self.viewer.isopen

    @property
    def action_space(self) -> gym.spaces.Discrete:
        action_spec = self._env.action_spec()
        return gym.spaces.Discrete(action_spec.maximum - action_spec.minimum + 1)

    @property
    def observation_space(self) -> gym.spaces.Box:
        obs_spec = self._env.observation_spec()[self.obs_key]
        if isinstance(obs_spec, specs.BoundedArray):
            return gym.spaces.Box(
                low=float(obs_spec.minimum),
                high=float(obs_spec.maximum),
                shape=obs_spec.shape,
                dtype=obs_spec.dtype,
            )
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype,
        )

    @property
    def reward_range(self):
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float("inf"), float("inf")

    def __getattr__(self, attr):
        return getattr(self._env, attr)


class SymbolicAlchemy(gym.Wrapper):
    def __init__(self):
        super().__init__(self.init_new_env())

    def init_new_env(self):
        level_name = (
            "alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
        )
        env = symbolic_alchemy.get_symbolic_alchemy_level(
            level_name, seed=random.randint(0, 1_000_000)
        )
        env = GymFromDMEnv(env, "symbolic_obs")
        return env

    def reset(self, *args, **kwargs):
        self._alchemy_env = self.init_new_env()
        state, info = self._alchemy_env.reset()
        return state, info

    def step(self, action):
        return self._alchemy_env.step(action)


if __name__ == "__main__":
    env = SymbolicAlchemy()

    returns = []
    for ep in range(100):
        env.reset()
        return_ = 0.0
        for i in range(1000):
            next_state, rew, te, tr, info = env.step(env.action_space.sample())
            return_ += rew
            if te or tr:
                break
        returns.append(return_)
    print(sum(returns) / len(returns))
