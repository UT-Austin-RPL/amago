import random
import tqdm
import time

import gymnasium as gym
import numpy as np

try:
    import jax
    import xminigrid
except ImportError:
    jax = None
    xminigrid = None
else:
    from xminigrid.wrappers import DirectionObservationWrapper

from amago.envs.env_utils import AMAGO_ENV_LOG_PREFIX


class XLandMiniGridEnv(gym.Env):
    def __init__(
        self,
        k_shots: int = 25,
        rooms: int = 4,
        grid_size: int = 13,
        ruleset_benchmark: str = "trivial-1m",
        train_test_split: str = "train",
        train_test_split_key: int = 0,
        train_test_split_pct: float = 0.8,
    ):
        assert (
            jax is not None and xminigrid is not None
        ), "JAX and xminigrid must be installed to use this environment"
        self.rooms = rooms
        self.grid_size = grid_size
        self.k_shots = k_shots
        self.ruleset_benchmark = ruleset_benchmark

        benchmark = xminigrid.load_benchmark(name=ruleset_benchmark)
        train, test = benchmark.shuffle(key=jax.random.key(train_test_split_key)).split(
            prop=train_test_split_pct
        )
        assert train_test_split in ["train", "test"]
        self.benchmark = train if train_test_split == "train" else test

        env, self.env_params = xminigrid.make(
            f"XLand-MiniGrid-R{rooms}-{grid_size}x{grid_size}"
        )
        self.env = DirectionObservationWrapper(env)
        key = jax.random.key(random.randint(0, 1_000_000))
        self.reset_key, self.ruleset_key = jax.random.split(key)

        self.x_sample = jax.jit(self.benchmark.sample_ruleset)
        self.x_reset = jax.jit(self.env.reset)
        self.x_step = jax.jit(self.env.step)

        self.max_steps_per_episode = self.env_params.max_steps
        self.action_space = gym.spaces.Discrete(self.env.num_actions(self.env_params))
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(low=0, high=14, shape=(5, 5, 2), dtype=np.uint8),
                "direction_done": gym.spaces.Box(
                    low=0, high=1, shape=(5,), dtype=np.uint8
                ),
            }
        )
        self.reset()

    @property
    def suggested_max_seq_len(self):
        return self.max_steps_per_episode * self.k_shots

    def tstep_to_obs(self, tstep):
        obs = tstep.observation
        direction_done = np.zeros(5, dtype=np.uint8)
        direction_done[obs["direction"].argmax().item()] = 1
        direction_done[-1] = int(tstep.last())
        return {
            "grid": np.array(obs["img"], dtype=np.uint8),
            "direction_done": direction_done,
        }

    def reset(self, *args, **kwargs):
        ruleset = self.x_sample(self.ruleset_key)
        self.env_params = self.env_params.replace(ruleset=ruleset)
        self.x_timestep = self.x_reset(self.env_params, self.reset_key)
        self.current_episode = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.reset_next = False
        return self.tstep_to_obs(self.x_timestep), {}

    def step(self, action):
        if self.reset_next:
            self.current_episode += 1
            self.x_timestep = self.x_reset(self.env_params, self.reset_key)
            self.episode_return = 0.0
            self.episode_steps = 0
            self.reset_next = False
        else:
            self.x_timestep = self.x_step(self.env_params, self.x_timestep, action)
            self.episode_steps += 1

        next_state = self.tstep_to_obs(self.x_timestep)
        reward = self.x_timestep.reward
        self.episode_return += reward
        ep_end = self.x_timestep.last()

        info = {}
        if ep_end:
            self.reset_next = True
            info[
                f"{AMAGO_ENV_LOG_PREFIX}Ep {self.current_episode} Return"
            ] = self.episode_return

        done = self.current_episode >= self.k_shots
        return next_state, reward, done, done, info


if __name__ == "__main__":
    env = XLandMiniGridEnv(
        rooms=4,
        grid_size=13,
        ruleset_benchmark="trivial-1m",
        train_test_split="train",
        train_test_split_key=0,
        k_shots=25,
    )
    steps = 0
    start = time.time()
    for episode in tqdm.tqdm(range(5)):
        next_state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action)
            steps += 1
    end = time.time()
    print(f"FPS: {steps / (end - start)}")
