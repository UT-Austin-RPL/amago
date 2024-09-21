import random
import tqdm
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np

try:
    import jax
    import xminigrid
except ImportError:
    jax = None
    xminigrid = None
else:
    import jax.numpy as jnp
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        DmEnvAutoResetWrapper,
        RulesAndGoalsObservationWrapper,
    )

from amago.envs.env_utils import AMAGO_ENV_LOG_PREFIX


class XLandMiniGridEnv(gym.Env):
    def __init__(
        self,
        parallel_envs: int,
        k_shots: int = 25,
        rooms: int = 4,
        grid_size: int = 13,
        ruleset_benchmark: str = "small-1m",
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
        self.parallel_envs = parallel_envs

        benchmark = xminigrid.load_benchmark(name=ruleset_benchmark)
        train, test = benchmark.shuffle(key=jax.random.key(train_test_split_key)).split(
            prop=train_test_split_pct
        )
        assert train_test_split in ["train", "test"]
        self.benchmark = train if train_test_split == "train" else test

        env, self.env_params = xminigrid.make(
            f"XLand-MiniGrid-R{rooms}-{grid_size}x{grid_size}"
        )
        self.x_env = RulesAndGoalsObservationWrapper(
            DirectionObservationWrapper(DmEnvAutoResetWrapper(env))
        )
        key = jax.random.key(random.randint(0, 1_000_000))
        reset_key, ruleset_key = jax.random.split(key)
        self.reset_keys = jax.random.split(reset_key, num=parallel_envs)
        self.ruleset_keys = jax.random.split(ruleset_key, num=parallel_envs)

        self.x_sample = jax.vmap(jax.jit(self.benchmark.sample_ruleset))
        self.x_reset = jax.vmap(jax.jit(self.x_env.reset), in_axes=(0, 0))
        self.x_step = jax.vmap(jax.jit(self.x_env.step), in_axes=(0, 0, 0))

        obs_shapes = self.x_env.observation_shape(self.env_params)
        self.max_steps_per_episode = self.env_params.max_steps
        self.action_space = gym.spaces.Discrete(self.x_env.num_actions(self.env_params))
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    low=0, high=14, shape=obs_shapes["img"], dtype=np.uint8
                ),
                "direction_done": gym.spaces.Box(
                    low=0, high=1, shape=(5,), dtype=np.uint8
                ),
                "goal": gym.spaces.Box(
                    low=0, high=14, shape=obs_shapes["goal_encoding"], dtype=np.uint8
                ),
            }
        )
        self.reset()

    @property
    def suggested_max_seq_len(self):
        return self.max_steps_per_episode * self.k_shots

    def get_obs(self):
        obs = self.x_timestep.observation
        done = self.x_timestep.last()
        direction_done = np.concatenate(
            (obs["direction"], done[:, np.newaxis]), axis=-1
        )
        return {
            "grid": np.array(obs["img"], dtype=np.uint8),
            "direction_done": direction_done,
            "goal": np.array(obs["goal_encoding"], dtype=np.uint8),
        }

    def reset(self, *args, **kwargs):
        ruleset = self.x_sample(self.ruleset_keys)
        self.env_params = self.env_params.replace(ruleset=ruleset)
        self.x_timestep = self.x_reset(self.env_params, self.reset_keys)
        self.current_episode = jnp.zeros(self.parallel_envs, dtype=jnp.int32)
        self.episode_return = jnp.zeros(self.parallel_envs, dtype=jnp.float32)
        self.episode_steps = jnp.zeros(self.parallel_envs, dtype=jnp.int32)
        return self.get_obs(), {}

    def replace_rules(self, replace):
        new_ruleset = self.x_sample(self.ruleset_keys)
        ruleset = self.env_params.ruleset
        updated = xminigrid.types.RuleSet(
            goal=jnp.select(replace, new_ruleset.goal, ruleset.goal),
            rules=jnp.select(replace, new_ruleset.rules, ruleset.rules),
            init_tiles=jnp.select(replace, new_ruleset.init_tiles, ruleset.init_tiles),
        )
        self.env_params = self.env_params.replace(ruleset=updated)
        return updated

    def step(self, action):
        action = np.array(action)
        assert action.shape == (self.parallel_envs,)

        self.episode_steps += 1
        ep_end = self.x_timestep.last()
        ep_continues = ~ep_end
        info = defaultdict(list)
        if ep_end.any():
            for env in range(self.parallel_envs):
                if ep_end[env]:
                    info[
                        f"{AMAGO_ENV_LOG_PREFIX}Ep {self.current_episode[env]} Return"
                    ].append(self.episode_return[env])

        # if the env needs to be reset right now...
        self.x_timestep = self.x_step(self.env_params, self.x_timestep, action)
        # ...it has now been reset
        self.episode_return *= ep_continues
        self.episode_steps *= ep_continues
        self.current_episode += ep_end

        reward = self.x_timestep.reward
        self.episode_return += reward

        # handle meta-resets by changing the ruleset
        done = self.current_episode >= self.k_shots
        if done.any():
            self.replace_rules(replace=done)
        self.current_episode *= ~done
        # the reset will kick in on the next `step`

        return self.get_obs(), reward, done, done, dict(info)


if __name__ == "__main__":
    env = XLandMiniGridEnv(
        parallel_envs=2,
        rooms=4,
        grid_size=13,
        ruleset_benchmark="trivial-1m",
        train_test_split="train",
        train_test_split_key=0,
        k_shots=10,
    )
    env.reset()
    steps = 100000
    start = time.time()
    for step in tqdm.tqdm(range(steps)):
        action = [env.action_space.sample() for _ in range(env.parallel_envs)]
        next_state, reward, done, _, info = env.step(action)
    end = time.time()
    print(f"FPS: {steps / (end - start)}")
