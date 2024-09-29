import random
from collections import defaultdict
from typing import Optional

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
    from jax import device_put as dp
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        DmEnvAutoResetWrapper,
        RulesAndGoalsObservationWrapper,
    )

from amago.envs import AMAGO_ENV_LOG_PREFIX


def _swap_rules(old_goal, old_rule, old_tile, new_goal, new_rule, new_tile, replace):
    goal = jnp.select(replace, new_goal, old_goal)
    rule = jnp.select(replace, new_rule, old_rule)
    tile = jnp.select(replace, new_tile, old_tile)
    return goal, rule, tile


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
        jax_device: Optional[int] = None,
    ):
        assert (
            jax is not None and xminigrid is not None
        ), "JAX and xminigrid must be installed to use this environment"
        self.rooms = rooms
        self.grid_size = grid_size
        self.k_shots = k_shots
        self.ruleset_benchmark = ruleset_benchmark
        self.parallel_envs = parallel_envs
        self.jax_device = (
            jax.devices()[jax_device]
            if jax_device is not None
            else jax.devices("cpu")[0]
        )
        print(f"Using JAX device {self.jax_device} for XLandMiniGridEnv")

        # this always uses cuda memory for some reason?
        benchmark = xminigrid.load_benchmark(name=ruleset_benchmark)
        train, test = benchmark.shuffle(key=jax.random.key(train_test_split_key)).split(
            prop=train_test_split_pct
        )
        assert train_test_split in ["train", "test"]
        self.benchmark = train if train_test_split == "train" else test

        env, self.env_params = xminigrid.make(
            f"XLand-MiniGrid-R{rooms}-{grid_size}x{grid_size}"
        )
        self.env_params = dp(self.env_params, self.jax_device)
        self.x_env = RulesAndGoalsObservationWrapper(
            DirectionObservationWrapper(DmEnvAutoResetWrapper(env))
        )
        key = jax.random.key(random.randint(0, 1_000_000))
        self._reset_key, self._ruleset_key = jax.random.split(key)

        self.x_sample = jax.vmap(
            jax.jit(self.benchmark.sample_ruleset, device=self.jax_device)
        )
        self.x_reset = jax.vmap(
            jax.jit(self.x_env.reset, device=self.jax_device), in_axes=(0, 0)
        )
        self.x_step = jax.vmap(
            jax.jit(self.x_env.step, device=self.jax_device), in_axes=(0, 0, 0)
        )
        self.x_swap = jax.jit(_swap_rules, device=self.jax_device)

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

    def new_reset_keys(self):
        keys = dp(
            jax.random.split(self._reset_key, num=self.parallel_envs + 1),
            self.jax_device,
        )
        self._reset_key = keys[0]
        return keys[1:]

    def new_ruleset_keys(self):
        keys = dp(
            jax.random.split(self._ruleset_key, num=self.parallel_envs + 1),
            self.jax_device,
        )
        self._ruleset_key = keys[0]
        return keys[1:]

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
        ruleset = self.x_sample(self.new_ruleset_keys())
        self.env_params = self.env_params.replace(ruleset=ruleset)
        self.x_timestep = self.x_reset(self.env_params, self.new_reset_keys())
        self.current_episode = jnp.zeros(
            self.parallel_envs, dtype=jnp.int32, device=self.jax_device
        )
        self.episode_return = jnp.zeros(
            self.parallel_envs, dtype=jnp.float32, device=self.jax_device
        )
        self.episode_steps = jnp.zeros(
            self.parallel_envs, dtype=jnp.int32, device=self.jax_device
        )
        return self.get_obs(), {}

    def replace_rules(self, replace):
        new_ruleset = self.x_sample(self.new_ruleset_keys())
        ruleset = self.env_params.ruleset
        goals, rules, tiles = self.x_swap(
            ruleset.goal,
            ruleset.rules,
            ruleset.init_tiles,
            new_ruleset.goal,
            new_ruleset.rules,
            new_ruleset.init_tiles,
            replace,
        )
        updated = xminigrid.types.RuleSet(
            goal=goals,
            rules=rules,
            init_tiles=tiles,
        )
        self.env_params = self.env_params.replace(ruleset=updated)
        return updated

    def step(self, action):
        action = jnp.array(action, device=self.jax_device)
        assert action.shape == (self.parallel_envs,)

        self.episode_steps += 1
        ep_end = self.x_timestep.last()
        ep_continues = ~ep_end

        # log episode statistics
        ep_end_idxs = jnp.nonzero(ep_end)[0]
        if ep_end_idxs.shape[0] > 0:
            info = defaultdict(list)
            ep_returns = jnp.take(self.episode_return, ep_end_idxs)
            ep_nums = jnp.take(self.current_episode, ep_end_idxs)
            for ep_num, ep_return in zip(ep_nums, ep_returns):
                info[f"{AMAGO_ENV_LOG_PREFIX}Ep {ep_num} Return"].append(ep_return)
        else:
            info = {}

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

        return self.get_obs(), reward, done, done, info


if __name__ == "__main__":
    import tqdm
    import time

    from amago.envs import AMAGOEnv, SequenceWrapper

    env = SequenceWrapper(
        AMAGOEnv(
            XLandMiniGridEnv(
                parallel_envs=512,
                rooms=4,
                grid_size=13,
                ruleset_benchmark="trivial-1m",
                train_test_split="train",
                train_test_split_key=0,
                k_shots=25,
                jax_device=0,
            ),
            batched_envs=512,
            env_name="XLandMiniGridEnv",
        )
    )

    env.reset()
    steps = 100000
    start = time.time()
    for step in tqdm.tqdm(range(steps)):
        action = np.array(
            [env.action_space.sample() for _ in range(env.parallel_envs)]
        )[:, np.newaxis]
        next_state, reward, done, _, info = env.step(action)
    end = time.time()
    print(f"FPS: {steps / (end - start)}")
