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
    goal = jnp.where(jnp.expand_dims(replace, 1), new_goal, old_goal)
    rule = jnp.where(jnp.expand_dims(replace, (1, 2)), new_rule, old_rule)
    tile = jnp.where(jnp.expand_dims(replace, (1, 2)), new_tile, old_tile)
    return goal, rule, tile


def _swap_trees(old_tree, new_tree, replace):

    def select_fn(old, new):
        diff = old.ndim - replace.ndim
        axes = tuple([1 + i for i in range(diff)])
        idxs = jnp.expand_dims(replace, axes)
        return jnp.where(idxs, new, old)

    return jax.tree_util.tree_map(select_fn, old_tree, new_tree)


class XLandMinigridVectorizedGym(gym.Env):
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

        # according to gymnax, jit(vmap()) is faster than vmap(jit())... seems to be true here
        self.x_sample = jax.jit(
            jax.vmap(self.benchmark.sample_ruleset, in_axes=(0,)),
            device=self.jax_device,
        )
        self.x_reset = jax.jit(
            jax.vmap(self.x_env.reset, in_axes=(0, 0)), device=self.jax_device
        )
        self.x_step = jax.jit(
            jax.vmap(self.x_env.step, in_axes=(0, 0, 0)), device=self.jax_device
        )
        self.x_swap_rules = jax.jit(_swap_rules, device=self.jax_device)
        self.x_swap_tsteps = jax.jit(_swap_trees, device=self.jax_device)
        self.x_swap_tsteps_new = jax.jit(_swap_trees, device=self.jax_device)

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
        # establish placeholder values for resets
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
        return (self.max_steps_per_episode + 1) * self.k_shots

    def get_obs(self):
        obs = self.x_timestep.observation
        done = self.x_timestep.last()
        direction_done = np.concatenate(
            (obs["direction"], done[:, np.newaxis]), axis=-1
        )
        # it seems more in the spirit of AdA's XLand to give the "goal" as an observation
        # but let the agent meta-learn the "rules"... though it's not clear if that
        # makes xland minigrid easier than it was meant to be...
        return {
            "grid": np.array(obs["img"], dtype=np.uint8),
            "direction_done": direction_done,
            "goal": np.array(obs["goal_encoding"], dtype=np.uint8),
        }

    def reset(self, *args, **kwargs):
        # set all done, then reset if done
        is_done = jnp.ones(self.parallel_envs, dtype=jnp.bool_, device=self.jax_device)
        return self.reset_if_done(is_done)

    def reset_if_done(self, is_done):
        # generate a new ruleset in parallel for every env
        new_ruleset = self.x_sample(self.new_ruleset_keys())
        ruleset = self.env_params.ruleset

        # keep the old ruleset if the env is not done, else take new ruleset
        goals, rules, tiles = self.x_swap_rules(
            ruleset.goal,
            ruleset.rules,
            ruleset.init_tiles,
            new_ruleset.goal,
            new_ruleset.rules,
            new_ruleset.init_tiles,
            is_done,
        )
        updated = xminigrid.types.RuleSet(
            goal=goals,
            rules=rules,
            init_tiles=tiles,
        )
        self.env_params = self.env_params.replace(ruleset=updated)

        # reset every env
        reset_timestep = self.x_reset(self.env_params, self.new_reset_keys())

        # keep old timestep if not done, else take new timestep
        new_timestep = self.x_swap_tsteps(self.x_timestep, reset_timestep, is_done)
        self.x_timestep = new_timestep

        # keep counters if not done, else reset them
        not_done = ~is_done
        self.current_episode *= not_done
        self.episode_return *= not_done
        self.episode_steps *= not_done
        return self.get_obs(), {}

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
        reward = self.x_timestep.reward.copy()
        self.episode_return += reward

        # meta-resets need to happen on the same timestep as they occur
        done = self.current_episode >= self.k_shots
        if done.any():
            self.reset_if_done(done)

        next_obs = self.get_obs()
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.bool_)
        return next_obs, reward, done, done, info

    def render(self, env_idx: int, *args, **kwargs):
        return self.x_env.render(
            self.env_params, jax.tree_map(lambda x: x[env_idx], self.x_timestep)
        )


if __name__ == "__main__":
    import tqdm
    import time
    import matplotlib.pyplot as plt
    from amago.envs import AMAGOEnv, SequenceWrapper

    env = XLandMinigridVectorizedGym(
        parallel_envs=4,
        rooms=1,
        grid_size=9,
        ruleset_benchmark="trivial-1m",
        train_test_split="train",
        train_test_split_key=0,
        k_shots=2,
        jax_device=6,
    )
    env = AMAGOEnv(env, env_name="XLandMiniGridEnv", batched_envs=4)
    env = SequenceWrapper(env)

    render_idxs = None

    if render_idxs is not None:
        fig, axs = plt.subplots(1, len(render_idxs))

    env.reset()
    steps = 3_000
    start = time.time()
    for step in tqdm.tqdm(range(steps)):
        print(env.current_episode)
        action = np.array(
            [env.action_space.sample() for _ in range(env.parallel_envs)]
        )[:, np.newaxis]
        next_state, reward, terminated, truncated, info = env.step(action)

        if render_idxs is not None:
            for _i, ax in zip(render_idxs, axs):
                ax.clear()
                img = env.unwrapped.render(env_idx=_i)
                ax.imshow(img)
            plt.pause(0.01)
            fig.canvas.draw()

    end = time.time()
    print(f"FPS: {steps / (end - start)}")
