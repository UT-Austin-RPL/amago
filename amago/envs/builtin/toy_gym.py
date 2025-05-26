"""
Custom toy gym environments.
"""

import copy
import random

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

from amago.envs import AMAGO_ENV_LOG_PREFIX


class MetaFrozenLake(gym.Env):
    """
    A version of gym's toy FrozenLake problem that demonstrates the
    strengths of black-box meta-RL. The agent begins in an unknown
    lake layout and has k attempts to find and exploit a path to the
    goal. The concept of k explicit attempts can be removed and
    replaced with long trials that penalize failure without resets.
    Explicit partial observability can be added in the form of noisy
    sensor observations. Discrete action indices can be shuffled
    between tasks.

    Args:
        size: The size of the lake map (n x n grid).
        k_episodes: The number of attempts agents have on each new
            lake layout.
        hard_mode: If True, randomly shuffle discrete action indices
            on reset and add noise to the agent's position sensors.
            Randomly create "holes" in the ice behind the agent's
            path to penalize backtracking. These changes introduce
            partial observability and increase memory demands. Defaults
            to False.
        recover_mode: If False, falling through the ice terminates the
            episode. If True, the agent is allowed to recover to its
            previous position but receives a penalty. Defaults to False.
    """

    def __init__(
        self,
        size: int,
        k_episodes: int = 10,
        hard_mode: bool = False,
        recover_mode: bool = False,
    ):
        self.size = size
        self.k_episodes = k_episodes
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(shape=(4,), low=0.0, high=1.0)
        self.hard_mode = hard_mode
        self.recover_mode = recover_mode
        self.reset()

    def reset(self, *args, **kwargs):
        self.current_map = [[t for t in row] for row in generate_random_map(self.size)]
        self.action_mapping = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.hard_mode and random.random() < 0.5:
            temp = self.action_mapping[1]
            self.action_mapping[1] = self.action_mapping[3]
            self.action_mapping[3] = temp
        self.current_k = 0
        return self.soft_reset()

    def make_obs(self, reset_signal: bool):
        if self.hard_mode and random.random() < 0.25:
            x = min(max(self.x + random.choice([-1, 0, 1]), 0), self.size - 1)
            y = min(max(self.y + random.choice([-1, 0, 1]), 0), self.size - 1)
        else:
            x, y = self.x, self.y
        return np.array(
            [
                x / self.size,
                y / self.size,
                reset_signal,
                self.current_k / self.k_episodes,
            ],
            dtype=np.float32,
        )

    def soft_reset(self):
        self.active_map = copy.deepcopy(self.current_map)
        self.x, self.y = 0, 0
        obs = self.make_obs(reset_signal=True)
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action)
        move_x, move_y = self.action_mapping[action]
        next_x = max(min(self.x + move_x, self.size - 1), 0)
        next_y = max(min(self.y + move_y, self.size - 1), 0)

        if (
            (self.x, self.y) != (next_y, next_y)
            and self.hard_mode
            and random.random() < 0.33
        ):
            self.active_map[self.x][self.y] = "H"

        on = self.active_map[next_x][next_y]
        if on == "G":
            reward = 1.0
            soft_reset = True
        elif on == "H":
            reward = 0.0 if not self.recover_mode else -0.1
            soft_reset = not self.recover_mode
            if self.recover_mode:
                next_x = self.x
                next_y = self.y
        else:
            reward = 0.0
            soft_reset = False

        self.x = next_x
        self.y = next_y

        if soft_reset:
            next_state, info = self.soft_reset()
            success = on == "G"
            info[f"{AMAGO_ENV_LOG_PREFIX}Episode {self.current_k} Success"] = success
            self.current_k += 1
        else:
            next_state, info = self.make_obs(False), {}

        terminated = self.current_k >= self.k_episodes
        return next_state, reward, terminated, False, info

    def render(self, *args, **kwargs):
        render_map = copy.deepcopy(self.active_map)
        render_map[self.x][self.y] = "A"
        print(f"\nFrozen Lake (k={self.k_episodes}, Hard Mode={self.hard_mode})")
        for row in render_map:
            print(" ".join(row))


class RoomKeyDoor(gym.Env):
    """A version of the Dark Room Key-Door Env.

    Based on Algorithm Distillation (Laskin et al., 2022).

    Args:
        dark: If True, the key and door position are hidden from the
            agent. If False, they are revealed, which is an easy way
            to demonstrate that meta-learning problems are created by
            partial observability. Defaults to True.
        size: The size of the grid (n x n). Defaults to 9.
        max_episode_steps: The maximum number of steps in each episode before the
            task is reset. Defaults to 50.
        meta_rollout_horizon: The agent has this many timsteps to adapt to
            each world layout. The best solution is to infer the key and door locations
            and then solve the task as many times as possible within this time limit.
            Defaults to 500.
        start_location: The starting location of the agent. Defaults to
            "random". Can also be set to a specific (x, y) coordinate.
        key_location: The location of the key. Defaults to "random". Can
            also be set to a specific (x, y) coordinate.
        goal_location: The location of the goal. Defaults to "random".
            Can also be set to a specific (x, y) coordinate.
        randomize_actions: If True, the discrete action indices are
            randomly shuffled on each reset. Defaults to False.
    """

    def __init__(
        self,
        dark: bool = True,
        size: int = 9,
        max_episode_steps: int = 50,
        meta_rollout_horizon: int = 500,
        start_location: tuple[int, int] | str = "random",
        key_location: tuple[int, int] | str = "random",
        goal_location: tuple[int, int] | str = "random",
        randomize_actions: bool = False,
    ):
        self.dark = dark
        self.size = size
        self.H = max_episode_steps
        self.H_meta = meta_rollout_horizon
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4 if self.dark else 8,)
        )
        self.action_space = gym.spaces.Discrete(5)
        self.goal_location = goal_location
        self.key_location = key_location
        self.start_location = start_location
        self.randomize_actions = randomize_actions

    def reset_same_task(self):
        self.pos = self.start
        self.episode_time = 0
        self.has_key = False

    def reset(self, *args, **kwargs):
        self.generate_task()
        self.global_time = 0
        self.reset_same_task()
        self.reset_next_step = False
        return self.obs(), {}

    def generate_task(self):
        self.start = np.array(
            random.choices(range(self.size), k=2)
            if self.start_location == "random"
            else self.start_location
        )
        self.key = np.array(
            random.choices(range(self.size), k=2)
            if self.key_location == "random"
            else self.key_location
        )
        self.goal = np.array(
            random.choices(range(self.size), k=2)
            if self.goal_location == "random"
            else self.goal_location
        )
        self.dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
        if self.randomize_actions:
            random.shuffle(self.dirs)

    def step(self, action: int):
        self.global_time += 1
        if self.reset_next_step:
            self.reset_same_task()
            self.reset_next_step = False
            reward = 0.0
        else:
            self.episode_time += 1
            self.pos = np.clip(self.pos + np.array(self.dirs[action]), 0, self.size - 1)
            reward = 0.0
            if self.has_key and (self.pos == self.goal).all():
                reward = 1.0
                self.reset_next_step = True
            elif not self.has_key and (self.pos == self.key).all():
                reward = 1.0
                self.has_key = True
            if self.episode_time >= self.H:
                self.reset_next_step = True
        metadone = self.global_time >= self.H_meta
        return self.obs(), reward, metadone, metadone, {}

    def obs(self):
        x, y = self.pos
        norm = lambda j: float(j) / self.size
        # time and has_key keep this fully observed
        base = [norm(x), norm(y), self.has_key, float(self.episode_time) / self.H]
        if not self.dark:
            goal_x, goal_y = self.goal
            key_x, key_y = self.key
            base += [norm(goal_x), norm(goal_y), norm(key_x), norm(key_y)]
        return np.array(base, dtype=np.float32)

    def render(self, *args, **kwargs):
        img = [["." for _ in range(self.size)] for _ in range(self.size)]
        player_x, player_y = self.pos
        goal_x, goal_y = self.goal
        key_x, key_y = self.key
        img[player_x][player_y] = "P"
        img[goal_x][goal_y] = "D"
        img[key_x][key_y] = "K"
        print(
            f"{'Dark' if self.dark else 'Light'} Room Key-Door: Key = {self.key}, Door = {self.goal}, Player = {self.pos}"
        )
        for row in img:
            print(" ".join(row))
