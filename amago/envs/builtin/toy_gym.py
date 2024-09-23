import copy
import random

import gymnasium as gym
import numpy as np


class RandomLunar(gym.Env):
    def __init__(
        self,
        k_shots=2,
    ):
        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.k_shots = k_shots

    def reset(self, *args, **kwargs):
        self.current_gravity = random.uniform(-3.0, -0.1)
        self.current_wind = random.uniform(0.0, 20.0)
        self.current_turbulence = random.uniform(0.0, 2.0)
        self.env = gym.make(
            "LunarLander-v2",
            continuous=True,
            gravity=self.current_gravity,
            enable_wind=True,
            wind_power=self.current_wind,
            turbulence_power=self.current_turbulence,
        )
        self.current_k = 0
        return self.env.reset()

    def step(self, action):
        done = False
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            next_state, info = self.env.reset()
            self.current_k += 1
        if self.current_k >= self.k_shots:
            done = True
        return next_state, reward, done, False, info


from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class MetaFrozenLake(gym.Env):
    def __init__(
        self,
        size: int,
        k_shots: int = 10,
        hard_mode: bool = False,
        recover_mode: bool = False,
    ):
        self.size = size
        self.k_shots = k_shots
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
            [x / self.size, y / self.size, reset_signal, self.current_k / self.k_shots],
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
            self.current_k += 1
            next_state, info = self.soft_reset()
        else:
            next_state, info = self.make_obs(False), {}

        terminated = self.current_k >= self.k_shots
        return next_state, reward, terminated, False, info

    def render(self, *args, **kwargs):
        render_map = copy.deepcopy(self.active_map)
        render_map[self.x][self.y] = "A"
        print(f"\nFrozen Lake (k={self.k_shots}, Hard Mode={self.hard_mode})")
        for row in render_map:
            print(" ".join(row))
