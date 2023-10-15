import random

"""
This environment code is from https://github.com/twni2016/Memory-RL
and discussed in "When Do Transformers Shine in RL? Decoupling Memory from
Credit Assignment", Ni et al., 2023.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os


class TMazeBase(gym.Env):
    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        ambiguous_position: bool = False,
        expose_goal: bool = False,
        add_timestep: bool = False,
    ):
        """
        The Base class of TMaze, decouples episode_length and corridor_length

        Other variants:
            (Osband, 2016): distract_reward = eps > 0, goal_reward is given at T-junction (no choice).
                This only tests the exploration and discounting of agent, no memory required
            (Osband, 2020): ambiguous_position = True, add_timestep = True, supervised = True.
                This only tests the memory of agent, no exploration required (not implemented here)
        """
        super().__init__()
        assert corridor_length >= 1 and episode_length >= 1
        assert penalty <= 0.0

        self.episode_length = episode_length
        self.corridor_length = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = goal_reward
        self.penalty = penalty
        self.distract_reward = distract_reward

        self.ambiguous_position = ambiguous_position
        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(4)  # four directions
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        self.tmaze_map = np.zeros(
            (3 + 2, self.oracle_length + self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[
            [self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1
        ] = True  # goal candidates

        obs_dim = 2 if self.ambiguous_position else 3
        if self.expose_goal:  # test Markov policies
            assert self.ambiguous_position is False
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def position_encoding(self, x: int, y: int, goal_y: int):
        if x == 0:
            # oracle position
            if not self.oracle_visited:
                # only appear at first
                exposure = goal_y
                self.oracle_visited = True
            else:
                exposure = 0

        if self.ambiguous_position:
            if x == 0:
                # oracle position
                return [0, exposure]
            elif x < self.oracle_length + self.corridor_length:
                # intermediate positions (on the corridor)
                return [0, 0]
            else:
                # T-junction or goal candidates
                return [1, y]
        else:
            if self.expose_goal:
                return [x, y, goal_y if self.oracle_visited else 0]
            if x == 0:
                # oracle position
                return [x, y, exposure]
            else:
                return [x, y, 0]

    def timestep_encoding(self):
        return (
            [
                self.time_step,
            ]
            if self.add_timestep
            else []
        )

    def get_obs(self):
        return np.array(
            self.position_encoding(self.x, self.y, self.goal_y)
            + self.timestep_encoding(),
            dtype=np.float32,
        )

    def reward_fn(self, done: bool, x: int, y: int, goal_y: int):
        if done:  # only give bonus at the final time step
            return float(y == goal_y) * self.goal_reward
        else:
            # a penalty (when t > o) if x < t - o (desired: x = t - o)
            rew = float(x < self.time_step - self.oracle_length) * self.penalty
            if x == 0:
                return rew + self.distract_reward
            else:
                return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        # transition
        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            # valid move
            self.x, self.y = self.x + move_x, self.y + move_y

        terminated = truncated = self.time_step >= self.episode_length
        rew = self.reward_fn(terminated, self.x, self.y, self.goal_y)
        return self.get_obs(), rew, terminated, truncated, {}

    def reset(self, *args, **kwargs):
        self.x, self.y = self.oracle_length, 0
        self.goal_y = random.choice([-1, 1])

        self.oracle_visited = False
        self.time_step = 0
        return self.get_obs(), {}

    def visualize(self, trajectories: np.array, idx: str):
        from utils import logger

        # trajectories: (B, T+1, O)
        batch_size, seq_length, _ = trajectories.shape
        xs = np.arange(seq_length)

        for traj in trajectories:
            # plot the 0-th element
            plt.plot(xs, traj[:, 0])

        plt.xlabel("Time Step")
        plt.ylabel("Position X")
        plt.savefig(
            os.path.join(logger.get_dir(), "plt", f"{idx}.png"),
            dpi=200,  # 200
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


class TMazeAlt(TMazeBase):
    """
    Changes the movement penalty to make the environment
    (much) more sample efficient.

    https://github.com/twni2016/Memory-RL/issues/1
    """

    def reward_fn(self, action, done: bool, y: int, goal_y: int):
        assert self.distract_reward == 0
        if done:
            return float(y == goal_y) * self.goal_reward
        else:
            rew = (
                float(action != 0 and self.time_step > self.oracle_length)
                * self.penalty
            )
            return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)
        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            self.x, self.y = self.x + move_x, self.y + move_y
        terminated = truncated = self.time_step >= self.episode_length
        rew = self.reward_fn(action, terminated, self.y, self.goal_y)
        return self.get_obs(), rew, terminated, truncated, {}


class TMazeAltPassive(TMazeAlt):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
    ):
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


class TMazeClassicPassive(TMazeBase):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
    ):
        """
        Classic TMaze with Passive Memory
            assert episode_length == corridor_length + 1
            (Bakker, 2001): ambiguous_position = True. penalty = 0
                This is too hard even for T = 10 for vanilla agents because the exploration is extremely hard.
                This tests both memory and exploration
            **(tmaze_classic; this work)**: based on (Bakker, 2001), set penalty < 0
                Unit-tests memory
        """
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


class TMazeAltActive(TMazeAlt):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
    ):
        oracle_length = 1
        super().__init__(
            episode_length=corridor_length + 2 * oracle_length + 1,
            corridor_length=corridor_length,
            oracle_length=oracle_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


class TMazeClassicActive(TMazeBase):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
    ):
        """
        Classic TMaze with Active Memory
            assert episode_length == corridor_length + 1 + 2o
            where o is the length between the starting point and oracle that gives the goal information
            TMazeClassicPassive is a special case of o = 0.
        """
        oracle_length = 1
        super().__init__(
            episode_length=corridor_length + 2 * oracle_length + 1,
            corridor_length=corridor_length,
            oracle_length=oracle_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )
