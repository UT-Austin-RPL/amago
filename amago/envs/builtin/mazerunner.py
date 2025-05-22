from functools import partial
import copy
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist

from amago.envs import AMAGOEnv


def random_maze(width: int = 11, height: int = 11, complexity: float = 0.75, density: float = 0.75) -> np.ndarray:
    """Generate a maze layout.

    Code from https://github.com/zuoxingdong/mazelab
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x = random.randrange(0, shape[1] // 2 + 1) * 2
        y = random.randrange(0, shape[0] // 2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[random.randrange(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_

    return Z.astype(int)


class MazeRunnerGymEnv(gym.Env):
    """Core MazeRunner Environment

    Agent navigates a randomly generated maze to reach a sequence of goal locations.

    Args:
        maze_dim: Dimension of the maze.
        min_num_goals: Minimum number of goal locations to navigate to.
        max_num_goals: Maximum number of goal locations to navigate to.
        time_limit: Max episode length.
        goal_in_obs: Whether to include the goal location in the observation. Defaults to True.
        randomized_action_space: Whether to randomly permute the action space indices on reset. Defaults to False.
    """

    def __init__(
        self,
        maze_dim: int,
        min_num_goals: int,
        max_num_goals: int,
        time_limit: int,
        goal_in_obs: bool = True,
        randomized_action_space: bool = False,
    ):
        self.goal_in_obs = goal_in_obs
        assert min_num_goals <= max_num_goals
        self.max_num_goals = max_num_goals
        self.min_num_goals = min_num_goals
        self.time_limit = time_limit
        self.randomized_action_space = randomized_action_space
        self.rng = np.random.default_rng()

        # mazes have odd side length
        self.maze_dim = (maze_dim // 2) * 2 + 1
        self.reset()
        obs_dim = 7 + (int(goal_in_obs) * 2 * max_num_goals)
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(obs_dim,)
        )
        self.action_space = gym.spaces.Discrete(5)

        self._plotting = False
        self._fig, self._ax = None, None

    def _generate_new_maze(self):
        # generate a random maze
        core_maze = random_maze(self.maze_dim, self.maze_dim)
        # set the bottom area empty
        top_spawn = self.maze_dim - 5
        bottom_spawn = self.maze_dim - 1
        core_maze[top_spawn:bottom_spawn, 1:-1] = 0
        core_maze[top_spawn + 1 : bottom_spawn, self.maze_dim // 2 - 1] = 1
        core_maze[top_spawn + 1 : bottom_spawn, self.maze_dim // 2 + 1] = 1
        return core_maze

    def goal_sequence(self, flatten: bool = True):
        goals = copy.deepcopy(self.goal_positions)
        for i in range(self.active_goal_idx):
            goals[i] = (-1, -1)

        while len(goals) < self.max_num_goals:
            goals.insert(0, (-1, -1))

        goals = np.array(goals, dtype=np.int32)
        if flatten:
            goals = goals.flatten()
        return goals

    def reset(self, *args, **kwargs):
        # make a maze as an n x n array
        self.maze = self._generate_new_maze()
        self.start = (self.maze_dim - 2, self.maze_dim // 2)
        empty_locations = [x for x in zip(*np.where(self.maze == 0))]
        empty_locations.remove(self.start)
        num_goals = random.randint(self.min_num_goals, self.max_num_goals)
        self.goal_positions = random.sample(empty_locations, k=num_goals)
        self.action_dirs = np.array([[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]])
        if self.randomized_action_space:
            self.action_dirs = self.action_dirs[self.rng.permutation(5)]
        self.active_goal_idx = 0
        self.pos = self.start
        self.timer = 0
        self._enforce_reset = False
        self._plotting = False
        self._goal_render_texts = [None for _ in range(num_goals)]
        return self._get_obs(), {}

    def step(self, act):
        assert not self._enforce_reset, "Reset the environment with `env.reset()`"
        self.timer += 1
        chosen_dir = self.action_dirs[act]
        desired_loc = tuple(self.pos + chosen_dir)

        valid = True
        if self.maze[desired_loc] != 0:
            valid = False
        else:
            for coord in desired_loc:
                if coord < 0 or coord >= self.maze_dim:
                    valid = False

        if valid:
            self.pos = desired_loc

        success = self.pos == self.goal_positions[self.active_goal_idx]
        terminated = False
        rew = float(success)
        if success:
            if self.active_goal_idx == len(self.goal_positions) - 1:
                terminated = True
                self._enforce_reset = True
            self.active_goal_idx += 1

        truncated = self.timer >= self.time_limit
        # timer is in the observation; enforce finite horizon POMDP
        terminated = terminated or truncated
        return self._get_obs(), rew, terminated, truncated, {"success": success}

    def go_back_to_start(self):
        self.pos = self.start
        return self._get_obs()

    def _get_obs(self):
        i, j = tuple(self.pos)

        space_west = 0
        seek_west = j - 1
        while seek_west > 0:
            if self.maze[i, seek_west] == 0:
                seek_west -= 1
                space_west += 1
            else:
                break

        space_east = 0
        seek_east = j + 1
        while seek_east < self.maze_dim:
            if self.maze[i, seek_east] == 0:
                seek_east += 1
                space_east += 1
            else:
                break

        space_north = 0
        seek_north = i - 1
        while seek_north > 0:
            if self.maze[seek_north, j] == 0:
                seek_north -= 1
                space_north += 1
            else:
                break

        space_south = 0
        seek_south = i + 1
        while seek_south < self.maze_dim:
            if self.maze[seek_south, j] == 0:
                seek_south += 1
                space_south += 1
            else:
                break

        obs = np.array(
            [
                i / self.maze_dim,
                j / self.maze_dim,
                space_west / self.maze_dim,
                space_north / self.maze_dim,
                space_east / self.maze_dim,
                space_south / self.maze_dim,
                self.timer / self.time_limit,
            ],
            dtype=np.float32,
        )

        if self.goal_in_obs:
            obs = np.concatenate((obs, self.goal_sequence() / self.maze_dim))
        return obs

    def render(self, *args, **kwargs):
        if not self._plotting:
            self.start_plotting()
            plt.ion()
            self._plotting = True

        plt.tight_layout()
        background = np.ones((self.maze_dim, self.maze_dim, 3), dtype=np.uint8)
        maze_img = (
            background * abs(np.expand_dims(self.maze, -1) - 1) * 255
        )  # zero out (white) where there is a valid path

        # TODO: give us more colors
        goal_color_wheel = [
            [240, 3, 252],
            [255, 210, 87],
            [3, 219, 252],
            [252, 2157, 3],
        ]
        for i, goal_pos in enumerate(self.goal_positions):
            if self.active_goal_idx > i:
                continue
            x, y = goal_pos
            maze_img[x, y, :] = goal_color_wheel[i % len(goal_color_wheel)]

        maze_img[self.pos[0], self.pos[1], :] = [110, 110, 110]  # grey
        plt.imshow(maze_img)

        for i, goal_pos in enumerate(self.goal_positions):
            if self.active_goal_idx > i:
                continue
            y, x = goal_pos
            self._goal_render_texts[i] = plt.text(
                x, y, str(i), ha="center", va="center"
            )

        self._ax.set_title(
            f"k={self.goal_positions}, active_goal={self.goal_positions[self.active_goal_idx]}"
        )
        plt.draw()
        plt.pause(0.1)

    def start_plotting(self):
        if self._fig:
            plt.close()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)


class RelabelInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env, MazeRunnerGymEnv)
        super().__init__(env)
        k = self.env.max_num_goals
        n = self.env.maze_dim
        # fmt: off
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "achieved": gym.spaces.Box(low=0, high=n, shape=(1, 2), dtype=np.int32),
                "goals": gym.spaces.Box(low=0, high=n, shape=(k, 2), dtype=np.int32),
            }
        )
        # fmt: on

    def add_goal_info(self, obs):
        return {
            "obs": obs,
            "achieved": np.array(self.env.pos)[np.newaxis, :],
            "goals": self.env.goal_sequence(flatten=False),
        }

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return self.add_goal_info(obs), info

    def step(self, action):
        next_obs, *rest = super().step(action)
        return self.add_goal_info(next_obs), *rest


class MazeRunnerAMAGOEnv(AMAGOEnv):
    """AMAGOEnv MazeRunner environment.

    Configures MazeRunnerGymEnv for hindsight experience replay.

    Args:
        maze_dim: Dimension of the maze.
        num_goals: Number of goal locations to navigate to.
        time_limit: Max episode length.
        randomized_action_space: Whether to randomly permute the action space indices on reset. Defaults to False.
    """

    def __init__(
        self,
        maze_dim: int,
        num_goals: int,
        time_limit: int,
        randomized_action_space: bool,
    ):
        env = RelabelInfoWrapper(
            MazeRunnerGymEnv(
                maze_dim=maze_dim,
                # simplification in re-implementation: every episode has the same # goals
                min_num_goals=num_goals,
                max_num_goals=num_goals,
                goal_in_obs=False,
                time_limit=time_limit,
                # appendix calls this "rewired" actions
                randomized_action_space=randomized_action_space,
            )
        )
        super().__init__(
            env,
            batched_envs=1,
            env_name=f"MazeRunner-{maze_dim}x{maze_dim}-k{num_goals}",
        )
