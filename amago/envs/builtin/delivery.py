import random
import copy

import numpy as np
import gymnasium as gym

from amago.envs import AMAGOEnv
from amago.hindsight import GoalSeq


class DeliveryEnv(AMAGOEnv):
    def __init__(
        self,
        length: int = 30,
        min_packages: int = 2,
        max_packages=4,
        max_timesteps: int = 180,
    ):
        env = DeliveryGymEnv(length, min_packages, max_packages, max_timesteps)
        super().__init__(env, horizon=max_timesteps)
        self.max_goals = max_packages
        self.min_goals_ = min_packages
        self.length_ = length
        self.max_timesteps_ = max_timesteps

    def step(self, action):
        return super().step(action, normal_rl_reward=False, normal_rl_reset=True)

    @property
    def env_name(self):
        return f"delivery_length{self.length_}_packages{self.min_goals_}-{self.max_goals}_horizon{self.max_timesteps_}"

    @property
    def achieved_goal(self):
        return [
            np.array(g).astype(np.float32)[np.newaxis]
            for g in self.env.unwrapped.achieved_goals
        ]

    @property
    def kgoal_space(self):
        return gym.spaces.Box(
            low=0.0, high=self.env.unwrapped.length, shape=(self.max_goals, 1)
        )

    @property
    def goal_sequence(self):
        goals = [
            np.array(a).astype(np.float32)[np.newaxis]
            for a in self.env.unwrapped.og_destinations
        ]
        active_idx = self.env.unwrapped.active_idx
        return GoalSeq(seq=goals, active_idx=active_idx)


class DeliveryGymEnv(gym.Env):
    def __init__(
        self,
        length: int = 100,
        min_packages: int = 2,
        max_packages: int = 6,
        max_timesteps: int = 400,
    ):
        super().__init__()
        self.length = length
        self.min_packages = min_packages
        self.max_packages = max_packages
        self.max_timesteps = max_timesteps

        # forward, left, right, package 1, package 2, package 3, package 4, no_op
        self.action_space = gym.spaces.Discrete(8)
        # [x, remaining, at_fork]
        self.observation_space = gym.spaces.Box(low=0.0, high=length, shape=(3,))

    def reset(self, *args, **kwargs):
        self.num_packages = random.randint(self.min_packages, self.max_packages)
        self._pos = 0
        self._time = 0
        num_forks = random.randint((self.length // 5) // 2, self.length // 5)
        self._forks = random.sample(range(1, self.length), k=num_forks)
        self._at_fork = False
        possible_destinations = set(range(1, self.length)) - set(self._forks)
        self._destinations = sorted(
            random.sample(list(possible_destinations), k=self.num_packages)
        )
        self.og_destinations = copy.deepcopy(self._destinations)
        self.active_idx = 0
        self.achieved_goals = []
        self._correct_path = [random.choice([0, 1]) for _ in self._forks]
        self._packages_remaining = self.num_packages * 3
        self._correct_placement = random.choice([0, 1, 2, 3])
        return self.make_obs(), {}

    def step(self, action):
        reward = 0.0
        self._time += 1
        # saved for convenience in KGoal wrapper
        self.achieved_goals = []

        # use current pos to figure out fork logic
        try:
            fork_num = self._forks.index(self._pos)
        except ValueError:
            fork_num = None
        self._at_fork = fork_num is not None

        if action == 0:
            # move forward, if we can
            if not self._at_fork:
                self._pos += 1
        elif action == 1:
            if self._at_fork:
                correct_decision = self._correct_path[fork_num]
                if correct_decision == 0:
                    # go down hallway
                    self._pos += 1
                else:
                    # wrong; go back to start
                    self._pos = 0
        elif action == 2:
            if self._at_fork:
                correct_decision = self._correct_path[fork_num]
                if correct_decision == 1:
                    self._pos += 1
                else:
                    self._pos = 0
        elif action == 3:
            if self._at_fork:
                pass
            elif (
                self._correct_placement == 0
                and self._packages_remaining > 0
                and self._pos > 0
            ):
                self.achieved_goals = [self._pos]
                self._packages_remaining -= 1
                if self._pos == self._destinations[0]:
                    reward = 1.0
                    self.active_idx += 1
                    del self._destinations[0]
            else:
                self._pos = 0
        elif action == 4:
            if self._at_fork:
                pass
            elif (
                self._correct_placement == 1
                and self._packages_remaining > 0
                and self._pos > 0
            ):
                self.achieved_goals = [self._pos]
                self._packages_remaining -= 1
                if self._pos == self._destinations[0]:
                    reward = 1.0
                    self.active_idx += 1
                    del self._destinations[0]
            else:
                self._pos = 0
        elif action == 5:
            if self._at_fork:
                pass
            elif (
                self._correct_placement == 2
                and self._packages_remaining > 0
                and self._pos > 0
            ):
                self.achieved_goals = [self._pos]
                self._packages_remaining -= 1
                if self._pos == self._destinations[0]:
                    reward = 1.0
                    self.active_idx += 1
                    del self._destinations[0]
            else:
                self._pos = 0
        elif action == 6:
            if self._at_fork:
                pass
            elif (
                self._correct_placement == 3
                and self._packages_remaining > 0
                and self._pos > 0
            ):
                self.achieved_goals = [self._pos]
                self._packages_remaining -= 1
                if self._pos == self._destinations[0]:
                    reward = 1.0
                    self.active_idx += 1
                    del self._destinations[0]
            else:
                self._pos = 0
        elif action == 7:
            # no_op
            pass
        else:
            raise ValueError(f"Invalid action {action}")

        self._pos = self._pos % self.length  # wrap around
        self._at_fork = self._pos in self._forks
        done = self._time >= self.max_timesteps

        truncated = self._time >= self.max_timesteps
        terminated = len(self._destinations) == 0
        return self.make_obs(), reward, terminated, truncated, {}

    def make_obs(self):
        info = [self._pos, self._packages_remaining, self._at_fork]
        return np.array(info).astype(np.float32)
