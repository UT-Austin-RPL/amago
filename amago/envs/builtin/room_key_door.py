import random

import numpy as np
import gymnasium as gym


class RoomKeyDoor(gym.Env):
    """
    A version of the Dark Room Key-Door Env.

    Based on Algorithm Distillation (Laskin et al., 2022)
    """

    def __init__(
        self,
        dark: bool = True,
        size: int = 9,
        max_episode_steps: int = 50,
        # the appendix version of this experiment (vs. AD)
        # used a fixed starting location to be consistent
        # with the version in our AD replication. The main
        # text version (vs. AMAGO-GRU) uses the random starts.
        start_location: tuple[int, int] | str = "random",
        key_location: tuple[int, int] | str = "random",
        goal_location: tuple[int, int] | str = "random",
    ):
        self.dark = dark
        self.size = size
        self.H = max_episode_steps
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4 if self.dark else 8,)
        )
        self.action_space = gym.spaces.Discrete(5)
        self.goal_location = goal_location
        self.key_location = key_location
        self.start_location = start_location

    def reset(self, new_task=True, **kwargs):
        if new_task:
            start, key, goal = self.generate_task()
            self.goal = np.array(goal)
            self.key = np.array(key)
            self.start = np.array(start)
        self.has_key = False
        self.t = 0
        self.pos = self.start
        return self.obs(), {}

    def generate_task(self):
        start = (
            random.choices(range(self.size), k=2)
            if self.start_location == "random"
            else self.start_location
        )
        key = (
            random.choices(range(self.size), k=2)
            if self.key_location == "random"
            else self.key_location
        )
        goal = (
            random.choices(range(self.size), k=2)
            if self.goal_location == "random"
            else self.goal_location
        )
        return start, key, goal

    def step(self, action: int):
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
        self.pos = np.clip(self.pos + np.array(dirs[action]), 0, self.size - 1)
        reward = 0.0
        terminated = False
        if self.has_key and (self.pos == self.goal).all():
            reward = 1.0
            terminated = True
        elif not self.has_key and (self.pos == self.key).all():
            reward = 1.0
            self.has_key = True
        truncated = self.t >= self.H
        self.t += 1
        return self.obs(), reward, terminated, truncated, {}

    def obs(self):
        x, y = self.pos
        norm = lambda j: float(j) / self.size
        # time and has_key keep this fully observed
        base = [norm(x), norm(y), self.has_key, float(self.t) / self.H]
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


if __name__ == "__main__":
    env = RoomKeyDoor(
        dark=True,
        size=9,
        max_episode_steps=50,
        key_location="random",
        goal_location="random",
        start_location="random",
    )

    from amago.envs.builtin import GymEnv
    import time

    env = GymEnv(
        env,
        env_name="darkroom",
        horizon=500,
        start=0,
        zero_shot=False,
        convert_from_old_gym=False,
        soft_reset_kwargs={"new_task": False},
    )

    for _ in range(10):
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            env.render()
            """
            action = input()
            raw_action = {"a": 0, "w": 1, "d": 2, "s": 3, "n": 4}[action]
            """
            raw_action = np.array(env.action_space.sample(), dtype=np.uint8)
            next_state, reward, terminated, truncated, info = env.step(raw_action)
            time.sleep(0.1)
            """
            print(next_state)
            print(reward)
            print(f"terminated: {terminated}")
            print(f"truncated: {truncated}")
            """
