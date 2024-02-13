import random
import warnings

try:
    import metaworld
except:
    warnings.warn(
        "Missing metaworld Install: `pip install amago[envs]` or `pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb`",
        category=Warning,
    )
import gymnasium as gym
import numpy as np

from amago.envs.env_utils import space_convert
from amago.envs.builtin.gym_envs import GymEnv


class Metaworld(GymEnv):
    def __init__(self, benchmark_name: str, split: str, k_shots):
        if benchmark_name == "ml10":
            benchmark = metaworld.ML10()
        elif benchmark_name == "ml45":
            benchmark = metaworld.ML45()
        else:
            try:
                benchmark = metaworld.ML1(benchmark_name)
            except:
                raise ValueError(f"Unrecognized metaworld benchmark {benchmark_name}")

        env = KShotMetaworld(benchmark, split, k_shots)
        super().__init__(
            gym_env=env,
            env_name=f"metaworld_{benchmark_name}",
            horizon=k_shots * 500 + 1,
            start=0,
            zero_shot=True,
        )

    @property
    def env_name(self):
        return self.env.task_name


class KShotMetaworld(gym.Env):
    reward_scales = {}

    def __init__(self, benchmark, split: str, k_shots: int):
        assert split in ["train", "test"]
        self.benchmark = benchmark
        self.split = split
        self.k_shots = k_shots
        self._envs = self.get_env_funcs(benchmark, train_set=split == "train")
        self.reset()
        self.action_space = space_convert(self.env.action_space)
        og_obs = space_convert(self.env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=np.asarray(og_obs.low.tolist() + [0]),
            high=np.asarray(og_obs.high.tolist() + [1.0]),
        )

    def get_env_funcs(self, benchmark, train_set: bool):
        classes = benchmark.train_classes if train_set else benchmark.test_classes
        tasks = benchmark.train_tasks if train_set else benchmark.test_tasks
        env_tasks = {}
        for name, env_cls in classes.items():
            all_tasks = [task for task in tasks if task.env_name == name]
            env_tasks[name] = (env_cls(), all_tasks)
        return env_tasks

    def soft_reset(self):
        self.current_time = 0
        obs = self

    def get_obs(self, og_obs, soft_reset: bool):
        return np.concatenate((og_obs, [float(soft_reset)])).astype(np.float32)

    def reset(self, *args, **kwargs):
        self.current_trial = 0
        self.current_time = 0
        self.successes = 0
        self.trial_success = 0.0
        new_task = random.choice(list(self._envs.keys()))
        self.env, tasks = self._envs[new_task]
        self.task_name = new_task
        self.env.set_task(random.choice(tasks))
        obs = self.env.reset()
        # print(f"Task: {self.task_name}, Target Pos: {self.env._target_pos}, Object Pos: {self.env.obj_init_pos}")
        return self.get_obs(obs, True), {}

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self.trial_success = max(info["success"], self.trial_success)
        self.current_time += 1

        soft_reset = False
        if self.current_time >= 500 or done:
            soft_reset = True
            self.current_time = 0
            self.successes += self.trial_success
            self.trial_success = 0.0
            next_obs = self.env.reset()
            self.current_trial += 1

        truncated = terminated = self.current_trial >= self.k_shots

        if self.task_name in self.reward_scales:
            reward *= self.reward_scales[self.task_name]

        return (
            self.get_obs(next_obs, soft_reset),
            reward,
            terminated,
            truncated,
            {"success": self.successes},
        )


if __name__ == "__main__":
    benchmark = metaworld.ML45()
    env = KShotMetaworld(benchmark, "train", k_shots=10)
    for ep in range(10):
        print(env.task_name)
        input()
        env.reset()
        done = False
        step = 0
        while not done:
            print(f"{ep}, {step}")
            next_state, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            done = terminated or truncated
            step += 1
