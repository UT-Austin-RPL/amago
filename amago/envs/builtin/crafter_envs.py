import math
import datetime
import time
import warnings
import copy
import random

try:
    import crafter
    import cv2
except ImportError:
    warnings.warn(
        "Missing crafter install; `pip install amago[envs] or `pip install crafter`",
        category=Warning,
    )
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from einops import rearrange

from amago.envs import AMAGOEnv
from amago.hindsight import GoalSeq
from amago.envs.env_utils import space_convert, DiscreteActionWrapper


class CrafterOptionalRender(crafter.Env):
    def __init__(self, reward, length, render: bool):
        super().__init__(reward=reward, length=length)

        self.enable_render = render

    def reset(self, *args, **kwargs):
        return super().reset()

    def render(self, mode="human"):
        return super().render(size=(512, 512))

    def _obs(self):
        # actual observation creation handled by CrafterEnv below
        if self.enable_render:
            return super().render(size=None)
        else:
            return None


class CrafterEnv(AMAGOEnv):
    TOKENS = {
        0: "collect",
        1: "defeat",
        2: "eat",
        3: "make",
        4: "place",
        5: "coal",
        6: "diamond",
        7: "drink",
        8: "iron",
        9: "sapling",
        10: "stone",
        11: "wood",
        12: "skeleton",
        13: "zombie",
        14: "cow",
        15: "plant",
        16: "pickaxe",
        17: "sword",
        18: "furnace",
        19: "table",
        20: "travel",
        21: "pad",
        22: "wake",
        23: "up",
    }
    # add travel coordinate tokens
    for i, m in enumerate(range(0, 61, 5)):
        TOKENS[24 + i] = f"{m}m"

    def __init__(
        self,
        directed: bool = True,
        k: int = 3,
        min_k: int = 1,
        time_limit: int = 5000,
        obs_kind: str = "render",
        use_tech_tree: bool = False,
        verbose: bool = False,
        save_video_to: str | None = None,
    ):
        env = CrafterOptionalRender(
            reward=not directed, length=time_limit, render=obs_kind != "textures"
        )
        if save_video_to is not None:
            env = crafter.recorder.VideoRecorder(env, directory=save_video_to)
            self.video_hook = env
        else:
            self.video_hook = None
        env = DiscreteActionWrapper(env)
        if obs_kind == "render":
            obs_shape = (64, 64, 3)
            self.observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0, high=255, shape=obs_shape, dtype=np.uint8
                    ),
                }
            )
        elif obs_kind == "crop":
            self.observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0, high=255, shape=(48, 64, 3), dtype=np.uint8
                    ),
                    "inventory": gym.spaces.Box(
                        low=0.0, high=1.0, shape=(16,), dtype=np.float32
                    ),
                }
            )
        elif obs_kind == "textures":
            self.observation_space = gym.spaces.Dict(
                {
                    "textures": gym.spaces.Box(
                        low=0, high=64, shape=(9, 7), dtype=np.uint8
                    ),
                    "info": gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(2 + 16 + 2 + 1,), dtype=np.float32
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unrecognized `obs_kind` '{obs_kind}'. Options are: render, crop, textures."
            )
        self.obs_kind = obs_kind
        self.clear_fixed_task()
        self.k = k
        self.min_k = min_k
        self.obs_kind = obs_kind
        self.dense_reward = not directed
        self._plotting = None
        self._ax = None
        self.verbose = verbose
        self.use_tech_tree = use_tech_tree
        self._best_tech_tree_idx = 0
        # skipping super().__init__() to handle observation space edge case
        self.horizon = time_limit
        self.start = 0
        self.discrete = True
        self.env = env
        self.action_size = env.action_space.n
        self.action_space = space_convert(env.action_space)
        self._env_name = (
            "Crafter" if self.dense_reward else "Crafter-Instruction-Conditioned"
        ) + self.obs_kind

    @property
    def env_name(self):
        return self._env_name

    def set_env_name(self, name: str):
        self._env_name = name

    @property
    def achieved_goal(self):
        seq = [np.array(g) for g in self._achieved_goals]
        if self.dense_reward:
            seq = [np.ones_like(g) for g in seq]
        return seq

    @property
    def goal_sequence(self):
        seq = [np.array(g) for g in self._task_sequence]
        if self.dense_reward:
            seq = [np.zeros_like(g) for g in seq]
        return GoalSeq(seq=seq, active_idx=self._active_idx)

    def get_game_info(self):
        info = {
            "inventory": self.env.unwrapped._player.inventory.copy(),
            "achievements": self.env.unwrapped._player.achievements.copy(),
            "semantic": self.env.unwrapped._sem_view(),
            "player_pos": self.env.unwrapped._player.pos,
        }
        return info

    @property
    def kgoal_space(self):
        return gym.spaces.Box(low=0, high=99, shape=(self.k, 3))

    def render(self, *args, **kwargs):
        if self.obs_kind == "textures":
            if not self._plotting:
                plt.switch_backend("tkagg")
                fig = plt.figure(figsize=(9, 6))
                self._ax = fig.add_subplot(111)
                plt.ion()
                self._plotting = True
            obs = self.env.render()
            obs = self.obs(obs, {}, kind="render")
            plt.tight_layout()
            plt.imshow(obs)
            plt.draw()
            plt.pause(0.001)
        else:
            return self.env.unwrapped.render()

    def obs(self, raw_obs, info, kind="textures"):
        assert kind in ["render", "crop", "textures"]

        if kind == "crop":
            scaled = raw_obs[:48, :, :]  # crop black inventory portion
            inv = (
                np.array(list(info["inventory"].values()), dtype=np.float32) / 10.0
            )  # get missing inventory
            return {"image": scaled, "inventory": inv}
        elif kind == "textures":
            gridworld_view = self.get_gridworld_view(info).astype(np.uint8) + 1
            h, w = gridworld_view.shape
            pad_view = np.zeros((7, 9), dtype=np.uint8)
            pad_view[:h, :w] = gridworld_view
            """
            # original version: breaks the spatial ordering of embeddings at the world
            # border when the grid is smaller than expected. I get this wrong in the paper
            # version and it leads to weird behavior at the edge of the map.
            pad_view = np.zeros((9 * 7,), dtype=np.uint8)
            pad_view[: len(gridworld_view)] = gridworld_view.flatten()
            """
            player_facing = np.array(self.env.unwrapped._player.facing)
            daylight = np.array(self.env.unwrapped._world.daylight / 1.0)[np.newaxis]
            inv = np.array(list(info["inventory"].values())) / 10.0
            pos = (np.array(info["player_pos"]).flatten() - 30.0) / 60.0
            return {
                "textures": pad_view,
                "info": np.concatenate((player_facing, inv, pos, daylight)).astype(
                    np.float32
                ),
            }
        else:
            obs = {"image": raw_obs}

        return obs

    def step(self, action):
        out = super().step(
            action, normal_rl_reward=self.dense_reward, normal_rl_reset=True
        )
        if self.verbose:
            for i, goal in enumerate(self.goal_sequence.seq):
                str = self._token2str(goal.astype(np.uint8).tolist())
                print(f"{' '.join(str)} {'DONE' if i < self._active_idx else ''}")
            print()
        return out

    def inner_step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.obs(next_state, info, kind=self.obs_kind)
        self._achieved_goals = self.goal_monitor(info)
        self._last_game_info = copy.deepcopy(info)
        if not self.dense_reward:
            reward = 0.0
            for achv in self._achieved_goals:
                if achv == self._task_sequence[self._active_idx]:
                    self._active_idx += 1
                    reward = 1.0
                    if self._active_idx >= len(self._task_sequence):
                        done = True
                    break
        return next_state, reward, done, False, info

    def inner_reset(self, seed=None, options=None):
        if self.video_hook is not None and self.video_hook._frames is not None:
            # navigating around the problem that the VideoRecorder only saves on
            # termination but success is determined in a wrapper above this.
            # not a perfect system - just here for the jupyter notebook demo.
            self.video_hook._env._timestamp = datetime.datetime.now().strftime(
                "%Y%m%dT%H%M%S"
            )
            self.video_hook._save()
        obs = self.env.reset()
        info = self.get_game_info()
        self._last_game_info = copy.deepcopy(info)
        self._achieved_goals = self.goal_monitor(info)
        self._task_sequence = self.generate_task()
        self._active_idx = 0
        return self.obs(obs, info, kind=self.obs_kind), {}

    def _dict_delta(self, dict_t, dict_t1):
        delta = {key: dict_t1[key] - dict_t[key] for key in dict_t1 if key in dict_t}
        return delta

    def _str2token(self, words: list[str]):
        str2token_map = {v: k for k, v in self.TOKENS.items()}
        tokens = [str2token_map[s] for s in words]
        while len(tokens) < 3:
            tokens.append(str2token_map["pad"])
        return tokens

    def _token2str(self, tokens: list[int]):
        words = [self.TOKENS[t] for t in tokens]
        if "pad" in words:
            words.remove("pad")
        return words

    """
    Goal frequencies are conditioned on their type (the first word of the goal).
    TASKS[type][idx] = (goal, probability)
    """

    TASKS = {
        "collect": [
            [
                ("sapling", 0.2),
                ("wood", 0.2),
                ("stone", 0.18),
                ("drink", 0.17),
                ("coal", 0.1),
                ("iron", 0.1),
                ("diamond", 0.05),
            ]
        ],
        "defeat": [[("skeleton", 0.4), ("zombie", 0.6)]],
        "eat": [[("cow", 0.7), ("plant", 0.3)]],
        "make": [
            [("iron", 0.1), ("stone", 0.3), ("wood", 0.6)],
            [("pickaxe", 0.5), ("sword", 0.5)],
        ],
        "place": [[("furnace", 0.1), ("plant", 0.2), ("stone", 0.3), ("table", 0.4)]],
        "travel": [
            (f"{x}m", f"{y}m", math.sqrt((x - 32) ** 2 + (y - 32) ** 2))
            for x in range(0, 61, 5)
            for y in range(0, 61, 5)
        ],
    }

    TECH_TREE = [
        "collect_wood",
        "place_table",
        "make_wood_pickaxe",
        "collect_coal",
        "make_wood_sword",
        "collect_stone",
        "make_stone_pickaxe",
        "make_stone_sword",
        "collect_iron",
        "place_furnace",
        "make_iron_pickaxe",
        "make_iron_sword",
        "collect_diamond",
    ]

    def set_fixed_task(self, task: list[str]):
        self._fixed_task = [self._str2token(g) for g in task]

    def clear_fixed_task(self):
        self._fixed_task = None

    def generate_task(self):
        if self._fixed_task is not None:
            return self._fixed_task
        if self.use_tech_tree and random.random() < 0.333:
            # 1/3 number was not tuned
            return self._generate_tech_tree_task()
        return self._generate_random_task()

    def _generate_tech_tree_task(self):
        tech_tree_limit = min(self._best_tech_tree_idx + 1, len(self.TECH_TREE) - 1)
        k_this_ep = random.randint(self.min_k, min(self.k, tech_tree_limit))
        idxs = sorted(random.sample(range(tech_tree_limit), k=k_this_ep))
        subgoals = [self.TECH_TREE[idx] for idx in idxs]
        task = [self._str2token(g.split("_")) for g in subgoals]
        return task

    def _generate_random_task(self):
        def _choices(options):
            choices = [c[0] for c in options]
            weights = [c[1] for c in options]
            return choices, weights

        def _sample(tasks, weights):
            return random.choices(tasks, weights=weights, k=1)[0]

        def _sample_task(key):
            task = [key]
            for option in self.TASKS[key]:
                choices, weights = _choices(option)
                task.append(_sample(choices, weights))
            return task

        goal_seq = []
        k_this_ep = random.randint(self.min_k, self.k)
        for _ in range(k_this_ep):
            r = random.random()
            # these probabilities look specific but were not tuned and would
            # probably be sub-optimal if we really cared about Crafter as an end goal...
            if r < 0.3:
                goal = _sample_task("collect")
            elif r < 0.4:
                goal = _sample_task("defeat")
            elif r < 0.5:
                goal = _sample_task("eat")
            elif r < 0.65:
                goal = _sample_task("make")
            elif r < 0.8:
                goal = _sample_task("place")
            else:
                dist = list(range(1, 45))
                probs = np.flip(np.arange(1, 45, dtype=np.float32))
                probs /= probs.sum()
                probs = probs.tolist()
                chosen_dist = random.choices(dist, weights=probs, k=1)[0]
                sorted_travel_tasks = sorted(
                    self.TASKS["travel"],
                    key=lambda j: j[-1] + random.random(),
                    reverse=True,
                )
                for x, y, dist in sorted_travel_tasks:
                    if dist <= chosen_dist:
                        goal = ["travel", x, y]
                        break
                else:
                    goal = ["travel", x, y]
            goal_seq.append(self._str2token(goal))
        return goal_seq

    def get_gridworld_view(self, game_info):
        view = game_info["semantic"]
        y, x = game_info["player_pos"]
        c = lambda j: max(min(j, 64), 0)
        local_view = view.T[c(x - 3) : c(x + 4), c(y - 4) : c(y + 5)]
        return local_view

    def goal_monitor(self, game_info):
        achieved_goals = []
        achv_delta = self._dict_delta(
            self._last_game_info["achievements"], game_info["achievements"]
        )
        for key, value in achv_delta.items():
            if value > 0:
                if key in self.TECH_TREE:
                    tech_tree_idx = self.TECH_TREE.index(key)
                    if tech_tree_idx > self._best_tech_tree_idx:
                        self._best_tech_tree_idx = tech_tree_idx
                tokens = self._str2token(key.split("_"))
                achieved_goals.append(tokens)
        x, y = game_info["player_pos"]

        clip = lambda z: max(min(z, 60), 0)
        x_low, x_high = clip(math.floor(x / 5.0) * 5), clip(math.ceil(x / 5.0) * 5)
        y_low, y_high = clip(math.floor(y / 5.0) * 5), clip(math.ceil(y / 5.0) * 5)
        dists = []
        for x_cand in [x_low, x_high]:
            for y_cand in [y_low, y_high]:
                dist = math.sqrt((x - x_cand) ** 2 + (y - y_cand) ** 2)
                dists.append((x_cand, y_cand, dist))
        dists = sorted(dists, key=lambda j: j[-1])
        closest_x, closest_y, closest_dist = dists[0]
        if closest_dist < 2.0:
            # travel goals have an error tolerance of 2 units
            achieved_goals.append(
                self._str2token(["travel", f"{closest_x}m", f"{closest_y}m"])
            )

        if self.verbose:
            print(f"Achieved: {[self._token2str(g) for g in achieved_goals]}")
        return achieved_goals


if __name__ == "__main__":
    env = CrafterEnv(obs_kind="crop")
    env.reset()
    env.step(np.array(env.action_space.sample(), dtype=np.uint8))
