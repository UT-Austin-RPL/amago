import random
import os
import math
import warnings
import copy
import pickle
from dataclasses import dataclass, asdict
from typing import Optional, Iterable
from abc import ABC, abstractmethod

import torch
import numpy as np
import gin

from amago import utils


@dataclass
class Timestep:
    obs: dict[str, np.ndarray]
    # action from the *previous* timestep
    prev_action: np.ndarray
    # reward from the previous timestep
    reward: float
    # time as an int
    time_idx: int
    # meta-rollout terminal signal
    terminal: bool

    def __eq__(self, other):
        if (
            (self.time_idx != other.time_idx)
            or (self.reward != other.reward)
            or (self.terminal != other.terminal)
        ):
            return False
        if (self.prev_action != other.prev_action).any():
            return False
        if len(self.obs.keys()) != len(other.obs.keys()):
            return False
        for (k1, v1), (k2, v2) in zip(self.obs.items(), other.obs.items()):
            if k1 != k2 or (v1 != v2).any():
                return False
        return True


@dataclass
class FrozenTraj:
    obs: dict[str, np.ndarray]
    rl2s: np.ndarray
    time_idxs: np.ndarray
    rews: np.ndarray
    dones: np.ndarray
    actions: np.ndarray

    def to_dict(self) -> dict[np.ndarray]:
        d = asdict(self)
        for obs_k, obs_v in d["obs"].items():
            # flatten obs dict but mark w/ special prefix
            d[f"_OBS_KEY_{obs_k}"] = obs_v
        del d["obs"]
        return d

    @staticmethod
    def from_dict(d: dict[np.ndarray]):
        args = {"obs": {}}
        for k, v in d.items():
            if k.startswith("_OBS_KEY_"):
                # fold the flattened obs dict back to original keys
                args["obs"][k.replace("_OBS_KEY_", "", 1)] = v
            else:
                args[k] = v
        return FrozenTraj(**args)


class Trajectory:
    def __init__(self, timesteps=Optional[Iterable[Timestep]]):
        self.timesteps = timesteps or []

    def add_timestep(self, timestep: Timestep):
        assert isinstance(timestep, Timestep)
        self.timesteps.append(timestep)

    @property
    def total_return(self):
        return sum([t.reward for t in self.timesteps])

    def __getitem__(self, i):
        return self.timesteps[i]

    def _make_sequence(self, timesteps) -> np.ndarray:
        obs = utils.stack_list_array_dicts([t.obs for t in timesteps], axis=0)
        actions = np.stack([t.prev_action for t in timesteps], axis=0)
        rews = np.stack([t.reward for t in timesteps], axis=0)[:, np.newaxis]
        rl2 = np.concatenate((rews, actions), axis=-1).astype(np.float32)
        return obs, rl2

    def make_sequence(self, last_only: bool = False):
        if last_only:
            return self._make_sequence([self.timesteps[-1]])
        else:
            return self._make_sequence(self.timesteps)

    def __len__(self):
        return len(self.timesteps)

    def save_to_disk(self, path: str, save_as: str):
        if save_as == "trajectory":
            with open(f"{path}.traj", "wb") as f:
                pickle.dump(self, f)
        elif save_as == "npz":
            frozen = self.freeze()
            np.savez(path, **frozen.to_dict())
        elif save_as == "npz-compressed":
            frozen = self.freeze()
            np.savez_compressed(path, **frozen.to_dict())
        else:
            raise ValueError(
                f"Unrecognized Trajectory `save_to_disk` format `save_as = {save_as}` (options are: 'trajectory' (slowest read/write, can relabel), 'npz' (fastest, cannot relabel), 'npz-compressed' (save space, cannot relabel)"
            )

    def freeze(self) -> FrozenTraj:
        obs, rl2s = self.make_sequence()
        time_idxs = np.array([t.time_idx for t in self.timesteps], dtype=np.int64)
        rews = np.array([t.reward for t in self.timesteps[1:]], dtype=np.float32)[
            :, np.newaxis
        ]
        dones = np.array([t.terminal for t in self.timesteps[1:]], dtype=bool)[
            :, np.newaxis
        ]
        actions = np.array(
            [t.prev_action for t in self.timesteps[1:]], dtype=np.float32
        )
        return FrozenTraj(
            obs=obs,
            rl2s=rl2s,
            time_idxs=time_idxs,
            rews=rews,
            dones=dones,
            actions=actions,
        )

    def __eq__(self, other):
        if len(other) != len(self):
            return False
        for t_self, t_other in zip(self.timesteps, other.timesteps):
            if t_self != t_other:
                return False
        return True


class Relabeler:

    def __call__(self, traj: Trajectory) -> Trajectory:
        return traj
