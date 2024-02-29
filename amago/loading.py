import os
import random
import time
import shutil
from dataclasses import dataclass
from operator import itemgetter
from functools import partial, lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

from .hindsight import Trajectory, Relabeler


# @lru_cache(maxsize=1000)
def load_traj_from_disk(path: str) -> Trajectory:
    traj = Trajectory.load_from_disk(path)
    return traj


class TrajDset(Dataset):
    """
    Load trajectory files from disk in parallel with pytorch Dataset/DataLoader
    pipeline.
    """

    def __init__(
        self,
        relabeler: Relabeler,
        dset_root: str = None,
        dset_name: str = None,
        dset_split: str = "train",
        items_per_epoch: int = None,
        max_seq_len: int = None,
    ):
        assert dset_split in ["train", "val", "test"]
        assert dset_root is not None and os.path.exists(dset_root)
        self.max_seq_len = max_seq_len
        self.dset_split = dset_split
        self.dset_path = (
            os.path.join(dset_root, dset_name, dset_split) if dset_name else None
        )
        self.length = items_per_epoch if dset_name else None
        self.filenames = []
        self.refresh_files()
        self.relabeler = relabeler

    def __len__(self):
        # this length is used by DataLoaders to end an epoch
        if self.length is None:
            return self.count_trajectories()
        else:
            return self.length

    @property
    def disk_usage(self):
        bytes = sum(
            os.path.getsize(os.path.join(self.dset_path, f)) for f in self.filenames
        )
        return bytes * 1e-9

    def clear(self):
        # remove files on disk
        if os.path.exists(self.dset_path):
            shutil.rmtree(self.dset_path)
            os.makedirs(self.dset_path)

    def refresh_files(self):
        # find the new .traj files from the previous rollout
        if self.dset_path is not None and os.path.exists(self.dset_path):
            self.filenames = os.listdir(self.dset_path)

    def count_trajectories(self) -> int:
        # get the real dataset size
        return len(self.filenames)

    def filter(self, delete_pct: float):
        """
        Imitates fixed-size replay buffers by clearing .traj files on disk.
        """
        assert delete_pct <= 1.0 and delete_pct >= 0.0

        traj_infos = []
        for traj_filename in self.filenames:
            env_name, rand_id, unix_time = os.path.splitext(traj_filename)[0].split("_")
            time, _ = unix_time.split(".")
            traj_infos.append(
                {
                    "env": env_name,
                    "rand": rand_id,
                    "time": int(time),
                    "filename": traj_filename,
                }
            )
        traj_infos = sorted(traj_infos, key=lambda d: d["time"])
        num_to_remove = round(len(traj_infos) * delete_pct)
        to_delete = list(map(itemgetter("filename"), traj_infos[:num_to_remove]))
        for file_to_delete in to_delete:
            os.remove(os.path.join(self.dset_path, file_to_delete))

    def __getitem__(self, i):
        filename = random.choice(self.filenames)
        traj = load_traj_from_disk(os.path.join(self.dset_path, filename))
        hseq = self.relabeler(traj)
        data = RLData(hseq)
        if self.max_seq_len is not None:
            data = data.random_slice(length=self.max_seq_len)
        return data


class RLData:
    def __init__(self, traj: Trajectory):
        if traj.frozen:
            obs = traj._frozen_obs
            goals = traj._frozen_goals
            rl2s = traj._frozen_rl2s
        else:
            obs, goals, rl2s = traj.make_sequence()
        # dtype cast needs to happen inside TstepEncoder
        self.obs = {k: torch.from_numpy(v) for k, v in obs.items()}
        self.goals = torch.from_numpy(goals).float()
        self.rl2s = torch.from_numpy(rl2s).float()
        self.time_idxs = torch.Tensor([t.raw_time_idx for t in traj.timesteps]).long()
        # rews, dones, and actions are shifted back by one timestep
        to_torch = lambda x: torch.from_numpy(np.array(x))
        self.rews = (
            to_torch([t.reward for t in traj.timesteps[1:]]).float().unsqueeze(-1)
        )
        self.dones = (
            to_torch([t.terminal for t in traj.timesteps[1:]]).bool().unsqueeze(-1)
        )
        self.actions = to_torch([t.prev_action for t in traj.timesteps[1:]]).float()

    def __len__(self):
        return len(self.actions)

    def random_slice(self, length: int):
        i = random.randrange(0, max(len(self) - length + 1, 1))
        # the causal RL loss requires these off-by-one lengths
        self.obs = {k: v[i : i + length + 1] for k, v in self.obs.items()}
        self.goals = self.goals[i : i + length + 1]
        self.rl2s = self.rl2s[i : i + length + 1]
        self.time_idxs = self.time_idxs[i : i + length + 1]
        self.dones = self.dones[i : i + length]
        self.rews = self.rews[i : i + length]
        self.actions = self.actions[i : i + length]
        return self


MAGIC_PAD_VAL = 0
pad = partial(pad_sequence, batch_first=True, padding_value=MAGIC_PAD_VAL)


@dataclass
class Batch:
    """
    Keeps data organized during training step
    """

    obs: dict[torch.Tensor]
    goals: torch.Tensor
    rl2s: torch.Tensor
    rews: torch.Tensor
    dones: torch.Tensor
    actions: torch.Tensor
    time_idxs: torch.Tensor

    def to(self, device):
        self.obs = {k: v.to(device) for k, v in self.obs.items()}
        self.goals = self.goals.to(device)
        self.rl2s = self.rl2s.to(device)
        self.rews = self.rews.to(device)
        self.dones = self.dones.to(device)
        self.actions = self.actions.to(device)
        self.time_idxs = self.time_idxs.to(device)


def RLData_pad_collate(samples: list[RLData]) -> Batch:
    assert samples[0].obs.keys() == samples[-1].obs.keys()
    obs = {k: pad([s.obs[k] for s in samples]) for k in samples[0].obs.keys()}
    goals = pad([s.goals for s in samples])
    rl2s = pad([s.rl2s for s in samples])
    rews = pad([s.rews for s in samples])
    dones = pad([s.dones for s in samples])
    actions = pad([s.actions for s in samples])
    time_idxs = pad([s.time_idxs for s in samples])
    return Batch(
        obs=obs,
        goals=goals,
        rl2s=rl2s,
        rews=rews,
        dones=dones,
        actions=actions,
        time_idxs=time_idxs,
    )
