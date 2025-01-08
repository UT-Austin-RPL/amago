import os
import random
import shutil
import pickle
from dataclasses import dataclass
from operator import itemgetter
from functools import partial
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

from .hindsight import Trajectory, Relabeler, FrozenTraj


def load_traj_from_disk(path: str) -> Trajectory | FrozenTraj:
    _, ext = os.path.splitext(path)
    if ext == ".traj":
        with open(path, "rb") as f:
            disk = pickle.load(f)
        traj = Trajectory(timesteps=disk.timesteps)
        return traj
    elif ext == ".npz":
        disk = FrozenTraj.from_dict(np.load(path))
        return disk
    else:
        raise ValueError(
            f"Unrecognized trajectory file extension `{ext}` for path `{path}`."
        )


def get_path_to_trajs(dset_root: str, dset_name: str, fifo: bool) -> str:
    return os.path.join(dset_root, dset_name, "buffer", "fifo" if fifo else "protected")


class TrajDset(Dataset):
    """
    Load trajectory files from disk in parallel with pytorch Dataset/DataLoader
    pipeline.
    """

    def __init__(
        self,
        relabeler: Relabeler,
        dset_root: str,
        dset_name: str,
        items_per_epoch: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        padded_sampling: str = "none",
    ):
        assert dset_root is not None and os.path.exists(dset_root)
        self.relabeler = relabeler
        self.fifo_path = get_path_to_trajs(dset_root, dset_name, fifo=True)
        self.protected_path = get_path_to_trajs(dset_root, dset_name, fifo=False)
        os.makedirs(self.fifo_path, exist_ok=True)
        os.makedirs(self.protected_path, exist_ok=True)
        self.length = items_per_epoch
        self.max_seq_len = max_seq_len
        assert padded_sampling in ["none", "right", "left", "both"]
        self.padded_sampling = padded_sampling
        self.refresh_files()

    def __len__(self):
        # this length is used by DataLoaders to end an epoch
        if self.length is None:
            return self.count_trajectories()
        else:
            return self.length

    @property
    def disk_usage(self):
        bytes = sum(os.path.getsize(f) for f in self.all_filenames)
        return bytes * 1e-9

    def clear(self, delete_protected: bool = False):
        # remove files on disk
        if os.path.exists(self.fifo_path):
            shutil.rmtree(self.fifo_path)
            self.fifo_filenames = set()
        if os.path.exists(self.protected_path) and delete_protected:
            shutil.rmtree(self.protected_path)
            self.protected_filenames = set()
        self.all_filenames = list(self.fifo_filenames | self.protected_filenames)

    def _list_abs_path_to_files(self, dir: str):
        names = []
        for ext in [".traj", ".npz"]:
            names.extend(
                [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext)]
            )
        return set(names)

    def refresh_files(self):
        # find the new .traj files from the previous rollout
        self.fifo_filenames = self._list_abs_path_to_files(self.fifo_path)
        self.protected_filenames = self._list_abs_path_to_files(self.protected_path)
        self.all_filenames = list(self.fifo_filenames | self.protected_filenames)

    def count_trajectories(self) -> int:
        return len(self.all_filenames)

    def count_deletable_trajectories(self) -> int:
        return len(self.fifo_filenames)

    def count_protected_trajectories(self) -> int:
        return len(self.protected_filenames)

    def filter(self, new_size: int):
        """
        Imitates fixed-size FIFO replay buffers by clearing .traj files on disk in time order.
        """
        if len(self.fifo_filenames) <= new_size:
            # skip the sort
            return

        path_to_name = lambda path: os.path.basename(os.path.splitext(path)[0])
        traj_infos = []
        for traj_filename in self.fifo_filenames:
            env_name, rand_id, unix_time = path_to_name(traj_filename).split("_")
            traj_infos.append(
                {
                    "env": env_name,
                    "rand": rand_id,
                    "time": float(unix_time),
                    "filename": traj_filename,
                }
            )
        traj_infos = sorted(traj_infos, key=lambda d: d["time"])
        num_to_remove = max(len(traj_infos) - new_size, 0)
        to_delete = list(map(itemgetter("filename"), traj_infos[:num_to_remove]))
        for file_to_delete in to_delete:
            os.remove(file_to_delete)
            self.fifo_filenames.discard(file_to_delete)
        self.all_filenames = list(self.fifo_filenames | self.protected_filenames)

    def __getitem__(self, i):
        filename = random.choice(self.all_filenames)
        traj = load_traj_from_disk(filename)
        traj = self.relabeler(traj)
        data = RLData(traj)
        if self.max_seq_len is not None:
            data = data.random_slice(
                length=self.max_seq_len, padded_sampling=self.padded_sampling
            )
        return data


class RLData:
    def __init__(self, traj: Trajectory | FrozenTraj):
        if isinstance(traj, Trajectory):
            traj = traj.freeze()
        assert isinstance(traj, FrozenTraj)
        self.obs = {k: torch.from_numpy(v) for k, v in traj.obs.items()}
        self.rl2s = torch.from_numpy(traj.rl2s).float()
        self.time_idxs = torch.from_numpy(traj.time_idxs).long()
        self.rews = torch.from_numpy(traj.rews).float()
        self.dones = torch.from_numpy(traj.dones).bool()
        self.actions = torch.from_numpy(traj.actions).float()
        self.safe_randrange = lambda l, h: random.randrange(l, max(h, l + 1))

    def __len__(self):
        return len(self.actions)

    def random_slice(self, length: int, padded_sampling: str = "none"):
        if len(self) <= length:
            start = 0
        elif padded_sampling == "none":
            start = self.safe_randrange(0, len(self) - length + 1)
        elif padded_sampling == "both":
            start = self.safe_randrange(-length + 1, len(self) - 1)
        elif padded_sampling == "left":
            start = self.safe_randrange(-length + 1, len(self) - length + 1)
        elif padded_sampling == "right":
            start = self.safe_randrange(0, len(self) - 1)
        else:
            raise ValueError(
                f"Unrecognized `padded_sampling` mode: `{padded_sampling}`"
            )
        stop = start + length
        start = max(start, 0)
        # the causal RL loss requires these off-by-one lengths
        tcp = slice(start, stop + 1)
        tc = slice(start, stop)
        self.obs = {k: v[tcp] for k, v in self.obs.items()}
        self.rl2s = self.rl2s[tcp]
        self.time_idxs = self.time_idxs[tcp]
        self.dones = self.dones[tc]
        self.rews = self.rews[tc]
        self.actions = self.actions[tc]
        return self


MAGIC_PAD_VAL = 4.0
pad = partial(pad_sequence, batch_first=True, padding_value=MAGIC_PAD_VAL)


@dataclass
class Batch:
    """
    Keeps data organized during training step
    """

    obs: dict[torch.Tensor]
    rl2s: torch.Tensor
    rews: torch.Tensor
    dones: torch.Tensor
    actions: torch.Tensor
    time_idxs: torch.Tensor

    def to(self, device):
        self.obs = {k: v.to(device) for k, v in self.obs.items()}
        self.rl2s = self.rl2s.to(device)
        self.rews = self.rews.to(device)
        self.dones = self.dones.to(device)
        self.actions = self.actions.to(device)
        self.time_idxs = self.time_idxs.to(device)
        return self


def RLData_pad_collate(samples: list[RLData]) -> Batch:
    assert samples[0].obs.keys() == samples[-1].obs.keys()
    obs = {k: pad([s.obs[k] for s in samples]) for k in samples[0].obs.keys()}
    rl2s = pad([s.rl2s for s in samples])
    rews = pad([s.rews for s in samples])
    dones = pad([s.dones for s in samples])
    actions = pad([s.actions for s in samples])
    time_idxs = pad([s.time_idxs for s in samples])
    return Batch(
        obs=obs,
        rl2s=rl2s,
        rews=rews,
        dones=dones,
        actions=actions,
        time_idxs=time_idxs,
    )
