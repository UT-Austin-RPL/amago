import os
import random
import shutil
import pickle
from dataclasses import dataclass
from operator import itemgetter
from functools import partial
from typing import Optional, Any
from abc import ABC, abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from accelerate import Accelerator
import gin

from .hindsight import Trajectory, Relabeler, FrozenTraj


@dataclass
class RLData:
    obs: dict[str, torch.Tensor]
    rl2s: torch.FloatTensor
    rews: torch.FloatTensor
    dones: torch.BoolTensor
    actions: torch.FloatTensor
    time_idxs: torch.LongTensor

    def __len__(self):
        return len(self.actions)

    def random_slice(self, length: int, padded_sampling: str = "none"):
        _safe_randrange = lambda l, h: random.randrange(l, max(h, l + 1))
        if len(self) <= length:
            start = 0
        elif padded_sampling == "none":
            start = _safe_randrange(0, len(self) - length + 1)
        elif padded_sampling == "both":
            start = _safe_randrange(-length + 1, len(self) - 1)
        elif padded_sampling == "left":
            start = _safe_randrange(-length + 1, len(self) - length + 1)
        elif padded_sampling == "right":
            start = _safe_randrange(0, len(self) - 1)
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


class RLDataset(ABC, Dataset):
    def __init__(self):
        self.experiment = None

    def configure_from_experiment(self, experiment):
        self.experiment = experiment
        self.items_per_epoch = (
            experiment.train_batches_per_epoch
            * experiment.batch_size
            * experiment.accelerator.num_processes
        )
        self.max_seq_len = experiment.max_seq_len
        self.padded_sampling = experiment.padded_sampling
        self.max_size = experiment.dset_max_size
        self.has_edit_rights = experiment.has_dset_edit_rights

    def check_configured(self):
        if self.experiment is None:
            raise ValueError(
                "Dataset not configured. Call `configure_from_experiment()` first."
            )

    def __len__(self):
        self.check_configured()
        return self.items_per_epoch

    @property
    @abstractmethod
    def save_new_trajs_to(self) -> Optional[str]:
        raise NotImplementedError

    @property
    def ready_for_training(self) -> bool:
        return self.experiment is not None

    def on_end_of_epoch(self, epoch: int) -> dict[str, Any]:
        return {}

    def delete(self, delete_protected: bool = False):
        pass

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sample_random_trajectory(self) -> RLData:
        raise NotImplementedError

    def __getitem__(self, i):
        self.check_configured()
        data = self.sample_random_trajectory()
        if self.max_seq_len is not None:
            data = data.random_slice(
                length=self.max_seq_len, padded_sampling=self.padded_sampling
            )
        return data


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


@gin.configurable
class DiskTrajDataset(RLDataset):
    """
    Load trajectory files saved from the AMAGOEnvs (the default in most cases)
    """

    def __init__(
        self,
        dset_root: str,
        dset_name: str,
        relabeler: Optional[Relabeler] = None,
    ):
        super().__init__()
        self.relabeler = Relabeler() if relabeler is None else relabeler
        # create two directories for the FIFO and protected buffers
        self.fifo_path = get_path_to_trajs(dset_root, dset_name, fifo=True)
        self.protected_path = get_path_to_trajs(dset_root, dset_name, fifo=False)
        os.makedirs(self.fifo_path, exist_ok=True)
        os.makedirs(self.protected_path, exist_ok=True)
        self._refresh_files()

    @property
    def save_new_trajs_to(self) -> Optional[str]:
        return self.fifo_path

    @property
    def _disk_usage(self):
        bytes = sum(os.path.getsize(f) for f in self.all_filenames)
        return bytes * 1e-9

    @property
    def ready_for_training(self) -> bool:
        return super().ready_for_training and len(self.all_filenames) > 0

    def delete(self, delete_protected: bool = False):
        self.check_configured()
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

    def _refresh_files(self):
        # find the new .traj files from the previous rollout
        self.fifo_filenames = self._list_abs_path_to_files(self.fifo_path)
        self.protected_filenames = self._list_abs_path_to_files(self.protected_path)
        self.all_filenames = list(self.fifo_filenames | self.protected_filenames)

    def _count_trajectories(self) -> int:
        return len(self.all_filenames)

    def _count_deletable_trajectories(self) -> int:
        return len(self.fifo_filenames)

    def _count_protected_trajectories(self) -> int:
        return len(self.protected_filenames)

    def _filter(self):
        """
        Imitates fixed-size FIFO replay buffers by clearing .traj files on disk in time order.
        """
        if len(self.fifo_filenames) <= self.max_size:
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
        num_to_remove = max(len(traj_infos) - self.max_size, 0)
        to_delete = list(map(itemgetter("filename"), traj_infos[:num_to_remove]))
        for file_to_delete in to_delete:
            os.remove(file_to_delete)
            self.fifo_filenames.discard(file_to_delete)
        self.all_filenames = list(self.fifo_filenames | self.protected_filenames)

    def get_description(self) -> str:
        self.check_configured()
        return f"""DiskTrajDataset
        \t\t FIFO Buffer Path: {self.fifo_path}
        \t\t Protected Buffer Path: {self.protected_path}
        \t\t FIFO Buffer Max Size: {self.max_size}
        \t\t FIFO Buffer Initial Size: {self._count_deletable_trajectories()}
        \t\t Protected Buffer Initial Size: {self._count_protected_trajectories()}
        \t\t Trajectory File Max Sequence Length: {self.max_seq_len}
        \t\t Trajectory Padded Sampling: {self.padded_sampling}
        """

    def on_end_of_epoch(self, epoch: int) -> dict[str, Any]:
        self._refresh_files()
        if not self.has_edit_rights:
            return
        old_size = self._count_trajectories()
        self.experiment.accelerator.wait_for_everyone()
        if self.experiment.accelerator.is_main_process:
            self._filter()
        self.experiment.accelerator.wait_for_everyone()
        self._refresh_files()
        dset_size = self._count_trajectories()
        fifo_size = self._count_deletable_trajectories()
        protected_size = self._count_protected_trajectories()
        return {
            "Trajectory Files Saved in FIFO Replay Buffer": fifo_size,
            "Trajectory Files Saved in Protected Replay Buffer": protected_size,
            "Total Trajectory Files in Replay Buffer": dset_size,
            "Trajectory Files Deleted": old_size - dset_size,
            "Buffer Disk Space (GB)": self._disk_usage,
        }

    def sample_random_trajectory(self) -> RLData:
        self.check_configured()
        filename = random.choice(self.all_filenames)
        traj = load_traj_from_disk(filename)
        traj = self.relabeler(traj)
        return self._traj_to_rl_data(traj)

    def _traj_to_rl_data(self, traj: Trajectory | FrozenTraj) -> RLData:
        if isinstance(traj, Trajectory):
            traj = traj.freeze()
        assert isinstance(traj, FrozenTraj)
        obs = {k: torch.from_numpy(v) for k, v in traj.obs.items()}
        rl2s = torch.from_numpy(traj.rl2s).float()
        time_idxs = torch.from_numpy(traj.time_idxs).long()
        rews = torch.from_numpy(traj.rews).float()
        dones = torch.from_numpy(traj.dones).bool()
        actions = torch.from_numpy(traj.actions).float()
        return RLData(
            obs=obs,
            rl2s=rl2s,
            time_idxs=time_idxs,
            rews=rews,
            dones=dones,
            actions=actions,
        )
