import pickle
from dataclasses import dataclass, asdict
from typing import Optional, Iterable

import numpy as np

from amago import utils


@dataclass
class Timestep:
    obs: dict[str, np.ndarray]
    # action from the *previous* timestep
    prev_action: np.ndarray
    # reward from the previous timestep
    reward: np.ndarray
    # time as an int
    time_idx: np.ndarray
    # meta-rollout terminal signal
    terminal: np.ndarray
    # environment batch dimension
    batched_envs: int

    def __post_init__(self):
        assert (
            self.prev_action.ndim == 2
            and self.prev_action.shape[0] == self.batched_envs
        )
        assert self.reward.ndim == 1 and self.reward.shape[0] == self.batched_envs
        assert self.time_idx.ndim == 1 and self.time_idx.shape[0] == self.batched_envs
        assert self.terminal.ndim == 1 and self.terminal.shape[0] == self.batched_envs

    def as_input(self):
        rl2 = np.concatenate(
            (self.reward[:, np.newaxis], self.prev_action), axis=-1
        ).astype(np.float32)
        return self.obs, rl2, self.time_idx[:, np.newaxis]


def split_batched_timestep(t: Timestep) -> list[Timestep]:
    batched = t.batched_envs
    obs = utils.unstack_dict(t.obs, axis=0, split=True)
    prev_actions = np.split(t.prev_action, batched, axis=0)
    rewards = np.split(t.reward, batched, axis=0)
    time_idxs = np.split(t.time_idx, batched, axis=0)
    terminals = np.split(t.terminal, batched, axis=0)
    timesteps = []
    for i in range(len(obs)):
        timesteps.append(
            Timestep(
                obs=obs[i],
                prev_action=prev_actions[i],
                reward=rewards[i],
                time_idx=time_idxs[i],
                terminal=terminals[i],
                batched_envs=1,
            )
        )
    return timesteps


@dataclass
class FrozenTraj:
    obs: dict[str, np.ndarray]
    rl2s: np.ndarray
    time_idxs: np.ndarray
    rews: np.ndarray
    dones: np.ndarray
    actions: np.ndarray

    def __post_init__(self):
        assert self.rl2s.ndim == 2
        assert self.time_idxs.ndim == 2
        assert self.rews.ndim == 2
        assert self.dones.ndim == 2
        assert self.actions.ndim == 2

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
        assert timestep.batched_envs == 1
        self.timesteps.append(timestep)

    @property
    def total_return(self):
        return sum([t.reward for t in self.timesteps])

    def __getitem__(self, i):
        return self.timesteps[i]

    def as_input_sequence(self):
        obs = utils.stack_list_array_dicts([t.obs for t in self.timesteps], axis=1)
        actions = np.stack([t.prev_action for t in self.timesteps], axis=1)
        rewards = np.stack([t.reward for t in self.timesteps], axis=1)[..., np.newaxis]
        # match cat order of `Timestep.as_input`
        rl2s = np.concatenate((rewards, actions), axis=-1).astype(np.float32)
        time = np.stack([t.time_idx for t in self.timesteps], axis=1)[..., np.newaxis]
        return obs, rl2s, time

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
        # fmt: off
        obs, rl2s, time = self.as_input_sequence()
        rews = np.stack([t.reward for t in self.timesteps[1:]], axis=1, dtype=np.float32)[..., np.newaxis]
        dones = np.stack([t.terminal for t in self.timesteps[1:]], axis=1, dtype=bool)[..., np.newaxis]
        actions = np.stack([t.prev_action for t in self.timesteps[1:]], axis=1, dtype=np.float32)
        # squeeze batch dimension here...
        batched, length, dim = rl2s.shape
        assert batched == 1, "attempting to freeze a batched trajectory"
        return FrozenTraj(
            obs={k:v.squeeze(0) for k,v in obs.items()},
            rl2s=rl2s.squeeze(0),
            time_idxs=time.squeeze(0),
            rews=rews.squeeze(0),
            dones=dones.squeeze(0),
            actions=actions.squeeze(0),
        )
        # fmt: on


class Relabeler:
    def __call__(self, traj: Trajectory) -> Trajectory:
        return traj
