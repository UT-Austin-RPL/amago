"""
Trajectory datastructures
"""

import pickle
import copy
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Optional, Iterable

import numpy as np

from amago import utils


@dataclass
class Timestep:
    """Stores a single timestep of rollout data.

    Time-aligned to the input format of the policy. Agents learn from sequences of timesteps.
    Each timestep contains the current observation and time_idx as well as everything that
    has happened since the last observation was revealed (previous action, reward, terminal).

    Args:
        obs: Dictionary of current observation keys and values.
        prev_action: The *previous* action taken by the agent.
        reward: The reward received by the agent after it took prev_action.
        time_idx: The integer index of the current timestep.
        terminal: The terminal signal of the environment. True if this is the final observation.
        batched_envs: The number of environments in the batch. Used to disambiguate batch dimension.
    """

    obs: dict[str, np.ndarray]
    prev_action: np.ndarray
    reward: np.ndarray
    time_idx: np.ndarray
    terminal: np.ndarray
    batched_envs: int

    def __post_init__(self):
        assert (
            self.prev_action.ndim == 2
            and self.prev_action.shape[0] == self.batched_envs
        )
        assert self.reward.ndim == 1 and self.reward.shape[0] == self.batched_envs
        assert self.time_idx.ndim == 1 and self.time_idx.shape[0] == self.batched_envs
        assert self.terminal.ndim == 1 and self.terminal.shape[0] == self.batched_envs

    def as_input(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Outputs Timestep data in the input format of the Agent.

        Returns:
            tuple: A tuple containing:
                - obs: Dictionary of observations with shape (batched_envs, dim_value)
                - rl2s: Tensor of meta-RL inputs (prev action, reward) with shape
                  (batched_envs, 1 + D_action)
                - time_idx: Tensor of time indices with shape (batched_envs, 1)
        """
        rl2 = np.concatenate(
            (self.reward[:, np.newaxis], self.prev_action), axis=-1
        ).astype(np.float32)
        return self.obs, rl2, self.time_idx[:, np.newaxis]

    def create_reset_version(self, reset_idxs: np.ndarray) -> "Timestep":
        """Manually assign indices of a batched timestep to be first in a new trajectory.

        Creates a new Timestep object with rewards, time_idxs, and terminal signals reset as if
        the environment was reset at the given reset_indices. Used for handling auto-resets in
        vectorized environments.

        Args:
            reset_idxs: Tensor of indices of parallel environments being reset.

        Returns:
            Timestep: New Timestep object with reset values for specified environments.
        """
        # this must match AMAGOEnv/SequenceWrapper.reset
        assert reset_idxs.shape[0] == self.batched_envs
        new = copy.deepcopy(self)
        new.reward[reset_idxs] = 0
        new.time_idx[reset_idxs] = 0
        new.terminal[reset_idxs] = False
        new.prev_action[reset_idxs] = 0
        return new


def split_batched_timestep(t: Timestep) -> list[Timestep]:
    """Split a batched timestep into a list of unbatched timesteps.

    Used to break up vectorized environments into independent trajectories.

    Args:
        t: Batched timestep to split. Batch dim is t.batched_envs.

    Returns:
        List of timesteps with length equal to the number of environments in the batch.
    """
    batched = t.batched_envs
    if batched == 1:
        return [t]
    obs = utils.split_dict(t.obs, axis=0)
    prev_actions = utils.split_batch(t.prev_action, axis=0)
    rewards = utils.split_batch(t.reward, axis=0)
    time_idxs = utils.split_batch(t.time_idx, axis=0)
    terminals = utils.split_batch(t.terminal, axis=0)
    timesteps = [
        Timestep(
            obs=obs[i],
            prev_action=prev_actions[i],
            reward=rewards[i],
            time_idx=time_idxs[i],
            terminal=terminals[i],
            batched_envs=1,
        )
        for i in range(batched)
    ]
    return timesteps


@dataclass
class FrozenTraj:
    """A finished trajectory that is ready to be used as training data.

    Args:
        obs: Dictionary of observations with shape (Batch, Length, dim_value)
        rl2s: Tensor of meta-RL inputs (prev action, reward) with shape (Batch, Length, 1 + D_action)
        time_idxs: Tensor of time indices with shape (Batch, Length, 1)
        rews: Tensor of rewards with shape (Batch, Length, 1)
        dones: Tensor of terminal signals with shape (Batch, Length, 1)
        actions: Tensor of actions with shape (Batch, Length, D_action)
    """

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
        """Get all trajectory data as an easily serializable dictionary.

        Returns:
            dict: Flat dictionary with observation data prefixed with _OBS_KEY_ for
                format restoration.
        """
        d = asdict(self)
        for obs_k, obs_v in d["obs"].items():
            # flatten obs dict but mark w/ special prefix
            d[f"_OBS_KEY_{obs_k}"] = obs_v
        del d["obs"]
        return d

    @classmethod
    def from_dict(cls, d: dict[np.ndarray]) -> "FrozenTraj":
        """Fold flattened observation dictionary back to original keys.

        Args:
            d: Flat dictionary with observation data prefixed with _OBS_KEY_.
                Typically output of FrozenTraj.to_dict.

        Returns:
            FrozenTraj: Object with original observation keys restored.
        """
        args = {"obs": {}}
        for k, v in d.items():
            if k.startswith("_OBS_KEY_"):
                # fold the flattened obs dict back to original keys
                args["obs"][k.replace("_OBS_KEY_", "", 1)] = v
            else:
                args[k] = v
        return cls(**args)


class Trajectory:
    """A sequence of timesteps.

    Stores a rollout and handles disk saves when using the default :py:class:`~amago.loading.RLDataset`.

    Args:
        timesteps: Iterable of :py:class:`Timestep` objects.
    """

    def __init__(self, timesteps=Optional[Iterable[Timestep]]):
        self.timesteps = timesteps or []

    def add_timestep(self, timestep: Timestep) -> None:
        """Add a timestep to the trajectory.

        Args:
            timestep: Timestep object to add.
        """
        assert isinstance(timestep, Timestep)
        assert timestep.batched_envs == 1
        self.timesteps.append(timestep)

    @property
    def total_return(self) -> float:
        """Calculate the total return of this trajectory.

        Returns:
            float: Sum of all rewards in the trajectory.
        """
        return sum([t.reward for t in self.timesteps])

    def __getitem__(self, i: int) -> Timestep:
        return self.timesteps[i]

    def as_input_sequence(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Returns a sequence of observations, rl2s, and time_idxs.

        Uses the trajectory data to gather the standard input sequences for the Agent.

        Returns:
            tuple: A tuple containing:
                - obs: Dictionary of observations with shape (Batch, Length, dim_value)
                - rl2s: Tensor of meta-RL inputs with shape (Batch, Length, 1 + D_action)
                - time: Tensor of time indices with shape (Batch, Length, 1)
        """
        obs = utils.stack_list_array_dicts([t.obs for t in self.timesteps], axis=1)
        actions = np.stack([t.prev_action for t in self.timesteps], axis=1)
        rewards = np.stack([t.reward for t in self.timesteps], axis=1)[..., np.newaxis]
        # match cat order of `Timestep.as_input`
        rl2s = np.concatenate((rewards, actions), axis=-1).astype(np.float32)
        time = np.stack([t.time_idx for t in self.timesteps], axis=1)[..., np.newaxis]
        return obs, rl2s, time

    def __len__(self) -> int:
        """The number of timesteps in this trajectory.

        Returns:
            int: The number of timesteps in this trajectory.
        """
        return len(self.timesteps)

    def save_to_disk(self, path: str, save_as: str) -> None:
        """Save the trajectory to disk.

        Args:
            path: Path to save the trajectory to.
            save_as: Format to save the trajectory in:
                - 'trajectory': Pickle file storing entire object (now rarely used)
                - 'npz': Standard numpy .npz file format
                - 'npz-compressed': Compressed numpy .npz file format
        """
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
        """Freeze the trajectory and return a :py:class:`FrozenTraj` object.

        :py:class:`FrozenTraj` saves time by precomputing input sequences for the Agent.

        Returns:
            FrozenTraj: Frozen trajectory object with precomputed sequences.
        """
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


class Relabeler(ABC):
    """A hook for modifying trajectory data during training.

    In the default :py:class:`~amago.loading.DiskTrajDataset`, Relabeler has the chance to edit input trajectories
    before they are passed to an agent for training. Enables Hindsight Experience Replay
    (HER) and variants. See examples/13_mazerunner_relabeling.py for an implementation.
    """

    def __call__(self, traj: Trajectory | FrozenTraj) -> FrozenTraj:
        return self.relabel(traj)

    @abstractmethod
    def relabel(self, traj: Trajectory | FrozenTraj) -> FrozenTraj:
        """Relabel a trajectory.

        Args:
            traj: Trajectory or FrozenTraj object to relabel. Can be modified in place.

        Returns:
            FrozenTraj: New FrozenTraj object with relabeled data.
        """
        raise NotImplementedError


class NoOpRelabeler(Relabeler):
    """A no-op relabeler that returns the input trajectory unchanged."""

    def relabel(self, traj: Trajectory | FrozenTraj) -> FrozenTraj:
        return traj
