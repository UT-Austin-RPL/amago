"""
HalfCheetah-Velocity implemented with HalfCheetah-v4.
"""

import random
from typing import List

import gymnasium as gym
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import numpy as np

from amago.envs import AMAGO_ENV_LOG_PREFIX


class _HalfCheetahV4ExposeVelReward(HalfCheetahEnv):
    def velocity_reward_term(self, x_velocity):
        return self._forward_reward_weight * x_velocity

    def step(self, action):
        # https://github.com/Farama-Foundation/Gymnasium/blob/81b87efb9f011e975f3b646bab6b7871c522e15e/gymnasium/envs/mujoco/half_cheetah_v4.py#L195
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        ctrl_cost = self.control_cost(action)
        forward_reward = self.velocity_reward_term(x_velocity)
        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


class HalfCheetahV4LogVelocity(_HalfCheetahV4ExposeVelReward):
    """A wrapper around HalfCheetah-V4 that will automatically log velocity metrics when used in AMAGO.

    Args:
        max_episode_steps: Step horizon at which the inner episode is
            truncated. Default 1000 matches the original gym registration.
            Lower values are useful for k-episode meta-trial wrappers that
            want to fit multiple inner episodes inside a fixed sequence
            length.
    """

    def __init__(self, max_episode_steps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_episode_steps = int(max_episode_steps)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._velocity_history = []
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # do the step count here without a wrapper, since we're skipping the
        # gym registration that would normally handle it.
        self.step_count += 1
        self._velocity_history.append(info["x_velocity"])
        truncated = truncated or self.step_count >= self.max_episode_steps
        if terminated or truncated:
            v = np.array(self._velocity_history)
            info[f"{AMAGO_ENV_LOG_PREFIX} Mean Velocity"] = v.mean().item()
            info[f"{AMAGO_ENV_LOG_PREFIX} Max Velocity"] = v.max().item()
            info[f"{AMAGO_ENV_LOG_PREFIX} Avg. Last 50 Timestep Velocity"] = (
                v[-50:].mean().item()
            )
        return obs, reward, terminated, truncated, info


class HalfCheetahV4_MetaVelocity(HalfCheetahV4LogVelocity):
    """
    A "remaster" of the classic HalfCheetahVelocity meta-RL task for modern gymnasium/mujoco.

    Reward terms are based on the version featured in the VariBAD codebase
    (https://github.com/lmzintgraf/varibad/blob/57e1795be142ace52d0c353097acf193d9067200/environments/mujoco/half_cheetah_vel.py#L8)

    A single hidden ``target_velocity`` persists across ``k_episodes`` inner
    episodes (each truncated by the parent class at ``max_episode_steps``).
    Between inner episodes the simulator is soft-reset; the hidden task is
    only resampled when ``reset(new_task=True)`` is called from outside (the
    default at the start of every meta-trial). A soft-reset flag (0/1) is
    appended to the obs so the agent can detect inner-episode boundaries
    without losing task identity. Per-episode metrics from the parent class
    (Mean / Max / Avg. Last 50 Timestep Velocity) are promoted to
    ``Trial {i} ...`` keys, matching the meta-RL logging convention used by
    ``KEpisodeMetaworld``.

    Args:
        ctrl_cost_weight: Defaults to half the normal ctrl cost, as in the original HalfCheetahVelocity task.
        velocity_reward_weight: Defaults to 1.0.

        .. note::
            The original HalfCheetahVelocity task has a max velocity of 3.
            For reference: a reasonably good policy optimizing HalfCheetah-v4
            with the standard reward (go as fast as possible) would reach a vel > 10.

        task_min_velocity: Minimum target velocity that determines the hidden task. Defaults to 0.0.
        task_max_velocity: Maximum target velocity that determines the hidden task. Defaults to 3.0.
        k_episodes: Number of inner episodes per meta-trial. Default 1
            recovers the unwrapped single-episode task. Set ``>=2`` for
            true meta-RL trials (e.g. 3 inner episodes of 200 steps with
            ``max_episode_steps=200``).
    """

    def __init__(
        self,
        # defaults to half the normal ctrl cost, as in the original HalfCheetahVelocity
        ctrl_cost_weight: float = 0.5 * 0.1,
        velocity_reward_weight: float = 1.0,
        # the original HalfCheetahVelocity task has a max velocity of 3.
        # for reference: a reasonably good policy optimizing HalfCheetah-v4
        # with the standard reward (go as fast as possible) would reach a vel > 10.
        # We can use the HalfCheetahV4LogVelocity env above to verify this.
        task_min_velocity: float = 0.0,
        task_max_velocity: float = 3.0,
        k_episodes: int = 1,
        **kwargs,
    ):
        super().__init__(
            forward_reward_weight=velocity_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            **kwargs,
        )
        self.task_min_velocity = task_min_velocity
        self.task_max_velocity = task_max_velocity
        self.k_episodes = int(k_episodes)

        # Append a soft-reset flag (0/1) to the parent observation space.
        flat = self.observation_space
        low = np.append(flat.low.astype(np.float32), np.float32(0.0))
        high = np.append(flat.high.astype(np.float32), np.float32(1.0))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.target_velocity: float = 0.0
        self.current_trial: int = 0
        self._trial_returns: List[float] = []
        self._current_trial_return: float = 0.0
        self.reset()

    @staticmethod
    def _augment(obs: np.ndarray, soft_reset: bool) -> np.ndarray:
        return np.append(obs, float(soft_reset)).astype(np.float32)

    def velocity_reward_term(self, x_velocity):
        return (
            self._forward_reward_weight
            * -np.abs(x_velocity - self.target_velocity).item()
        )

    def sample_target_velocity(self):
        # inherit & override to change the meta-task distribution.
        # note that you can pass different training / testing envs to amago.Experiment.
        # be sure to use `random` or be careful about np default_rng to ensure
        # tasks are different across async parallel actors!
        return random.uniform(self.task_min_velocity, self.task_max_velocity)

    def reset(self, *, new_task: bool = True, seed=None, options=None):
        # ``new_task=False`` is reserved for soft resets *inside* step();
        # external callers should use the default to start a fresh meta-trial.
        obs, info = super().reset(seed=seed, options=options)
        if new_task:
            self.target_velocity = self.sample_target_velocity()
            self.current_trial = 0
            self._trial_returns = []
        self._current_trial_return = 0.0
        return self._augment(obs, soft_reset=True), info

    def _log_per_trial_metrics(self, info: dict) -> None:
        ti = self.current_trial
        for suffix in (
            "Mean Velocity",
            "Max Velocity",
            "Avg. Last 50 Timestep Velocity",
        ):
            src = f"{AMAGO_ENV_LOG_PREFIX} {suffix}"
            if src in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} {suffix}"] = info.pop(src)
        achieved_key = (
            f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} Avg. Last 50 Timestep Velocity"
        )
        if achieved_key in info:
            info[
                f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} Target Velocity Error at Final 50 Timesteps"
            ] = abs(self.target_velocity - info[achieved_key])
        info[f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} Return"] = self._current_trial_return

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._current_trial_return += float(reward)
        soft_reset = False

        if terminated or truncated:
            self._log_per_trial_metrics(info)
            self._trial_returns.append(self._current_trial_return)
            self.current_trial += 1

            # Always soft-reset the simulator on inner-episode end (mirrors
            # KEpisodeMetaworld). The augmented obs's soft-reset flag is set
            # to 1 so the agent can detect the boundary.
            obs, _ = super().reset()
            soft_reset = True
            self._current_trial_return = 0.0

            if self.current_trial >= self.k_episodes:
                # Meta-trial end is a time-limit truncation, not a terminal
                # state: the underlying MDP could continue if we kept
                # observing. Leave terminated=False so the critic
                # bootstraps as usual.
                terminated = False
                truncated = True
                info[f"{AMAGO_ENV_LOG_PREFIX} Total Return"] = float(
                    np.sum(self._trial_returns)
                )
            else:
                terminated = truncated = False

        return self._augment(obs, soft_reset), reward, terminated, truncated, info
