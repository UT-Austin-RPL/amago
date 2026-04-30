"""
Wind: a 2D point-robot meta-RL navigation task with hidden wind perturbations.

Originally adapted from
https://github.com/twni2016/pomdp-baselines/blob/main/envs/meta/toy_navigation/wind.py
and present in early versions of AMAGO before being removed in May 2025.

Restored and updated to match the modern AMAGO meta-RL conventions used by
``HalfCheetahV4_MetaVelocity`` and ``KEpisodeMetaworld``: k-episode
meta-trials with persistent task identity, a soft-reset flag appended to
the observation, and per-trial metrics logged via ``AMAGO_ENV_LOG_PREFIX``.
"""

import random
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from amago.envs import AMAGO_ENV_LOG_PREFIX


class WindEnv(gym.Env):
    """2D point-robot meta-RL navigation with a hidden wind dynamics term.

    The agent commands 2D position deltas; each step adds a per-task wind
    vector to its motion: ``state += action + wind``. The hidden wind is
    fixed for the duration of a meta-trial, sampled from a discrete set of
    ``n_tasks`` reproducibly-seeded wind vectors. The goal is fixed at
    ``(0, 1)`` and the reward is sparse: ``+1`` whenever the agent is
    within ``goal_radius`` of the goal.

    A single hidden wind persists across ``k_episodes`` inner episodes of
    length ``max_episode_steps``. Between inner episodes the agent's
    position is reset to the origin (the wind stays); a soft-reset flag
    (0/1) is appended to the observation so the policy can detect the
    boundary. ``terminated=truncated=True`` only at the end of the full
    meta-trial.

    Observations
    ------------
    A 3-vector ``[x, y, soft_reset_flag]``.

    Args:
        max_episode_steps: Inner-episode horizon. Default 75 (matches the
            original POMDP-baselines setting).
        n_tasks: Number of unique winds in the discrete task set. Default
            60. Override ``sample_task_idx`` in subclasses to carve out
            train / eval splits.
        goal_radius: Sparse-reward arrival radius. Default 0.03.
        k_episodes: Number of inner episodes per meta-trial. Default 1
            recovers the unwrapped single-episode task. Set ``>=2`` for
            true meta-RL trials.
        wind_seed: Seed for the wind set RNG. Default 1337 matches the
            original (a numpy ``RandomState``, not the global numpy seed,
            so we do not leak into other code).
        wind_max: Max absolute value of each wind component. Default 0.08.
        dense_reward: If True, replace the sparse 0/1 goal-disk reward with
            a smooth distance-shaped reward ``exp(-||state - goal||)``.
            Bounded in ``(0, 1]``, peaks at 1 at the goal, gives non-zero
            gradient everywhere on the plane. Useful as a sanity-check
            alternative when debugging a stack that struggles with the
            sparse signal. Goal-disk success/membership is still tracked
            for logging (``Trial * Success``) regardless of this setting.
            Default False (preserves the original sparse reward).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 75,
        n_tasks: int = 60,
        goal_radius: float = 0.03,
        k_episodes: int = 1,
        wind_seed: int = 1337,
        wind_max: float = 0.08,
        dense_reward: bool = False,
    ):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self._max_episode_steps = int(max_episode_steps)
        self.k_episodes = int(k_episodes)
        self.goal_radius = float(goal_radius)
        self.dense_reward = bool(dense_reward)

        # Local RandomState so the deterministic task set does not affect
        # the global numpy RNG (the original called np.random.seed(1337)).
        rng = np.random.RandomState(wind_seed)
        self.winds: List[np.ndarray] = [
            np.array(
                [rng.uniform(-wind_max, wind_max), rng.uniform(-wind_max, wind_max)],
                dtype=np.float64,
            )
            for _ in range(self.n_tasks)
        ]
        self._goal = np.array([0.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        self._wind: np.ndarray = self.winds[0].copy()
        self._task_idx: int = 0
        self._state: np.ndarray = np.zeros(2, dtype=np.float32)
        self.step_count: int = 0
        self.current_trial: int = 0
        self._current_trial_return: float = 0.0
        self._trial_success: bool = False
        self._trial_returns: List[float] = []
        self._total_successes: float = 0.0

    # -- task sampling --------------------------------------------------------

    def get_all_task_idx(self):
        return range(len(self.winds))

    def sample_task_idx(self) -> int:
        # override in subclasses to control train/eval splits.
        return random.choice(list(self.get_all_task_idx()))

    def set_task(self, idx: int) -> None:
        self._wind = self.winds[idx].copy()
        self._task_idx = int(idx)

    # -- gymnasium API --------------------------------------------------------

    def _get_obs(self, soft_reset: bool) -> np.ndarray:
        return np.array(
            [self._state[0], self._state[1], float(soft_reset)],
            dtype=np.float32,
        )

    def is_goal_state(self) -> bool:
        return bool(np.linalg.norm(self._state - self._goal) <= self.goal_radius)

    def reset(
        self,
        *,
        new_task: bool = True,
        seed: Optional[int] = None,
        options=None,
    ):
        super().reset(seed=seed)
        if new_task:
            self.set_task(self.sample_task_idx())
            self.current_trial = 0
            self._trial_returns = []
            self._total_successes = 0.0
        self._state = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self._current_trial_return = 0.0
        self._trial_success = False
        return self._get_obs(soft_reset=True), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -0.1, 0.1)
        self._state = (self._state + action + self._wind).astype(np.float32)

        success = self.is_goal_state()
        if self.dense_reward:
            # Smooth distance-shaped reward: bounded in (0, 1], peaks at 1
            # at the goal, decays as exp(-||state - goal||). Gives non-zero
            # gradient everywhere; with a starting position at origin and
            # goal at (0, 1) the initial step's reward is ~exp(-1) ≈ 0.37
            # and rises monotonically as the agent approaches.
            dist = float(np.linalg.norm(self._state - self._goal))
            reward = float(np.exp(-dist))
        else:
            reward = 1.0 if success else 0.0
        if success:
            self._trial_success = True

        self._current_trial_return += float(reward)
        self.step_count += 1

        terminated = False
        truncated = False
        soft_reset = False
        info: dict = {}

        if self.step_count >= self._max_episode_steps:
            ti = self.current_trial
            info[f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} Return"] = (
                self._current_trial_return
            )
            info[f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} Success"] = float(
                self._trial_success
            )
            self._trial_returns.append(self._current_trial_return)
            self._total_successes += float(self._trial_success)

            # soft reset: position back to origin, wind unchanged
            self._state = np.zeros(2, dtype=np.float32)
            self.step_count = 0
            self._current_trial_return = 0.0
            self._trial_success = False
            soft_reset = True
            self.current_trial += 1

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
                info[f"{AMAGO_ENV_LOG_PREFIX} Total Successes"] = self._total_successes

        return self._get_obs(soft_reset), reward, terminated, truncated, info

    def render(self):
        # The original had a matplotlib plot helper; intentionally omitted
        # here to keep the env dependency-free. See git history before
        # commit 8321e61 for the original plotting code if needed.
        pass
