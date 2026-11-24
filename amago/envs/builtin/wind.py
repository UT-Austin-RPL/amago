"""
Wind: a 2D point-robot meta-RL navigation task with hidden wind perturbations.

Originally adapted from
https://github.com/twni2016/pomdp-baselines/blob/main/envs/meta/toy_navigation/wind.py
and present in early versions of AMAGO before being removed in May 2025.

Restored and updated to match the modern AMAGO meta-RL conventions used by
``HalfCheetahV4_MetaVelocity`` and ``KEpisodeMetaworld``: k-episode
meta-trials with persistent task identity, a soft-reset flag appended to
the observation, and per-trial metrics logged via ``AMAGO_ENV_LOG_PREFIX``.

A "circle goals" extension lets each task additionally vary the goal
location by sampling on the unit circle, with optional quadrant
restriction so train/test splits can be carved out by held-out arc.
"""

import random
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from amago.envs import AMAGO_ENV_LOG_PREFIX


# Math convention: quadrants are 1-indexed counter-clockwise from +x axis.
# Q1: angle ∈ [0,        π/2)    (x>0, y>0)
# Q2: angle ∈ [π/2,      π)      (x<0, y>0)
# Q3: angle ∈ [π,        3π/2)   (x<0, y<0)
# Q4: angle ∈ [3π/2,     2π)     (x>0, y<0)
def _angle_in_quadrant(theta: float, quadrant: int) -> bool:
    q_min = (quadrant - 1) * np.pi / 2
    q_max = quadrant * np.pi / 2
    theta_mod = float(theta) % (2.0 * np.pi)
    return q_min <= theta_mod < q_max


class WindEnv(gym.Env):
    """2D point-robot meta-RL navigation with a hidden wind dynamics term.

    The agent commands 2D position deltas; each step adds a per-task wind
    vector to its motion: ``state += action + wind``. The hidden wind is
    fixed for the duration of a meta-trial, sampled from a discrete set of
    ``n_tasks`` reproducibly-seeded wind vectors.

    By default the goal is fixed at ``(0, 1)``. Setting ``n_goals`` to an
    integer activates "circle goals" mode: the env additionally samples
    ``n_goals`` random goal locations on the unit circle and each task
    becomes a ``(wind_idx, goal_idx)`` pair drawn from the cross product
    of the two sets. The goal is **never** added to the observation -- it
    is part of the hidden task identity that the meta-RL policy must
    infer from the reward signal across the early inner episodes of a
    trial, then exploit on the remaining ones. ``goal_quadrants``
    restricts which quadrants of the circle goals are sampled from, so a
    held-out test split can be built by giving the train env ``[1, 3, 4]``
    and the test env ``[2]`` (or similar). At test time the agent must
    generalize its *exploration strategy* to a previously-blocked region
    of the circle.

    A single hidden wind persists across ``k_episodes`` inner episodes of
    length ``max_episode_steps``. Between inner episodes the agent's
    position is reset to the origin (the wind stays); a soft-reset flag
    (0/1) is appended to the observation so the policy can detect the
    boundary. ``terminated=truncated=True`` only at the end of the full
    meta-trial.

    Observations
    ------------
    A 3-vector ``[x, y, soft_reset_flag]``. Both the wind vector and the
    goal location are hidden task identity that the meta-RL policy must
    infer from the reward signal.

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
        dense_reward: If True, replace the sparse 0/1 goal-disk reward
            with ``exp(-||state - goal||) + 1{success}``. The shaping
            term is bounded in ``(0, 1]`` and gives a non-zero gradient
            everywhere on the plane; the +1 success bonus adds a sharp
            peak at the goal disk so the optimum is unambiguous (~2.0
            in-disk vs ~1.0 at the disk boundary). Useful as a sanity
            check when debugging a stack that struggles with the sparse
            signal. Goal-disk success/membership is still tracked for
            logging (``Trial * Success``) regardless of this setting.
            Default False (preserves the original sparse reward).
        n_goals: If ``None`` (default), the legacy single fixed goal at
            ``(0, 1)`` is used and ``goal_seed`` / ``goal_quadrants`` are
            ignored. If an int, sample that many random goals on the unit
            circle (rejection-sampled from ``goal_quadrants`` if set);
            tasks become ``(wind_idx, goal_idx)`` pairs drawn from the
            cross product of the wind set and goal set.
        goal_seed: Seed for the goal-set RNG. Independent of ``wind_seed``
            so train/test goal splits can be controlled separately.
            Default 4242. Ignored when ``n_goals is None``.
        goal_quadrants: Iterable of allowed quadrants (1-4, math
            convention) when sampling goals on the unit circle. ``None``
            (default) allows all four. To carve out a held-out split:
            train with ``[1, 3, 4]`` (block Q2), test with ``[2]``.
            Ignored when ``n_goals is None``.
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
        n_goals: Optional[int] = None,
        goal_seed: int = 4242,
        goal_quadrants: Optional[List[int]] = None,
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

        # ── Goal set ────────────────────────────────────────────────────────
        # ``n_goals=None`` keeps the original single fixed goal at (0, 1) so
        # existing scripts behave unchanged. Otherwise sample ``n_goals``
        # points on the unit circle, optionally restricted to a list of
        # allowed quadrants (rejection sampling).
        self._n_goals_arg = n_goals
        if n_goals is None:
            self.goals: List[np.ndarray] = [np.array([0.0, 1.0], dtype=np.float32)]
        else:
            n_goals_int = int(n_goals)
            if n_goals_int <= 0:
                raise ValueError(f"n_goals must be a positive int, got {n_goals_int}")
            if goal_quadrants is None:
                allowed: List[int] = [1, 2, 3, 4]
            else:
                allowed = [int(q) for q in goal_quadrants]
                if not allowed:
                    raise ValueError(
                        "goal_quadrants is empty; nothing to sample from."
                    )
                for q in allowed:
                    if q not in (1, 2, 3, 4):
                        raise ValueError(
                            f"goal_quadrants entries must be in {{1,2,3,4}}, got {q}"
                        )
            goal_rng = np.random.RandomState(goal_seed)
            self.goals = []
            # Plenty of headroom for rejection sampling even if only one
            # quadrant is allowed (acceptance rate >= 0.25).
            max_attempts = 100 * n_goals_int + 1000
            attempts = 0
            while len(self.goals) < n_goals_int:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Could not sample {n_goals_int} goals after "
                        f"{attempts} attempts (allowed quadrants: {allowed})."
                    )
                attempts += 1
                theta = goal_rng.uniform(0.0, 2.0 * np.pi)
                if not any(_angle_in_quadrant(theta, q) for q in allowed):
                    continue
                self.goals.append(
                    np.array(
                        [np.cos(theta), np.sin(theta)],
                        dtype=np.float32,
                    )
                )

        self._goal: np.ndarray = self.goals[0].copy()

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        self._wind: np.ndarray = self.winds[0].copy()
        self._task_idx: Tuple[int, int] = (0, 0)
        self._state: np.ndarray = np.zeros(2, dtype=np.float32)
        self.step_count: int = 0
        self.current_trial: int = 0
        self._current_trial_return: float = 0.0
        self._trial_success: bool = False
        # Step (1-indexed) at which is_goal_state() first fired this
        # inner episode, or ``_max_episode_steps`` if it never did.
        # Sentinel = max episode length keeps the metric well-defined
        # for unsuccessful episodes (a "saturated" failure value).
        self._trial_time_to_goal: int = self._max_episode_steps
        self._trial_returns: List[float] = []
        self._total_successes: float = 0.0

    # -- task sampling --------------------------------------------------------

    def get_all_task_idx(self) -> List[Tuple[int, int]]:
        """Cross product of (wind_idx, goal_idx). Returns a list of tuples."""
        return [
            (w, g) for w in range(len(self.winds)) for g in range(len(self.goals))
        ]

    def sample_task_idx(self) -> Tuple[int, int]:
        # Subclasses can override to control train/eval splits. Returns a
        # ``(wind_idx, goal_idx)`` tuple drawn uniformly from the cross
        # product. With the default ``n_goals=None`` the goal index is
        # always 0 so this matches the legacy single-int behavior in
        # spirit.
        return (
            random.randrange(len(self.winds)),
            random.randrange(len(self.goals)),
        )

    def set_task(self, idx) -> None:
        # Accept either a (wind_idx, goal_idx) tuple (new) or a bare int
        # (legacy callers that only knew about winds). In legacy mode the
        # int is the wind index and the goal stays at the only goal we
        # have (the fixed (0,1)).
        if isinstance(idx, (tuple, list)):
            wind_idx, goal_idx = int(idx[0]), int(idx[1])
        else:
            wind_idx, goal_idx = int(idx), 0
        self._wind = self.winds[wind_idx].copy()
        self._goal = self.goals[goal_idx].copy()
        self._task_idx = (wind_idx, goal_idx)

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
        self._trial_time_to_goal = self._max_episode_steps
        return self._get_obs(soft_reset=True), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -0.1, 0.1)
        self._state = (self._state + action + self._wind).astype(np.float32)

        success = self.is_goal_state()
        if self.dense_reward:
            # Smooth distance-shaped reward exp(-||state - goal||) bounded
            # in (0, 1] for non-zero gradient everywhere, plus a hard +1
            # success bonus when inside the goal disk so the optimum is a
            # sharp peak (~2.0 in-disk vs ~1.0 at the disk boundary)
            # rather than a barely-distinguishable plateau. With a start
            # at origin and goal at (0, 1) the initial step's shaping
            # term is exp(-1) ≈ 0.37 and rises monotonically; the bonus
            # then doubles the reward inside the goal disk.
            dist = float(np.linalg.norm(self._state - self._goal))
            reward = float(np.exp(-dist)) + (1.0 if success else 0.0)
        else:
            reward = 1.0 if success else 0.0
        self._current_trial_return += float(reward)
        self.step_count += 1
        # ``step_count`` is 1-indexed after the increment so the first
        # action is step 1. Capture the first success only; later
        # in-disk steps don't move the metric.
        if success and not self._trial_success:
            self._trial_time_to_goal = self.step_count
            self._trial_success = True

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
            # Time-to-goal (in env steps) for the first success in this
            # inner episode. Saturates at ``max_episode_steps`` when the
            # agent never enters the goal disk -- this keeps the metric
            # numerically well-defined and monotone in the "bad"
            # direction (lower = better, max = total failure).
            info[f"{AMAGO_ENV_LOG_PREFIX} Trial {ti} Time to Goal"] = float(
                self._trial_time_to_goal
            )
            self._trial_returns.append(self._current_trial_return)
            self._total_successes += float(self._trial_success)

            # soft reset: position back to origin, wind unchanged
            self._state = np.zeros(2, dtype=np.float32)
            self.step_count = 0
            self._current_trial_return = 0.0
            self._trial_success = False
            self._trial_time_to_goal = self._max_episode_steps
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
