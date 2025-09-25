"""
Exploration via gymnasium wrappers.
"""

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import gin


class ExplorationWrapper(ABC, gym.ActionWrapper):
    """Abstract base class for exploration wrappers.

    Args:
        amago_env: The environment to wrap.
    """

    def __init__(self, amago_env):
        super().__init__(amago_env)
        self.batched_envs = amago_env.batched_envs

    @abstractmethod
    def add_exploration_noise(self, action: np.ndarray, local_step: int) -> np.ndarray:
        """Add exploration noise to the action.

        Args:
            action: The action provided by the Agent.
            local_step: The number of timesteps since the last episode reset.

        Returns:
            The action to be taken in the environment.
        """
        raise NotImplementedError

    def action(self, a: np.ndarray) -> np.ndarray:
        if self.batched_envs == 1:
            a = a[np.newaxis, :]
        if not self.env.discrete:
            assert (
                abs(a).max() <= 1.0
            ), "Continuous action out of [-1, 1] bound (detected before exploration noise clipping would have covered up this problem!)"
        action = self.add_exploration_noise(a, self.env.step_count)
        if self.batched_envs == 1:
            action = np.squeeze(action, axis=0)
        return action

    @property
    def return_history(self):
        return self.env.return_history

    @property
    def success_history(self):
        return self.env.success_history


@gin.configurable
class BilevelEpsilonGreedy(ExplorationWrapper):
    """Implements the bi-level epsilon greedy exploration strategy visualized in Figure 13 of
    the AMAGO paper.

    Exploration noise decays both over the course of training and throughout each rollout.
    This more closely resembles the online exploration/exploitation strategy of a meta-RL
    agent in a new environment.

    Args:
        amago_env: The environment to wrap.

    Keyword Args:
        rollout_horizon: The expected maximum length of each rollout. `gin.REQUIRED`, meaning it must be
            configured manually via gin on a case-by-case basis, e.g.
            gin.bind_parameter("BilevelEpsilonGreedy.rollout_horizon", 100)
        eps_start_start: Exploration noise at the start of training and start of a rollout.
            Default is 1.0.
        eps_start_end: Exploration noise at the end of training and start of a rollout.
            Default is 0.05.
        eps_end_start: Exploration noise at the start of training and end of a rollout.
            Default is 0.8.
        eps_end_end: Exploration noise at the end of training and end of a rollout.
            Default is 0.01.
        steps_anneal: Linear schedule end point (in terms of steps taken in each actor
            process). Default is 1,000,000.
        randomize_eps: Treat the schedule as the max and sample uniform [0, max) for each
            actor. This tends to reduce the need to tune steps_anneal. Default is True.
    """

    def __init__(
        self,
        amago_env,
        rollout_horizon: int = gin.REQUIRED,
        eps_start_start: float = 1.0,
        eps_start_end: float = 0.05,
        eps_end_start: float = 0.8,
        eps_end_end: float = 0.01,
        steps_anneal: int = 1_000_000,
        randomize_eps: bool = True,
    ):
        super().__init__(amago_env)
        self.eps_start_start = eps_start_start
        self.eps_start_end = eps_start_end
        self.eps_end_start = eps_end_start
        self.eps_end_end = eps_end_end
        self.rollout_horizon = rollout_horizon
        self.randomize_eps = randomize_eps

        self.start_global_slope = (eps_start_start - eps_start_end) / steps_anneal
        self.end_global_slope = (eps_end_start - eps_end_end) / steps_anneal
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.multidiscrete = isinstance(self.env.action_space, gym.spaces.MultiDiscrete)
        assert not isinstance(
            self.env.action_space, gym.spaces.MultiBinary
        ), "Use a custom wrapper for MultiBinary actions"
        self.global_step = 0
        self.global_multiplier = 1.0

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        self.rng = np.random.default_rng()
        if self.randomize_eps:
            self.global_multiplier = self.rng.random(self.batched_envs)
        return out

    def current_eps(self, local_step: np.ndarray):
        ep_start = max(
            self.eps_start_start - self.start_global_slope * self.global_step,
            self.eps_start_end,
        )
        ep_end = max(
            self.eps_end_start - self.end_global_slope * self.global_step,
            self.eps_end_end,
        )
        local_progress = local_step / self.rollout_horizon
        current = self.global_multiplier * (
            ep_start - ((ep_start - ep_end) * local_progress)
        )
        return current

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        done = np.logical_or(terminated, truncated)
        if done.any():
            # handle auto-resets by resetting the global multiplier
            new_global_multiplier = self.rng.random(self.batched_envs)
            self.global_multiplier[done] = new_global_multiplier[done]
        return obs, rew, terminated, truncated, info

    def add_exploration_noise(self, action: np.ndarray, local_step: np.ndarray):
        assert action.shape[0] == self.batched_envs
        assert local_step.shape[0] == self.batched_envs

        noise = self.current_eps(local_step)
        if self.discrete:
            # epsilon greedy (DQN-style)
            num_actions = self.env.action_space.n
            random_action = self.rng.integers(
                0, num_actions, size=(self.batched_envs, 1)
            )
            use_random = np.expand_dims(self.rng.random(self.batched_envs) <= noise, 1)
            expl_action = (
                use_random * random_action + (1 - use_random) * action
            ).astype(np.uint8)
            assert expl_action.shape == (self.batched_envs, 1)
        elif self.multidiscrete:
            nvec = self.env.action_space.nvec
            random_actions = []
            for i, n in enumerate(nvec):
                random_sub = self.rng.integers(0, n, size=(self.batched_envs,))
                use_random = self.rng.random(self.batched_envs) <= noise.flatten()
                sub_action = np.where(use_random, random_sub, action[:, i])
                random_actions.append(sub_action)
            expl_action = np.stack(random_actions, axis=1).astype(np.int32)
            assert expl_action.shape == (self.batched_envs, len(nvec))
        else:
            # random noise (TD3-style)
            expl_action = action + noise[:, np.newaxis] * self.rng.standard_normal(
                action.shape
            )
            expl_action = np.clip(expl_action, -1.0, 1.0).astype(np.float32)
            assert expl_action.dtype == np.float32
            assert expl_action.shape == (self.batched_envs, action.shape[1])
        self.global_step += 1
        return expl_action


@gin.configurable
class EpsilonGreedy(BilevelEpsilonGreedy):
    """Standard exploration noise schedule.

    Classic epsilon-greedy exploration in discrete action spaces. TD3-style
    (action + epsilon * noise).clip(-1, 1) in continuous action spaces.

    Args:
        amago_env: The environment to wrap.

    Keyword Args:
        eps_start: Exploration noise at the start of training. Default is 1.0.
        eps_end: Exploration noise at the end of training. Default is 0.05.
        steps_anneal: Linear schedule end point (in terms of steps taken *in a single actor
            process*). Default is 1,000,000.
        randomize_eps: Treat the schedule as the max and sample uniform [0, max) for each
            parallel actor. This tends to reduce the need to tune steps_anneal. Default is
            True.
    """

    def __init__(
        self,
        amago_env,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        steps_anneal: int = 1_000_000,
        randomize_eps: bool = True,
    ):
        super().__init__(
            amago_env,
            rollout_horizon=float("inf"),
            eps_start_start=eps_start,
            eps_start_end=eps_end,
            eps_end_start=eps_start,
            eps_end_end=eps_end,
            steps_anneal=steps_anneal,
            randomize_eps=randomize_eps,
        )
