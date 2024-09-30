import random
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import gin


class ExplorationWrapper(ABC, gym.ActionWrapper):
    def __init__(self, amago_env):
        super().__init__(amago_env)
        self.batched_envs = amago_env.batched_envs

    @abstractmethod
    def add_exploration_noise(self, action: np.ndarray, local_step: int):
        raise NotImplementedError

    def action(self, a: np.ndarray):
        if self.batched_envs == 1:
            a = a[np.newaxis, :]
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
    """
    Implements the bi-level epsilon greedy exploration strategy visualized in Figure 13
    of the AMAGO paper. Exploration noise decays both over the course of training and
    throughout each rollout. This more closely resembles the online exploration/exploitation
    strategy of a meta-RL agent in a new environment.
    """

    def __init__(
        self,
        amago_env,
        rollout_horizon: int,
        eps_start_start: float = 1.0,  # start of training, start of rollout
        eps_start_end: float = 0.05,  # end of training, start of rollout
        eps_end_start: float = 0.8,  # start of training, end of rollout
        eps_end_end: float = 0.01,  # end of training, end of rollout
        steps_anneal: int = 1_000_000,
    ):
        super().__init__(amago_env)
        self.eps_start_start = eps_start_start
        self.eps_start_end = eps_start_end
        self.eps_end_start = eps_end_start
        self.eps_end_end = eps_end_end
        self.rollout_horizon = rollout_horizon

        self.start_global_slope = (eps_start_start - eps_start_end) / steps_anneal
        self.end_global_slope = (eps_end_start - eps_end_end) / steps_anneal
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.global_step = 0
        self.global_multiplier = 1.0

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        np.random.seed(random.randint(0, 1e6))
        self.global_multiplier = np.random.rand(self.batched_envs)
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

    def add_exploration_noise(self, action: np.ndarray, local_step: np.ndarray):
        assert action.shape[0] == self.batched_envs
        assert local_step.shape[0] == self.batched_envs

        noise = self.current_eps(local_step)
        if self.discrete:
            # epsilon greedy (DQN-style)
            num_actions = self.env.action_space.n
            random_action = np.random.randint(
                0, num_actions, size=(self.batched_envs, 1)
            )
            use_random = np.random.rand(self.batched_envs) <= noise
            expl_action = (
                use_random * random_action + (1 - use_random) * action
            ).astype(np.uint8)
            assert expl_action.shape[1] == 1
        else:
            # random noise (TD3-style)
            expl_action = action + noise * np.random.randn(*action.shape)
            expl_action = np.clip(expl_action, -1.0, 1.0).astype(np.float32)
            assert expl_action.dtype == np.float32
            assert expl_action.shape[1] == action.shape[1]
        self.global_step += 1

        assert expl_action.shape[0] == self.batched_envs
        return expl_action


@gin.configurable
class EpsilonGreedy(BilevelEpsilonGreedy):
    """
    Sets the parameters of the BilevelEpsilonGreedy wrapper to be equivalent to standard epsilon-greedy.
    """

    def __init__(
        self,
        amago_env,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        steps_anneal: int = 1_000_000,
    ):
        super().__init__(
            amago_env,
            rollout_horizon=float("inf"),
            eps_start_start=eps_start,
            eps_start_end=eps_end,
            eps_end_start=eps_start,
            eps_end_end=eps_end,
            steps_anneal=steps_anneal,
        )
