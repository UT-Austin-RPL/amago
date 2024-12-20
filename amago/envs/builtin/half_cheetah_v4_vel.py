import random

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
        truncated = truncated or self.step_count >= 1000
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
        **kwargs,
    ):
        super().__init__(
            forward_reward_weight=velocity_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            **kwargs,
        )
        self.task_min_velocity = task_min_velocity
        self.task_max_velocity = task_max_velocity
        self.reset()

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

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.target_velocity = self.sample_target_velocity()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            achieved = info[f"{AMAGO_ENV_LOG_PREFIX} Avg. Last 50 Timestep Velocity"]
            info[
                f"{AMAGO_ENV_LOG_PREFIX} Target Velocity Error at Final 50 Timesteps"
            ] = abs(self.target_velocity - achieved)
        return obs, reward, terminated, truncated, info
