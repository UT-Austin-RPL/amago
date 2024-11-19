import gymnax
from gymnax.wrappers import GymnaxToVectorGymWrapper
import jax.numpy as jnp
import numpy as np


class GymnaxCompatibility(GymnaxToVectorGymWrapper):
    """
    Convert gymnax Gym wrapper to the expected AMAGO interface.

        - Gymnax wants to give us the batched observation and action spaces,
          but AMAGO is expecting unbatched spaces.
        - It's also going to send out jax arrays, but we need numpy.

    A key point is that this only works because gymnax envs automatically reset.
    The "already_vectorized" mode in AMAGO relies on auto-resets because we cannot
    reset specific indices of the vectorized enviornment from the highest wrapper level.
    """

    @property
    def observation_space(self):
        return self.single_observation_space

    @property
    def action_space(self):
        return self.single_action_space

    def reset(self, *args, **kwargs):
        obs, info = super().reset()
        return np.array(obs), info

    def step(self, action):
        *outs, info = super().step(jnp.array(action))
        obs, rewards, te, tr = (np.array(o) for o in outs)
        return obs, rewards, te, tr, info
