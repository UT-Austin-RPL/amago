Setup the Environment
======================
|
|

AMAGO follows the ``gymnasium`` (0.26 < version < 0.30) environment API.
A typical ``gymnasium.Env`` simulates a single instance of an environment. 
We'll be collecting data in parallel by creating multiple independent instances. 
All we need to do is define a function that creates an :py:class:`~amago.envs.amago_env.AMAGOEnv`, for example:

.. code-block:: python

    import gymnasium
    import amago

    def make_env():
        env = gymnasium.make("Pendulum-v1")
        # `env_name` is used for logging eval metrics. multi-task envs
        # will sample a new task between resets and change the name accordingly.
        env = amago.envs.AMAGOEnv(env=env, env_name="Pendulum", batched_envs=1)
        return env

    sample_env = make_env()
    # If the obs space is not a dict, AMAGOEnv will create a default key of 'observation':
    sample_env.observation_spce
    # >>> Dict('observation': Box([-1. -1. -8.], [1. 1. 8.], (3,), float32))
    # environments return an `amago.hindsight.Timestep`
    sample_timestep, info = sample_env.reset()
    # each environment has a batch dimension of 1
    sample_timestep.obs["observation"].shape
    # >>> (1, 3)

    experiment = amago.Experiment(
        make_train_env=make_env,
        make_val_env=make_env,
        parallel_actors=36,
        env_mode="async", # or "sync" for easy debugging / reduced overhead
        ..., # we'll be walking through more arguments in the following sections
    )


.. note::
    We follow infinite-bootstrapping convention where environments are reset on ``done = terminated or truncated`` but RL training only uses ``done = terminated`` for value learning.

|

Vectorized Envs and ``jax``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some domains already parallelize computation over many environments: 
``step`` expects a batch of actions and returns a batch of observations. 
Examples include recent envs like `gymnax <https://github.com/RobertTLange/gymnax>`_ 
that use `jax <https://jax.readthedocs.io/en/latest/>`_ and a GPU to boost their framerate:

.. code-block:: python

    import gymnax
    from amago.envs.builtin.gymnax_envs import GymnaxCompatability

    def make_env():
        env, params = gymnax.make("Pendulum-v1")
        # AMAGO expects numpy data and an unbatched observation space
        vec_env = GymnaxCompatability(env, num_envs=512, params=params)
        # vec_env.reset()[0].shape >>> (512, 3) # already vectorized!
        return AMAGOEnv(env=vec_env, env_name="gymnax_Pendulum", batched_envs=512)

    experiment = amago.Experiment(
        make_train_env=make_env,
        make_val_env=make_env,
        parallel_actors=512, # match batch dim of environment
        env_mode="already_vectorized", # prevents spawning multiple async instances
        ...,
    )

There are some details in getting the pytorch agent and jax envs to cooperate and share a GPU. See :doc:`/examples/02_gymnax`.

|

Meta-RL and Auto-Resets
~~~~~~~~~~~~~~~~~~~~~~~~

Most meta-RL problems involve an environment that resets itself to the same task ``k`` times. 
There is no consistent way to handle this across different benchmarks. 
Therefore, **AMAGO expects the environment to be handling multi-trial resets on its own.** 
``terminated`` and ``truncated`` indicate that this environment interaction is finished and should be saved/logged. For example:

.. code-block:: python

    from amago.envs import AMAGO_ENV_LOG_PREFIX

    class MyMetaRLEnv(gym.Wrapper):
        
        def reset(self):
            self.sample_new_task_somehow()
            obs, info = self.env.reset()
            self.current_episode = 0
            self.episode_return = 0
            return obs, info
        
        def step(self, action):
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.episode_return += reward
            if terminated or truncated:
                # "trial-done"
                next_obs, info = self.reset_to_the_same_task_somehow()
                # we'll log anything in `info` that begins with `AMAGO_ENV_LOG_PREFIX`
                info[f"{AMAGO_ENV_LOG_PREFIX} Ep {self.current_episode} Return"] = self.episode_return
                self.episode_return = 0
                self.current_episode += 1
            # only indicate when the rollout is finished and the env needs to be completely reset
            done = self.current_episode >= self.k
            return next_obs, reward, done, done, info

An important limitation of this is that **while AMAGO will automatically organize meta-RL 
policy inputs for the previous action and reward, it is not aware of the reset signal**. 
If we need the trial reset signal it can go in the observation. 
We could concat an extra feature or make the observation a dict with an extra ``reset`` key. The `amago.envs.builtin` envs contain many examples.

|


Exploration
~~~~~~~~~~~~

Explorative action noise is implemented by a ``gymasium.Wrapper`` (`amago.envs.exploration`). 
Env creation automatically wraps the training envs in ``Experiment.exploration_wrapper_type``.

.. code-block:: python

    from amago.envs.exploration import EpsilonGreedy
    experiment = Experiment(
        exploration_wrapper_type=EpsilonGreedy,
        ...
    )