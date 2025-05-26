
Configure
========================
|

The :py:class:`~amago.experiment.Experiment` has lots of other kwargs to control things like the update:data ratio, optimization, and logging.

|
For additional control over the training process, use ``gin``. First, note that the general format of an AMAGO training script is:

.. code-block:: python

    from amago import Experiment
    from amago.envs import AMAGOEnv

    # define the env
    make_env = lambda: AMAGOEnv(...)

    # create a dataset
    dataset = DiskTrajDataset(...)

    # make the main conceptual choices
    tstep_encoder_type = CNNTstepEncoder
    traj_encoder_type = TformerTrajEncoder
    agent_type = Agent
    exploration_wrapper_type = EpsilonGreedy

    experiment = Experiment(
        dataset=dataset,
        # assign lots of *callables*, not instances, to the experiment
        make_train_env=make_env,
        make_val_env=make_env,
        tstep_encoder_type=tstep_encoder_type,
        traj_encoder_type=traj_encoder_type,
        agent_type=agent_type,
        exploration_wrapper_type=exploration_wrapper_type,
        ...
    )

    experiment.start()
    experiment.learn()
    experiment.evaluate_test(make_env)


We choose the :py:class:`~amago.envs.amago_env.AMAGOEnv`, :py:class:`~amago.loading.RLDataset`, and :py:class:`~amago.nets.tstep_encoders.TstepEncoder` 
because they are problem-specific. We also pick the the :py:class:`~amago.nets.traj_encoders.TrajEncoder` because it is the key feature of a sequence model agent.
:py:meth:`~amago.experiment.Experiment.start` is going to create an :py:class:`~amago.agent.Agent` based on the environment and our other choices. This follows a strict rule:

.. important::
    Anytime AMAGO needs to initialize/call a class/method, it infers the *positional* args (based on the environment and our other choices), 
    but **leaves every keyword argument set to its default value**. `gin <https://github.com/google/gin-config>`_ lets us edit those values without editing the source code, and keeps track of the settings we used on ``wandb``. 


The :doc:`/examples/index` show how almost every application of AMAGO looks the same aside from some minor ``gin`` configuration. ``gin`` can be complicated, but AMAGO tries to make it hard to get wrong:

.. tip::
    If something is ``@gin.configurable`` (there will be a note at the top of the documentation), it means that :py:class:`~amago.experiment.Experiment` wlll only ever call/construct it with default keyword arguments, 
    and *there is no other place where those values should be set or will be overridden*. The only exceptions are :py:class:`~amago.experiment.Experiment` and :py:class:`~amago.loading.RLDataset`, 
    which are explicitly constructed by the user before training begins, but are configurable for convenience.

|

For example, let's say we want to switch the :py:class:`~amago.nets.tstep_encoders.CNNTstepEncoder` to use a larger ``IMPALA`` architecture with twice as many channels as usual. The :doc:`API reference </api/amago>` for :py:class:`~amago.nets.tstep_encoders.CNNTstepEncoder` looks like this:

.. autoclass:: amago.nets.tstep_encoders.CNNTstepEncoder
    :noindex:
    :no-members:
    :no-undoc-members:
    :no-special-members:

|

Following our rule, ``obs_space`` and ``rl2_space`` are going be determined for us, but nothing will try to set ``cnn_type``, so it will default to :py:class:`~amago.nets.cnn.NatureishCNN`. The :py:class:`~amago.nets.cnn.IMPALAishCNN` looks like this:

|

.. autoclass:: amago.nets.cnn.IMPALAishCNN
    :noindex:
    :no-members:
    :no-undoc-members:
    :no-special-members:

So we can change the ``cnn_block_depths`` and ``post_group_norm`` by editing these values, but this would *not* be the place to change the ``activation``.
The most orgnaized way to set gin values is with ``.gin`` config files. But we can also do this with:

|

.. code-block:: python

    from amago.nets.cnn import IMPALAishCNN
    from amago.cli_utils import use_config

    config = {
        "amago.nets.tstep_encoders.CNNTstepEncoder.cnn_type" : IMPALAishCNN,
        "IMPALAishCNN.cnn_block_depths" : [32, 64, 64],
    }
    # changes the default values
    use_config(config)

    experiment = Experiment(
        tstep_encoder_type=CNNTstepEncoder,
        ...
    )

|

As a more complicated example, let's say we want to use a :py:class:`~amago.nets.traj_encoders.TformerTrajEncoder` with 6 layers of dimension 512, 16 heads, and sliding window attention with a window size of 256.

|

.. autoclass:: amago.nets.traj_encoders.TformerTrajEncoder
    :noindex:
    :no-members:
    :no-undoc-members:
    :no-special-members:
    :exclude-members: forward, emb_dim, init_hidden_state, reset_hidden_state

.. autoclass:: amago.nets.transformer.SlidingWindowFlexAttention
    :noindex:
    :no-members:
    :no-undoc-members:
    :no-special-members:

|

``gin.REQUIRED`` is reserved for settings that are not commonly used but would be so important and task-specific that there is no good default. You'll get an error if you use one but forget to configure it.

.. code-block:: python

    from amago.nets.traj_encoders import TformerTrajEncoder
    from amago.nets.transformer import SlidingWindowFlexAttention
    from amago.cli_utils import use_config

    config = {
        "TformerTrajEncoder.n_heads" : 16,
        "TformerTrajEncoder.d_model" : 512,
        "TformerTrajEncoder.d_ff" : 2048,
        "TformerTrajEncoder.attention_type": SlidingWindowFlexAttention,
        "SlidingWindowFlexAttention.window_size" : 128,
    }

    use_config(config)
    experiment = Experiment(
        traj_encoder_type=TformerTrajEncoder,
        ...
    )

|

Customizing the built-in :py:class:`~amago.nets.tstep_encoders.TstepEncoder`, :py:class:`~amago.nets.traj_encoders.TrajEncoder`, :py:class:`~amago.agent.Agent` and :py:class:`~amago.envs.exploration.ExplorationWrapper` is so common that there's easier ways to do it in :py:class:`~amago.cli_utils.use_config`. For example, we could've made the changes for all the previous examples at the same time with:

.. code-block:: python

    from amago.cli_utils import switch_traj_encoder, switch_tstep_encoder, switch_agent, switch_exploration, use_config
    from amago.nets.transformer import SlidingWindowFlexAttention
    from amago.nets.cnn import IMPALAishCNN

    config = {
        # these are niche changes customized a level below the `TstepEncoder` / `TrajEncoder`, so we still have to specify them
        "amago.nets.transformer.SlidingWindowFlexAttention.window_size" : 128,
        "amago.nets.cnn.IMPALAishCNN.cnn_block_depths" : [32, 64, 64],
    }
    tstep_encoder_type = switch_step_encoder(config, arch="cnn", cnn_type=IMPALAishCNN)
    traj_encoder_type = switch_traj_encoder(config, arch="transformer", d_model=512, d_ff=2048, n_heads=16, attention_type=SlidingWindowFlexAttention)
    exploration_wrapper_type = switch_exploration(config, strategy="egreedy", eps_start=1.0, eps_end=.01, steps_anneal=200_000)
    # also customize random RL details as an example
    agent_type = switch_agent(config, agent="multitask", num_critics=6, gamma=.998)
    use_config(config)

    experiment = Experiment(
        tstep_encoder_type=tstep_encoder_type,
        traj_encoder_type=traj_encoder_type,
        agent_type=agent_type,
        exploration_wrapper_type=exploration_wrapper_type,
        ...
    )

|

If we want to combine hardcoded changes like these with genuine ``.gin`` files, :py:func:`~amago.cli_utils.use_config` will take the paths.

.. code-block:: python

    # these changes are applied in order from left to right. if we override the same param
    # in multiple configs the final one will count. making gin this complicated is usually a bad idea.
    use_config(config, gin_configs=["environment_config.gin", "rl_config.gin"])

|

.. tip::
    You can view your active gin config (all the active hyperparameters used by an experiment) in the checkpoint directory as ``config.txt``, or on ``wandb`` in the ``Config`` section.


A full API reference is available in the :doc:`/api/amago` section.