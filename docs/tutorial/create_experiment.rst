Create an Experiment
=====================
|
|

1. Pick a Sequence Embedding (:py:class:`~amago.nets.tstep_encoders.TstepEncoder`)
-------------------------------------------------------------------------------------

Each timestep provides a dict observation along with the previous action and reward. 
AMAGO standardizes its training process by creating a :py:class:`~amago.nets.tstep_encoders.TstepEncoder` to map timesteps to a fixed size representation. 
After this, the rest of the network can be environment-agnostic. 
We include customizable defaults for the two most common cases of images (:py:class:`~amago.nets.tstep_encoders.CNNTstepEncoder`)
and state arrays (:py:class:`~amago.nets.tstep_encoders.FFTstepEncoder`). 
All we need to do is tell the :py:class:`~amago.experiment.Experiment` which type to use:

.. code-block:: python

    from amago.nets.tstep_encoders import CNNTstepEncoder

    experiment = amago.Experiment(
        make_train_env=make_env,
        ...,
        tstep_encoder_type=CNNTstepEncoder,
    )

|

2. Pick a Sequence Model (:py:class:`~amago.nets.traj_encoders.TrajEncoder`)
----------------------------------------------------------------------------

The :py:class:`~amago.nets.traj_encoders.TrajEncoder` is a seq2seq model that enables long-term memory and in-context learning by processing a sequence of :py:class:`~amago.nets.tstep_encoders.TstepEncoder` outputs. :py:mod:`amago.nets.traj_encoders` includes four built-in options:
:py:class:`~amago.nets.traj_encoders.FFTrajEncoder`, :py:class:`~amago.nets.traj_encoders.GRUTrajEncoder`, :py:class:`~amago.nets.traj_encoders.MambaTrajEncoder`, and :py:class:`~amago.nets.traj_encoders.TformerTrajEncoder`.

We can select a :py:class:`~amago.nets.traj_encoders.TrajEncoder` just like a :py:class:`~amago.nets.tstep_encoders.TstepEncoder`:

.. code-block:: python

    from amago.nets.traj_encoders import MambaTrajEncoder

    experiment = amago.Experiment(
        ...,
        traj_encoder_type=MambaTrajEncoder,
    )

|

3. Pick an :py:class:`~amago.agent.Agent`
--------------------

The :py:class:`~amago.agent.Agent` puts everything together and handles actor-critic RL training ontop of the outputs of the :py:class:`~amago.nets.traj_encoders.TrajEncoder`. 
There are two built-in (highly :doc:`configurable </tutorial/configuration>`) options: :py:class:`~amago.agent.Agent` and :py:class:`~amago.agent.MultiTaskAgent`.

We can switch between them with:

.. code-block:: python

    from amago.agent import MultiTaskAgent

    experiment = amago.Experiment(
        ...,
        agent_type=MultiTaskAgent,
    )

|

4. Start the :py:class:`~amago.experiment.Experiment` and Start Training
----------------------------------------------

Launch training with:

.. code-block:: python

    experiment = amago.Experiment(
        # final required args we haven't mentioned
        run_name="some_name", # a name used for checkpoints and logging
        ckpt_base_dir="some/place/", # path to checkpoint directory
        val_timesteps_per_epoch=1000, # give actors enough time to finish >= 1 episode
        max_seq_len=128, # maximum sequence length for the TrajEncoder
        ...
    )
    experiment.start()
    experiment.learn()

Checkpoints and logs are saved in:

.. code-block:: shell

    {Experiment.ckpt_base_dir}
        |-- {Experiment.run_name}/
            |-- config.txt # stores gin configuration details for reproducibility
            |-- wandb_logs/
            |-- ckpts/
                    |-- training_states/
                    |    | # full checkpoint dirs used to restore `accelerate` training runs
                    |    |-- {Experiment.run_name}_epoch_0/
                    |    |-- {Experiment.run_name}_epoch_{Experiment.ckpt_interval}/
                    |    |-- ...
                    |
                    |-- latest/
                    |    |--policy.pt # the latest model weights
                    |-- policy_weights/
                        | # standard pytorch weight files
                        |-- policy_epoch_0.pt
                        |-- policy_epoch_{Experiment.ckpt_interval}.pt
                        |-- ...

Each ``epoch``, we:

1. Interact with the training envs for ``train_timesteps_per_epoch``, creating a total of ``parallel_actors * train_timesteps_per_epoch`` new timesteps.
2. Save any training sequences that have finished, if applicable.
3. Compute the RL training objectives on ``train_batches_per_epoch`` batches sampled from the dataset.  Gradient steps are taken every ``batches_per_update`` batches.