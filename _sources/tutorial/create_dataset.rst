Create a Dataset
-----------------
|
|

AMAGO trains on sequence data loaded from an :py:class:`~amago.loading.RLDataset` that inherits from the pytorch ``Dataset``. 
Standard online RL can just use :py:class:`~amago.loading.DiskTrajDataset`, 
which tells the envs where to save sequences and deletes the oldest data when full (like a normal replay buffer).

.. code-block:: python

    from amago.loading import DiskTrajDataset

    dataset = DiskTrajDataset(
        dset_root="plenty_of_space",
        dset_name="give_this_replay_buffer_a_name",
        dset_max_size=10_000, # measured in *sequences*
    )
    # creates a directory sturcture like:
    # dset_root/
    #   dset_name/
    #     buffer/
    #       protected/
    #          optional place to move data you want to sample from but never delete
    #       fifo/
    #          envs write files here and dset deletes them when full
    experiment = amago.Experiment(
        ...,
        dataset=dataset,
        # optional control over the way all datasets sample from seqs longer than the policy's max input length
        padded_sampling="none",
        # optional control over the way envs write to the dataset:
        traj_save_len=1000, # write sequences after this many timesteps even if the episode hasn't finished
    )

.. tip::

   If data is coming from some other source (like an existing offline RL dataset) you can inherit from :py:class:`~amago.loading.RLDataset`. :doc:`/examples/14_d4rl` has an example. More on customization by inheritance in :doc:`/tutorial/customization`.