Customize
=====================
|
|

Almost anything else can be customized by inheriting from a base class and pointing ``Experiment`` to our custom version.


Timestep Encoder
~~~~~~~~~~~~~~~~

For example, if we want a custom `TstepEncoder`, we can implement the abstract methods and pass our module into the experiment:

.. code-block:: python

    from torch import nn
    import torch.nn.functional as F

    from amago import TstepEncoder
    # there's no specific requirement to use AMAGO's pytorch modules, but
    # we've built up a collection of common RL components that might be helpful!
    from amago.nets.cnn import NatureishCNN
    from amago.nets.ff import Normalization

    class MultiModalRobotTstepEncoder(TstepEncoder):
        def __init__(
            self,
            obs_space: gym.spaces.Dict,
            rl2_space: gym.spaces.Box,
        ):
            super().__init__(obs_space=obs_space, rl2_space=rl2_space)
            img_space = obs_space["image"]
            joint_space = obs_space["joints"]
            self.cnn = NatureishCNN(img_shape=img_space.shape)
            cnn_out_shape = self.cnn(self.cnn.blank_img).shape[-1]
            self.joint_rl2_emb = nn.Linear(joint_space.shape[-1] + rl2_space.shape[-1], 32)
            self.merge = nn.Linear(cnn_out_shape + 32, 128)
            # we'll represent each Timestep as a 64d vector
            self.output_layer = nn.Linear(128, 64)
            self.out_norm = Noramlization("layer", 64)

        @property
        def emb_dim(self):
            # tell the rest of the model what output shape to expect
            return 64
        
        def inner_forward(self, obs, rl2s, log_dict=None):
            img_features = self.cnn(obs["image"])
            joints_and_rl2s = torch.cat((obs["joints"], rl2s), dim=-1)
            joint_features = F.leaky_relu(self.joint_rl2_emb(joints_and_rl2s))
            merged = torch.cat((img_features, joint_features), dim=-1)
            merged = F.leaky_relu(self.merge(merged))
            out = self.out_norm(self.output_layer(merged))
            return out

    experiment = amago.Experiment(
        ...,
        tstep_encoder_type=MultiModalRobotTstepEncoder,
    )

:doc:`/examples/10_babyai` and :doc:`/examples/11_xland_minigrid` are relevant examples.


TrajEncoder (Seq2Seq)
~~~~~~~~~~~~~~~~~~~~~~

**Implement**: :py:class:`~amago.nets.traj_encoders.TrajEncoder`

**Substitute**: ``Experiment(traj_encoder_type=MyTrajEncoder, ...)``

|

Exploration Strategy
~~~~~~~~~~~~~~~~~~~~

**Implement**: :py:class:`~amago.envs.exploration.ExplorationWrapper`

**Substitute**: ``Experiment(exploration_wrapper_type=MyExplorationWrapper, ....)``

:doc:`/examples/04_tmaze` demonstrates a custom exploration strategy.

|

Agent
~~~~~

**Implement**: :py:class:`~amago.agent.Agent`

**Substitute**: ``Experiment(agent_type=MyAgent, ...)``


|

RLDataset
~~~~~~~~~

**Implement**: :py:class:`~amago.loading.RLDataset`

**Substitute**: ``dset = MyDataset(); experiment = Experiment(dataset=dset, ...)``

:doc:`/examples/14_d4rl` demonstrates a custom dataset.

|

(Continuous) Action Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Implement**: :py:class:`~amago.nets.policy_dists.PolicyOutput`

**Substitute**: ``config = {"amago.nets.actor_critic.Actor.continuous_dist_type" : MyPolicyOutput, ...}; use_config(config); experiment = Experiment(...)``

|