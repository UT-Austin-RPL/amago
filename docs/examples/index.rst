Examples
=========

.. toctree::
   :maxdepth: 1
   :caption: Example Training Scripts Table of Contents

   00_meta_frozen_lake
   01_basic_gym
   02_gymnax
   03_popgym_suite
   04_tmaze
   05_dark_key_door
   06_alchemy
   07_metaworld
   08_ale
   09_multitask_procgen
   10_babyai
   11_xland_minigrid
   12_half_cheetah_vel
   13_mazerunner_relabeling
   14_d4rl
|
|

The ``examples/`` folder includes helpful starting points for common cases.

To follow most of the examples you'll need to install the benchmark environments with ``pip install amago[envs]``. If you want to log to ``wandb`` or check out some of the example results, it's worth reading `this section of the tutorial <../tutorial.html#track-the-results>`_. The public ``wandb`` links include example commands (click the "Overview" tab). Building this set of public examples with the new version of AMAGO is an active work in progress.

Use the ``CUDA_VISIBLE_DEVICES`` environment variable to assign basic single-GPU examples to a specific GPU index. Most of the examples share a command line interface. Use ``--help`` for more information.


:doc:`0. Intro to Black-Box Meta-RL: Meta-Frozen-Lake <00_meta_frozen_lake>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/in_context_rl.png
   :alt: frozen_lake_diagram
   :width: 510
   :align: center

Learn more about adaptive policies with help from an intuitive meta-RL problem. Train an agent to adapt over multiple episodes by learning to avoid its previous mistakes.

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/a53gh0wy>`_

|

:doc:`01. Basic Gymnasium <01_basic_gym>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical RL benchmarks are MDPs and can be treated as a simple special case of the full agent. Memory is often redundant but these tasks can be helpful for testing.

`Example wandb for LunarLander-v2 with a Transformer <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/30ndyo2l>`_

`Example wandb for DM Control Suite Cheetah Run <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/0znibfm2>`_

|

:doc:`02. GPU-Accelerated Envs: Gymnax <02_gymnax>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/gymnax_logo.png
   :alt: gymnax_logo
   :width: 110
   :align: left

Like ``gymnasium``, but 1000x faster! Use ``jax`` to add more ``--parallel_actors`` and speedup experiments. `gymnax <https://github.com/RobertTLange/gymnax>`_ includes several interesting memory problems.

`Example wandb for MemoryChain-bsuite <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/7qe1pu41/>`_

**📊 Experimental 📊**. Support for ``gymnax`` is a new feature.

|

:doc:`03. POMDPs: POPGym <03_popgym_suite>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/popgym.png
   :alt: popgym_diagram
   :width: 180
   :align: left

`POPGym <https://arxiv.org/abs/2303.01859>`_ is a collection of memory unit-tests for RL agents. AMAGO is really good at POPGym and turns most of these tasks into quick experiments for fast prototyping. Our ``MultiDomainPOPGym`` env concatenates POPGym domains into a harder one-shot multi-task problem discussed in the followup paper.

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/nhyxu2g1>`_. These settings can be copied across every task in the ICLR paper.

|

:doc:`04. Super Long-Term Recall: T-Maze <04_tmaze>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/tmaze.png
   :alt: tmaze_diagram
   :width: 160
   :align: left

T-Maze is a modified version of the problem featured in `Ni et al., 2023 <https://arxiv.org/abs/2307.03864>`_. T-Maze answers the question: RL issues (mostly) aside, what is the most distant memory our sequence model can recall? When using Transformers, the answer is usually whatever we can fit on the GPU...

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/8t5bdqmu>`_

|

:doc:`05. Finite-Horizon Meta-RL: Dark Key-To-Door <05_dark_key_door>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common meta-RL problem where the environment resets for a fixed number of timesteps (rather than attempts) so that the agent is rewarded for finding a solution quickly in order to finish the task as many times as possible. Loosely based on experiments in `Algorithm Distillation <https://arxiv.org/abs/2210.14215>`_.

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/a6adhfg3>`_

|


:doc:`06. Meta-RL: Symbolic DeepMind Alchemy <06_alchemy>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/alchemy.png
   :alt: icrl_diagram
   :width: 110
   :align: left

Symbolic version of the `DeepMind Alchemy <https://arxiv.org/abs/2102.02926>`_ meta-RL domain.

**🔥 Challenging 🔥**. Alchemy has a hard local max strategy that can take many samples to break. We've found this domain to be very expensive and hard to tune, though we can usually match the pure-RL (VMPO) baseline from the original paper. We've never used Alchemy in our published results but maintain this script as a starting point.

Example wandb from a recent large-scale attempt with the Multi-Task agent: `Actor Process <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/s85fw2kn>`_ or `Learner Process <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/1ic57f70>`_.

|

:doc:`07. Meta-RL: Meta-World <07_metaworld>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/sawyer2.png
   :alt: icrl_diagram
   :width: 110
   :align: left

`Meta-World <https://meta-world.github.io>`_ creates a meta-RL benchmark out of robotic manipulation tasks. Meta-World ML45 is a great example of why we'd want to use the ``MultiTaskAgent`` learning update. For much more information please refer to our NeurIPS 2024 paper.

`Example wandb (MultiTaskAgent on ML45!) <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/gq9s8vxs>`_

|

:doc:`08. Multi-Task RL: Atari <08_ale>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/gopher.png
   :alt: icrl_diagram
   :width: 110
   :align: left

Multi-Task RL is a special case of meta-RL where the identity of each task is directly provided or can be inferred without memory. We focus on the uncommon setting of learning from *unclipped* rewards because it isolates the challenge of optimizing distinct reward functions. See the NeurIPS 2024 paper for more.

`Example wandb for an easy 4-game variant <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/gzgdshjb>`_

|

:doc:`09. Multi-Game Two-Episode Procgen <09_multitask_procgen>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/coinrun2.png
   :alt: icrl_diagram
   :width: 110
   :align: left

Multi-Game `Procgen <https://arxiv.org/abs/1912.01588>`_ has a similar feel to Atari. However, Procgen's procedural generation and partial observability (especially in "memory" mode) is better suited to multi-episodic adaptation. This example highlights the ``TwoAttemptMTProcgen`` setup used by experiments in the second paper.

|

:doc:`10. Multi-Task BabyAI <10_babyai>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/minigrid.png
   :alt: icrl_diagram
   :width: 100
   :align: left

`BabyAI <https://arxiv.org/abs/1810.08272>`_ is a collection of procedurally generated gridworld tasks with simple language instructions. We create a fun multi-task variant for adaptive agents.

`Example multi-seed report <https://wandb.ai/jakegrigsby/amago-v3-reference/reports/Multi-Task-BabyAI-AMAGOv2--Vmlldzo5ODAxNjc1>`_ (which uses an outdated version of AMAGO).

|

:doc:`11. XLand MiniGrid <11_xland_minigrid>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/xland_rules.png
   :alt: icrl_diagram
   :width: 150
   :align: left

`XLand-MiniGrid <https://arxiv.org/abs/2312.12044>`_ is a ``jax``-accelerated environment that brings the task diversity of `AdA <https://arxiv.org/abs/2301.07608>`_ to `Minigrid <https://arxiv.org/abs/2306.13831>`_/BabyAI-style gridworlds.

**📊 Experimental 📊**. Support for XLand MiniGrid is a new feature.

|

:doc:`12. Toy Meta-RL / Locomotion: HalfCheetahVelocity (w/ HalfCheetahV4) <12_half_cheetah_vel>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/half_cheetah.png
   :alt: half_cheetah_diagram
   :width: 100
   :align: left

A more modern remaster of the famous `HalfCheetahVel mujoco meta-RL benchmark <https://arxiv.org/pdf/1703.03400>`_, where the cheetah from the `HalfCheetah-v4 gymnasium task <https://gymnasium.farama.org/environments/mujoco/half_cheetah/>`_ needs to run at a randomly sampled (hidden) target velocity based on reward signals.

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/jveklygo>`_

|

:doc:`13. Hindsight Relabeling: MazeRunner <13_mazerunner_relabeling>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/mazerunner.png
   :alt: mazerunner_diagram
   :width: 100
   :align: left

Off-policy learning makes it easy to relabel old sequence data with new rewards. MazeRunner is a goal-conditioned POMDP navigation problem used to discuss & test the hindsight instruction relabeling technique in our paper. This example includes a template for using hindsight relabeling in the new version of AMAGO.

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/a728v6k0>`_

|

:doc:`14. Offline RL: D4RL <14_d4rl>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/d4rl.png
   :alt: d4rl_diagram
   :width: 100
   :align: left

Offline RL on the (original) `D4RL <https://arxiv.org/pdf/2004.07219>`_ datasets.

`Example wandb <https://wandb.ai/jakegrigsby/amago-v3-reference/runs/9ab15rr8>`_

|

15. Human-Level Competitive Pokémon: Metamon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../media/metamon_icon.png``
   :alt: metamon_icon
   :width: 100
   :align: left

`metamon <https://github.com/UT-Austin-RPL/metamon>`_ used `amago` to train top decile agents in Pokémon Showdown from human and self-collected battle data. 

Check out our `project website <https://metamon.tech>`_!

|

|
|
|