# AMAGO 

<img src="media/amago_logo_2.png" alt="amagologo" width="150" align="right"/>

### A simple and scalable agent for sequence-based Reinforcement Learning

AMAGO is POMDP solver with an emphasis on long sequences, sparse rewards, and large networks. It is:
- **Broadly Applicable**. Environments are converted to a universal sequence format for memory-enabled policies. Classic benchmarks, goal-conditioning, long-term memory, meta-learning, and generalization are all special cases of its POMDP format. Supports discrete and continuous actions.
- **Scalable**. AMAGO is powered by an efficient learning update that looks more like supervised sequence modeling than RL. It is specifically designed for training large policies (like Transformers) on long sequences (500+ timesteps). Large replay buffers are stored on disk.
- **Sample Efficient**. AMAGO is off-policy and can continuously reuse (and hindsight relabel) large datasets of past trajectories.
- **Easy to Use**. Technical details can be easily customized but are designed to require little hyperparamter tuning.


#### [Paper](https://arxiv.org/abs/2310.09971) | [Project Website](https://ut-austin-rpl.github.io/amago/)


## Background
AMAGO treats multi-task RL, meta-RL, RL generalization, and long-term memory as variations of the same POMDP problem that can be solved with sequence learning. This core framework goes by many different names including [implicit partial observability](https://arxiv.org/abs/2107.06277), [context-based meta-learning](https://arxiv.org/abs/2301.08028), or [contextual MDPs](https://arxiv.org/abs/2111.09794).

AMAGO learns from off-policy or offline data, which improves sample efficiency and enables hindsight experience replay for goal-conditioned environments. All of the low-level details have been redesigned from scratch to be scalable enough to train Transformers on long sequences. In general, AMAGO's training process looks more like supervised sequence modeling than state-of-the-art off-policy RL: trajectory sequences are saved and loaded from disk and RL loss functions are optimized by a single sequence model in one forward pass.

## Installation 

```bash
# download source
git clone git@github.com:UT-Austin-RPL/amago.git
cd amago
# make a fresh conda environment with python 3.10
conda create -n amago python==3.10
conda activate amago
# install core agent
pip install -e .
# AMAGO includes built-in support for a number of benchmark environments that can be installed with:
pip install -e .[envs]
```
The default Transformer policy (`nets.traj_encoders.TformerTrajEncoder`) has an option for [FlashAttention 2.0](https://github.com/Dao-AILab/flash-attention). FlashAttention leads to significant speedups on long sequences if your GPU is compatible. We try to install this for you with: `pip install -e .[flash]`, but please refer to the [official installation instructions](https://github.com/Dao-AILab/flash-attention) if you run into issues.

## Getting Started
Applying AMAGO to any new environment requires 6 basic choices. The `examples/` folder includes helpful starting points for common cases.

<p align="center">
<img src="media/amago_overview.png" alt="amagoarch" width="850"/>
</p>

1. **How do timesteps become vectors?** AMAGO standardizes its training process by creating a `TstepEncoder` to map timesteps to a fixed size representation. Timesteps (`hindsight.Timestep`) consist of observations, meta-RL data (rewards, dones, actions, reset signals), and optional goal descriptions. We include customizable defaults for the two most common cases of images and state arrays.
   
2. **How do sequences of timestep vectors become sequences of POMDP state vectors?** The `TrajEncoder` is a seq2seq model that enables long-term memory and in-context learning by processing `TstepEncoder` outputs. AMAGO is designed for large-scale sequence models like Transformers.

> **Quick Start:** Switch the `TstepEncoder` to a CNN (`CNNTstepEncoder`) for pixel environments or an MLP (`FFTstepEncoder`) for array environments. The `TformerTrajEncoder` is a very strong/stable default Transformer model. The `GRUTrajEncoder` (RNN) is a good backup option when memory is not as important, or when FlashAttention can't be installed.
  
3. **Do we need to use to the hindsight instruction relabeling technique from the AMAGO paper?** If yes, environment setup requires a few extra steps and we will refer you to a longer explanation (coming soon). Most other environments should be compatible automatically, and all of AMAGO's goal-conditioned features can be toggled off.

4. **What defines the end of a rollout?** Are we optimizing a single trial that ends when the environment terminates, or is this a meta-RL setting where we should auto reset to the same environment until a fixed time limit is reached? We can also define an exploration grace period where rewards do not count towards the return (see [E-RL^2](https://arxiv.org/abs/1803.01118), [DREAM](https://arxiv.org/abs/2008.02790), ...). In any case we need to define an upper bound on the rollout length (`horizon`).
   
5. **What is the memory limit of our policy?** True meta-RL and long-term memory tasks would require a `max_seq_len` >= the horizon. If the horizon is unrealisticlly long or unncessary for the complexity of the problem, we can approximate by shortening the context.
   
6. **How often do we save sequences as training samples, even if the rollout hasn't ended?** Parallel actors automatically save `.traj` files for rollouts up to this length, which should >= `max_seq_len`. As a general rule, `traj_save_len > horizon` unless rollouts are >> `max_seq_len` and we need to speedup reads from disk during training. This is not an option when hindsight relabeling because we need to see the entire trajectory to accurately relabel.

<p align="center">
<img src="media/context_length_diagram.png" alt="contextlength" width="850"/>
</p>

> **Quick Start:** Set all the sequence lengths to the `horizon` ("max context") when possible, and `max_seq_len < traj_save_len < horizon` ("partial context") otherwise. AMAGO should train stably even when it has too much context for simple problems.

## A Tour of AMAGO in 9 Examples

To follow most of the examples you'll need to install the benchmark environments with `pip install amago[envs]`.
AMAGO logs to [`wandb`](https://docs.wandb.ai). You can configure the project and account with environment variables:

```bash
export AMAGO_WANDB_PROJECT="wandb project name"
export AMAGO_WANDB_ENTITY="wandb username"
```

Example `wandb` training logs for each example coming soon.

1. **Regular MDPs (Classic Gym)**
   
Many popular benchmarks are MDPs and can be treated as a simple special case of the full agent. Toggling *off* memory, goal-conditioning/relabeling, and multi-episodic resets reduces AMAGO to a regular off-policy actor-critic like you've seen before. This is mainly meant to be an ablation to test the impact of memory. However, AMAGO is a stable variant of [REDQ](https://arxiv.org/abs/2101.05982)/[CRR](https://arxiv.org/abs/2006.15134) with improvements like "multi-gamma" training that are especially useful in sparse reward environments. See `examples/01_basic_gym.py` for an example.

2. **POMDPs and Long-Term Memory (POPGym)**
   
Using a memory-equipped `TrajEncoder`, but toggling *off* goals and multi-episodic resets creates an effective POMDP solver. AMAGO is efficient enough to use *entire* trajectories as context beyond 1k timesteps depending on model size, and the `TformerTrajEncoder` is a strong default Transformer tuned specifically for stability in RL. See `examples/02_popgym_suite.py` where the same hyperparameters can lead to state-of-the-art performance across the [POPGym](https://arxiv.org/abs/2303.01859) suite.

 
3. **Fixed-Horizon Meta-RL (Dark-Key-To-Door)**

From AMAGO's perspective, meta-RL problems are just POMDPs that automatically reset the task up until a fixed time limit. `TrajEncoder` sequence models let us remember and improve upon past attempts. `examples/03_dark_key_to_door.py` walks through a toy example from the [Algorithm Distillation](https://arxiv.org/abs/2210.14215) paper.

4. **Zero-Shot Adaptation to Goal-Conditioned Environments (Mazerunner)**

`examples/04_mazerunner.py` uses the hindsight instruction relabeling technique from the AMAGO paper on our MazeRunner navigation domain. The ability to relabel rewards in hindsight is a key advantage of off-policy adaptive agents.

5. and 6. **K-Shot Meta-RL (Metaworld and Alchemy)**

`examples/05_kshot_metaworld.py` uses [Metaworld](https://meta-world.github.io) to show how we can setup a meta-RL problem that ends after a certain number of episodes, rather than a fixed horizon `H`. We can let the environment automatically reset itself `k - 1` times while AMAGO pretends it's a zero-shot problem (as long as the resets are added to the observation space). `examples/06_alchemy.py` shows another example on the symbolic version of [DeepMind Alchemy](https://arxiv.org/abs/2102.02926).

7. **Goal-Conditioned Open-Worlds (Crafter)**

AMAGO can adapt to procedurally generated environments while completing multi-step instructions. `examples/07_crafter_with_instructions.py` shows how we turn [Crafter](https://danijar.com/project/crafter/) into an instruction-conditioned environment, and then use AMAGO's hindsight relabeling to explore sparse rewards. Please note that the open-source release is currently missing some of the features used in the paper's Crafter results, but these will be added in the coming weeks.
  
8. **Multi-Task Learning from Pixels (Retro Games)**

Meta-learning and multi-task learning without task labels are essentially the same problem. We can use short sequences to train an agent on multiple levels of the same video game (or multiple games). AMAGO's sequence format can be surprisingly helpful in cases where traditional frame-stacking isn't enough to make sense of action spaces that were designed for human players. `examples/08_multitask_mario.py` provides an example on levels from Super Mario using [stable-retro](https://stable-retro.farama.org). Note that this script requires some extra installation beyond `pip install amago[envs]`. 

9. **Super Long-Term Memory (Passive T-Maze)**

AMAGO provides a stable way to train long-sequence Transformers with RL, which can turn traditionally hard memory-based environments into simple problems. `examples/09_tmaze.py` adds a few exploration changes to the TMaze environment from [Ni et al., 2023](https://arxiv.org/abs/2307.03864), which lets us recall information for thousands of timesteps. We have been able to solve this environment all the way until `H=10,000` before finally running out of GPU memory for the `TformerTrajEncoder`.

## Advanced Configuration

AMAGO is built around [gin-config](https://github.com/google/gin-config), which makes it easy to customize experiments. `gin` makes hyperparameters a default value for an object's `kwargs`, and lets you set their value without editing the source code. You can read more about gin [here](https://github.com/google/gin-config/blob/master/docs/index.md). gin is great for research, but isn't ideal for usability... you do have to read the code to know what you're changing. The `examples/` avoid any `.gin` config files and let you switch between the most important settings without worrying about any of this.

## Roadmap

- Multimodal (dict-based) observations
- `MultiBinary` and `MultiDiscrete` action spaces
- Faster backward pass with second target critic using `tau=1.0`

## Reference and Acknowledgements
If you use AMAGO in your research, please consider citing our paper:

```
@article{grigsby2023amago,
  title={AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents},
  author={Jake Grigsby and Linxi Fan and Yuke Zhu},
  year={2023},
  eprint={2310.09971},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

Logo by [pngtree.com](pngtree.com)
