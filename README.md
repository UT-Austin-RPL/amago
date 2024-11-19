# AMAGO 
### Adaptive RL with Long-Term Memory

#### [[ICLR 2024](https://openreview.net/pdf?id=M6XWoEdmwf)] [[NeurIPS 2024](https://openreview.net/pdf?id=OSHaRf4TVU)]

<img src="media/amago_logo_2.png" alt="amagologo" width="175" align="right"/>


AMAGO follows a simple and scalable recipe for building RL agents that can generalize:
1. Turn meta-learning into a *memory* problem ("in-context RL") .
2. Put all of our effort into learning effective memory with end-to-end RL.
3. Treat zero-shot generalization and multi-task RL as special cases of meta-learning.
4. Then, we can use one method to solve a wide range of problems!

<br>

AMAGO is a high-powered off-policy version of [RL^2](https://arxiv.org/abs/1611.02779) for training large policies on long sequences. Please refer to our [paper](https://arxiv.org/abs/2310.09971) for a detailed explanation. Some highlights:

- **Broadly Applicable**. Long-term memory, meta-learning, multi-task RL, and zero-shot generalization are all special cases of its POMDP format. Supports discrete, continuous, and multi-binary actions. See examples below!
- **Scalable**. Train large policies on long context sequences across multiple GPUs with parallel actors, asynchronous learning/rollouts, and large replay buffers stored on disk.
- **Easy to Use**. Quickstart experiments on a broad range of environments. Technical details are easy to customize but designed to require little hyperparameter tuning.



## What is In-Context RL?

<p align="center">
<img src="media/in_context_rl.png" alt="icrl_diagram" width="900"/>
</p>

Standard RL agents can only generalize to aspects of their environment that 1) *they can observe* and 2) that *changed during training*. In other words, they cannot adapt to changes that are not explicitly revealed, no matter how much experience we collect or how much variety our environment provides. 



**Meta-RL** agents adapt to changes they *cannot directly observe*; these might range from subtle adjustments of their controls to entirely new reward functions. They do this by exploring their surroundings, inferring what they do not know, and adjusting their decisions to succeed in their current environment.

**In-Context RL** (ICRL), a.k.a *Black-Box Meta-RL*, is a simple approach that lets meta-learning emerge inside a sequence model. The idea is this: RL's goal is to maximize returns, and we could increase returns if we knew more about the environment, so meta-learning will happen naturally. ICRL effectively reduces meta-RL to the problem of training RL with memory. Its main advantage is its flexibility: ICRL blurs formal boundaries between generalization, meta-learning, multi-task RL, and long-term memory by letting us use the same method for every problem!

However, In-Context RL has two key disadvantages:

1. **Memory in RL is hard**, so reducing adaptation to memory may not actually get us very far. 
2. **Sample inefficiency**. ICRL is deep RL at its most extreme. We make no assumptions about the problem and let a fancy sequence model figure it out from data... and it'll take a lot of data. 

ICRL is not a new idea, but these challenges have limited adoption and prompted research on many alternative approaches. AMAGO is an effort to improve them and push meta-RL beyond toy research problems.


<br>

## Installation 
```bash
# download source
git clone git@github.com:UT-Austin-RPL/amago.git
# make a fresh conda environment with python 3.10
conda create -n amago python==3.10
conda activate amago
# install core agent
pip install -e amago
```

There are some optional installs for additional features:

- `pip install -e amago[flash]`: The default Transformer policy has an option for [FlashAttention 2.0](https://github.com/Dao-AILab/flash-attention). FlashAttention leads to significant speedups on long sequences if your GPU is compatible. Please refer to the [official installation instructions](https://github.com/Dao-AILab/flash-attention) if you run into issues.

- `pip install -e amago[mamba]`: Enables [Mamba](https://arxiv.org/abs/2312.00752) sequence model policies. 

- `pip install -e amago[envs]`: AMAGO comes with built-in support for a wide range of existing and custom meta-RL/generalization/memory domains (`amago/envs/builtin`) used in our experiments. This command installs (most of) the dependencies you'd need to run the [`examples/`](examples/).

> *NOTE*: AMAGO requires `gymnasium` <= 0.29. It is not compatible with the recent `gymnasium` 1.0 release. Please check your `gymnasium` version if you see environment-related error messages on startup.

This is an active long-term research project. Please be warned that the codebase is not stable and we make breaking changes frequently. 

<br>

## Tutorial
You can read a detailed tutorial in [here](tutorial.md). Full documentation coming soon.


<br>

## Examples

The [`examples/`](examples/) folder includes helpful starting points for common cases.


To follow most of the examples you'll need to install the benchmark environments with `pip install amago[envs]`. If you want to log to `wandb`, you can configure the project and account with environment variables:

```bash
export AMAGO_WANDB_PROJECT="wandb project name"
export AMAGO_WANDB_ENTITY="wandb username"
```

Use the `CUDA_VISIBLE_DEVICES` environment variable to assign basic single-GPU examples to a specific GPU index. Most of the examples share a command line interface. Use `--help` for more information.


The public `wandb` links include example commands (click the "Overview" tab). Building this set of public examples with the new version of AMAGO is an active work in progress.


### 0. **Intro to In-Context RL: Meta-Frozen Lake**
**[`00_kshot_frozen_lake.py`](examples/00_kshot_frozen_lake.py)**

<img src="media/robot.png" alt="frozen_lake_icon" width="110" align="left"/>

Learn more about in-context RL with help from an intuitive meta-RL problem. Train an agent to adapt over multiple episodes by learning to avoid its previous mistakes.

[Example `wandb`](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/a53gh0wy?nw=nwuserjakegrigsby)

<br>


### 1. **Basic Gymnasium**
**[`01_basic_gym.py`](examples/01_basic_gym.py)**

Typical RL benchmarks are MDPs and can be treated as a simple special case of the full agent. Memory is often redundant but these tasks can be helpful for testing.

[Example `wandb` for LunarLander-v2 with a Transformer](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/30ndyo2l?nw=nwuserjakegrigsby)

[Example `wandb` for DM Control Suite Cheetah Run](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/0znibfm2?nw=nwuserjakegrigsby)

<br>

### 2. **GPU-Accelerated Envs: Gymnax**
**[`02_gymnax.py`](examples/02_gymnax.py)**


<img src="media/gymnax_logo.png" alt="gymnax_logo" width="110" align="left"/>

Like `gymnasium`, but 1000x faster! Use `jax` to add more `--parallel_actors` and speedup experiments. [`gymnax`](https://github.com/RobertTLange/gymnax) includes several interesting memory problems.

[Example `wandb` for MemoryChain-bsuite](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/7qe1pu41/workspace?nw=nwuserjakegrigsby)

**📊 Experimental 📊**. Support for `gymnax` is a new feature.

<br>

### 3. **POMDPs: POPGym**
**[`03_popgym_suite.py`](examples/03_popgym_suite.py)**

<img src="media/popgym.png" alt="popgym_diagram" width="180" align="left"/>

[POPGym](https://arxiv.org/abs/2303.01859) is a collection of memory unit-tests for RL agents. AMAGO is really good at POPGym and turns most of these tasks into quick experiments for fast prototyping. Our `MultiDomainPOPGym` env concatenates POPGym domains into a harder one-shot multi-task problem discussed in the followup paper.

[Example `wandb`](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/nhyxu2g1?nw=nwuserjakegrigsby). These settings can be copied across every task in the ICLR paper.

<br>

### 4. **Super Long-Term Recall: T-Maze**
**[`04_tmaze.py`](examples/04_tmaze.py)**

<img src="media/tmaze.png" alt="tmaze_diagram" width="160" align="left"/>

T-Maze is a modified version of the problem featured in [Ni et al., 2023](https://arxiv.org/abs/2307.03864). T-Maze answers the question: RL issues (mostly) aside, what is the most distant memory our sequence model can recall? When using Transformers, the answer is usually whatever we can fit on the GPU...

[Example `wandb`](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/8t5bdqmu?nw=nwuserjakegrigsby)

<br>

### 5. **Finite-Horizon Meta-RL: Dark Key-To-Door**
**[`05_dark_key_door.py`](examples/05_dark_key_door.py)**

A common meta-RL problem where the environment resets for a fixed number of timesteps (rather than attempts) so that the agent is rewarded for finding a solution quickly in order to finish the task as many times as possible. Loosely based on experiments in [Algorithm Distillation](https://arxiv.org/abs/2210.14215).

<br>

### 6. **Meta-RL: Symbolic DeepMind Alchemy**
**[`06_alchemy.py`](examples/06_alchemy.py)**


<img src="media/alchemy.png" alt="icrl_diagram" width="110" align="left"/>


Symbolic version of the [DeepMind Alchemy](https://arxiv.org/abs/2102.02926) meta-RL domain.

**🔥 Challenging 🔥**. Alchemy has a hard local max strategy that can take many samples to break. We've found this domain to be very expensive and hard to tune, though we can usually match the pure-RL (VMPO) baseline from the original paper. We've never used Alchemy in our published results but maintain this script as a starting point.

Example `wandb` from a recent large-scale attempt with the Multi-Task agent: [Actor Process](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/s85fw2kn?nw=nwuserjakegrigsby) or [Learner Process](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/1ic57f70?nw=nwuserjakegrigsby).



<br>


### 7. **Meta-RL: Meta-World**
**[`07_metaworld.py`](examples/07_metaworld.py)**

<img src="media/sawyer2.png" alt="icrl_diagram" width="110" align="left"/>

[Meta-World](https://meta-world.github.io) creates a meta-RL benchmark out of robotic manipulation tasks. Meta-World ML45 is a great example of why we'd want to use the `MultiTaskAgent` learning update. For much more information please refer to our NeurIPS 2024 paper.

[Example `wandb` (`MultiTaskAgent` on ML45!)](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/gq9s8vxs?nw=nwuserjakegrigsby).

<br>

### 8. **Multi-Task RL: Atari**
**[`08_ale.py`](examples/08_ale.py)**

<img src="media/gopher.png" alt="icrl_diagram" width="110" align="left"/>

Multi-Task RL is a special case of meta-RL where the identity of each task is directly provided or can be inferred without memory. We focus on the uncommon setting of learning from *unclipped* rewards because it isolates the challenge of optimizing distinct reward functions. See the NeurIPS 2024 paper for more.

[Example `wandb` for an easy 4-game variant](https://wandb.ai/jakegrigsby/amago-v3-reference/runs/gzgdshjb?nw=nwuserjakegrigsby)

<br>

### 9. **Multi-Game Two-Episode Procgen**
**[`09_multitask_procgen.py`](examples/09_multitask_procgen.py)**

<img src="media/coinrun2.png" alt="icrl_diagram" width="110" align="left"/>

Multi-Game [Procgen](https://arxiv.org/abs/1912.01588) has a similar feel to Atari. However, Procgen's procedural generation and partial observability (especially in "memory" mode) is better suited to multi-episodic adaptation. This example highlights the `TwoShotMTProcgen` setup used by experiments in the second paper.

<br>

### 10. **Multi-Task BabyAI**
**[`10_babyai.py`](examples/10_babyai.py)**

<img src="media/minigrid.png" alt="icrl_diagram" width="100" align="left" />

[BabyAI](https://arxiv.org/abs/1810.08272) is a collection of procedurally generated gridworld tasks with simple lanugage instructions. We create a fun multi-task variant for adaptive agents.

[Example multi-seed report](https://wandb.ai/jakegrigsby/amago-v3-reference/reports/Multi-Task-BabyAI-AMAGOv2--Vmlldzo5ODAxNjc1) (which uses an outdated version of AMAGO).


<br>

### **11. XLand MiniGrid**
**[`11_xland_minigrid.py`](examples/11_xland_minigrid.py)**


<img src="media/xland_rules.png" alt="icrl_diagram" width="150" align="left" />

[XLand-MiniGrid](https://arxiv.org/abs/2312.12044) is a `jax`-accelerated environment that brings the task diversity of [AdA](https://arxiv.org/abs/2301.07608) to [Minigrid](https://arxiv.org/abs/2306.13831)/BabyAI-style gridworlds.

**📊 Experimental 📊**. Support for XLand MiniGrid is a new feature. 


<br>

---

<br>



## Citation
```
@inproceedings{
  grigsby2024amago,
  title={{AMAGO}: Scalable In-Context Reinforcement Learning for Adaptive Agents},
  author={Jake Grigsby and Linxi Fan and Yuke Zhu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=M6XWoEdmwf}
}

```

```
@inproceedings{
  grigsby2024amago,
  title={{AMAGO}-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers},
  author={Jake Grigsby and Justin Sasek and Samyak Parajuli and Daniel Adebi and Amy Zhang and Yuke Zhu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=OSHaRf4TVU}
}
```

