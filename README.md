# AMAGO: Adaptive RL
### Memory is All You Need

<img src="media/amago_logo_2.png" alt="amagologo" width="200" align="right"/>


AMAGO follows a simple and scalable recipe for building RL agents that can generalize:
1. Reduce meta-learning to a memory problem (aka "in-context learning") .
2. Put all of our effort into making memory+RL work really well.
3. Treat everything else as a special case of meta-learning.
4. Then, we can use the same in-context meta-learning method to solve a wide range of problems!

This perspective goes by many different names including [implicit partial observability](https://arxiv.org/abs/2107.06277), [context-based meta-learning](https://arxiv.org/abs/2301.08028), or [contextual MDPs](https://arxiv.org/abs/2111.09794). AMAGO is basically a high-powered off-policy version of [RL^2](https://arxiv.org/abs/1611.02779) for training large models with many parallel actors on multiple GPUs. All of the low-level details have been redesigned from scratch to be scalable enough to train long-context Transformers. Please refer to our [paper](https://arxiv.org/abs/2310.09971) for a detailed explanation. Some highlights:

- **Broadly Applicable**. Classic single-task control, goal-conditioning, long-term memory, meta-learning, multi-task RL, and zero-shot generalization are all special cases of its POMDP format. Supports discrete and continuous actions.
- **Scalable**. Designed for training large policies on long sequences. AMAGO supports multi-GPU training with parallel actors, asynchronous learning/rollouts, and large replay buffers (stored on disk).
- **Sample Efficient**. AMAGO is off-policy and can continuously reuse large datasets of past trajectories.
- **Easy to Use**. Quickstart experiments on a wide range of environments with example training scripts (`examples/`). Technical details are easy to customize but are designed to require little hyperparameter tuning. See a detailed tutorial below.


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

- `pip install -e amago[envs]`: AMAGO comes with built-in support for a wide range of existing and novel meta-RL/generalization/memory domains (`amago/envs/builtin`). This command installs most of the environments you'd need to run the [`examples/`](examples/).

This is an active long-term research project. Please be warned that the codebase is not stable and we make breaking changes frequently. 

<br>

## Tutorial

`amago.Experiment` creates the main rollout/learning/testing loop. Applying AMAGO to any new environment involves 7 basic steps. The `examples/` folder includes helpful starting points for common cases.

<p align="center">
<img src="media/amago_overview.png" alt="amagoarch" width="950"/>
</p>

<br>

### **1. Setup the Environment**
Standard `gymnasium` envs simulate a single instance of an environment. We'll be collecting data in parallel by creating multiple independent instances. All we need to do is define a function that creates an `AMAGOEnv`, for example:

```python
import gymnasium
import amago

def make_env():
  env = gymnasium.make("Pendulum-v1")
  # `env_name` is used for logging. Some multi-task envs will sample a new env between resets and change the name accordingly.
  env = amago.envs.AMAGOEnv(env=env, env_name="Pendulum", batched_envs=1)
  return env


sample_env = make_env()
# If the env doesn't have dict observations, AMAGOEnv will create one with a default key of 'observation':
sample_env.observation_spce
 >>> Dict('observation': Box([-1. -1. -8.], [1. 1. 8.], (3,), float32))
# environments return an `amago.hindsight.Timestep`
sample_timestep, info = sample_env.reset()
# each environment has a batch dimension of 1
sample_timestep.obs["observation"].shape
 >>> (1, 3) 

experiment = amago.Experiment(
  make_train_env=make_env,
  make_val_env=make_env,
  parallel_actors=36,
  env_mode="async", # or "sync" for easy debugging / reduced overhead
  ..., # we'll be walking through more arguments below
)

```

#### Vectorized Envs and `jax`
<details>

Some domains already parallelize computation over many environments such that `step` expects a batch of actions and returns a batch of observations. Examples include recent envs like [`gymnax`](https://github.com/RobertTLange/gymnax) that use [`jax`](https://jax.readthedocs.io/en/latest/) and a GPU to boost their framerate:
```python
import gymnax 
from amago.envs.builtin.gymnax_envs import GymnaxCompatability

def make_env():
  env, params = gymnax.make("Pendulum-v1")
  # AMAGO expects numpy data and an unbatched observation space
  vec_env = GymnaxCompatability(env, num_envs=512, params=params)
  # vec_env.reset()[0].shape >>> (512, 3) # already vectorized!
  return AMAGOEnv(env=vec_env, env_name=f"gymnax_Pendulum", batched_envs=512)

experiment = amago.Experiment(
  make_train_env=make_env,
  make_val_env=make_env,
  parallel_actors=512, # match batch dim of environment
  env_mode="already_vectorized", # prevents spawning multiple async instances
  ...,
)
```

There are some details in getting the pytorch agent and jax envs to cooperate and share a GPU. See `examples/02_gymnax.py`.

> *NOTE*: Support for `jax` and other GPU-batched envs is a recent experimental feature. Please refer to the latest `jax` documentation for instructions on installing versions compatible with your hardware.

</details>

<br>

#### Meta-RL and Auto-Resets

<details>

Most meta-RL problems involve an environment that resets itself to the same task `k` times. There is no consistent way to handle this across different benchmarks. Therefore, **AMAGO expects the environment to be handling `k`-shot resets on its own.** `terminated` and `truncated` indicate that this environment interaction is finished and should be saved/logged. For example:

```python
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
```

An important limitation of this is that **while AMAGO will automatically organize meta-RL policy inputs for the previous action and reward, it is not aware of the reset signal**. If we need the trial reset signal it can go in the observation. We could concat an extra feature or make the observation a dict with an extra `reset` key. The `envs/builtin/` envs contain many examples.

</details>

<br>

### **2. Pick a Sequence Embedding (`TstepEncoder`)**

Each timestep provides a dict observation along with meta-RL deta like the previous action and reward. AMAGO standardizes its training process by creating a `TstepEncoder` to map timesteps to a fixed size representation. After this, the rest of the network can be environment-agnostic. We include customizable defaults for the two most common cases of images (`nets.tstep_encoders.CNNTstepEncoder`) and state arrays (`nets.tstep_encoders.FFTstepEncoder`). All we need to do is tell the `Experiment` which type to use:

```python
from amago.nets.tstep_encoders import CNNTstepEncoder

experiment = amago.Experiment(
  make_train_env=make_env,
  ...,
  tstep_encoder_type=CNNTstepEncoder,
)
```

#### Create Your Own `TstepEncoder`

<details>

If our environment has multi-modal dict observations or we want to customize the network in a way that isn't covered by the defaults' options, we could do something like this:

```python
from torch import nn
import torch.nn.functional as F

from amago import TstepEncoder
# there's no specific requirement to use AMAGO's pytorch modules, but
# we've built up a collection of common RL pieces that might be helpful!
from amago.nets.cnn import NatureishCNN
from amago.nets.ff import Normalization

class MultiModalRobotTstepEncoder(TstepEncoder):
  def __init_(
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
    # tell the rest of the model what shape to expect
    return 64
  
  def inner_forward(self, obs, rl2s, log_dict=None):
    """
    `obs` is a dict and `rl2s` are the previous reward + action.
    All tensors have shape (batch, length, dim)
    """
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
``` 
</details>

<br>

### **3. Pick a Sequence Model (`TrajEncoder`)**

The `TrajEncoder` is a seq2seq model that enables long-term memory and in-context learning by processing a sequence of `TstepEncoder` outputs. `nets.traj_encoders` includes four built-in options :

1. `FFTrajEncoder`: processes each timestep independently with a residual feedforward block. It has no memory! This is a useful sanity-check that isolates the impact of memory on performance.

2. `GRUTrajEncoder`: a recurrent model. Long-term recall is challenging because we need to learn what to remember or forget at each timestep, and it may a while before new info is relevant to decision-making. However, inference speed is constant over long rollouts.

3. `MambaTrajEncoder`: [Mamba](https://arxiv.org/abs/2312.00752) is a state-space model with similar conceptual strengths and weaknesses as an RNN. However, it runs significantly faster during training.

4. `TformerTrajEncoder`: a Transformer model with a number of tricks for stability in RL. Transformers are great at RL memory problems because they don't "forget" anything and only need to learn to *retrieve* info at timesteps where it is immediately useful. There are several choices of self-attention mechanism. We reccomend [flash_attn](https://github.com/Dao-AILab/flash-attention) if it will run on your GPU. If not, we'll fall back to a slower pytorch version. There is experimental support for [`flex_attention`](https://pytorch.org/blog/flexattention/) --- a cool feature coming to pytorch 2.5. See the "Customize Anything" section for how to switch defaults.

We can select a `TrajEncoder` just like a `TstepEncoder`:

```python
from amago.nets.traj_encoders import MambaTrajEncoder

experiment = amago.Experiment(
  ...,
  traj_encoder_type=MambaTrajEncoder,
)
```

If we wanted to try out a new sequence model we could subclass `amago.TrajEncoder`like the `TstepEncoder` example above.


<br>

### **4. Pick an `Agent`**
The `Agent` puts everything together and handles actor-critic RL training ontop of the outputs of the `TrajEncoder`. There are two high-level options:

1. `Agent`: the default learning update is described in Appendix A of the paper. It's an off-policy actor-critic (think [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)) with some stability tricks like random critic ensembling and training over multiple discount factors in parallel.

2. `MultiTaskAgent`: The `Agent` training objectives depend on the scale of returns (`Q(s, a)`) across our dataset, which might be a problem when those returns vary widely... like when we're training on multiple tasks at the same time. `MultiTaskAgent` replaces critic regression with two-hot classification, and throws out the policy gradient in favor of filtered behavior cloning. This update is much better in hybrid meta-RL/multi-task problems where we're optimizing multiple reward functions (like Meta-World ML45, Multi-Game Procgen, or Multi-Task BabyAI). We wrote a whole second paper about it (coming soon NeurIPS 24)! 

We can switch between them with:
```python
from amago.agent import MultiTaskAgent

experiment = amago.Experiment(
  ...,
  agent_type=MultiTaskAgent,
)
```

It is also possible to subclass `amago.Agent` if we want to try something new. This would be much messier than changing the network architecture and would probably involve copy/pasting the base `Agent` and editing a few lines of code.

<br>

### **5. Configure the `Experiment`**
The `Experiment` has lots of other kwargs to control things like the ratio of data collection to learning updates, optimization, and logging. We might set up formal documentation at some point. For now, you can find an explanation of each setting in the comments at the top of `amago/experiment.py`

<br>

### **6. Configure Anything Else**
We try to keep the settings of each `Experiment` under control by using [`gin`](https://github.com/google/gin-config) to configure individual pieces like the `TstepEncoder`, `TrajEncoder`, `Agent`, and actor/critic heads. You can read more about `gin` [here](https://github.com/google/gin-config/blob/master/docs/index.md), but hopefully won't need to. We try to make this easy: our code follows a simple rule that, if something is marked `@gin.configruable`, none of its `kwargs` are set, meaning that the default value always gets used. `gin` lets you change that default value without editing the source code, and keeps track of the settings you used on `wandb` and in a `config.txt` file saved with your model checkpoints.

The `examples/` show how almost every application of AMAGO looks the same aside from some minor `gin` configuration.

#### CNN Architecture Example
<details>

For example, let's say we want to switch the `CNNTstepEncoder` to use a larger IMPALA architecture with twice as many channels as usual. The constructor for `CNNTstepEncoder` looks like this:

```python
# amago/nets/tstep_encoders.py
@gin.configurable
class CNNTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        rl2_space,
        cnn_type=cnn.NatureishCNN,
        channels_first: bool = False,
        img_features: int = 384,
        rl2_features: int = 12,
        d_output: int = 384,
        out_norm: str = "layer",
        activation: str = "leaky_relu",
        skip_rl2_norm: bool = False,
        hide_rl2s: bool = False,
        drqv2_aug: bool = False,
    ):
```

Following our rule, `obs_space` and `rl2_space` are going be determined for us, but nothing will try to set `cnn_type`, so it will default to `NatureishCNN`. The `IMPALAishCNN` looks like this:

```python
# amago/nets/cnn.poy
@gin.configurable
class IMPALAishCNN(CNN):
    def __init__(
        self,
        img_shape: tuple[int],
        channels_first: bool,
        activation: str,
        cnn_block_depths: list[int] = [16, 32, 32],
        post_group_norm: bool = True,
    ):
```
So we can change the `cnn_block_depths` and `post_group_norm` by editing these values, but this would *not* the be place to change the `activation`.
To change these defaults, we could use `.gin` configuration files, but we could also just do this:

```python

from amago.nets.cnn import IMPALAishCNN
from amago.cli_utils import use_config

config = {
  "amago.nets.tstep_encoders.CNNTstepEncoder.cnn_type" : IMPALAishCNN,
  "amago.nets.cnn.IMPALAishCNN.cnn_block_depths" : [32, 64, 64],
}
# changes the default values
use_config(config)

experiment = Experiment(
  tstep_encoder_type=CNNTstepEncoder,
  ...
)
```

</details>

#### Transformer Architecture Example
<details>

As another example, let's say we want to use a `TformerTrajEncoder` with 6 layers of dimension 512, 12 heads, and sliding window attention with a window size of 256.

```python
# amago/nets/traj_encoders.py
@gin.configurable
class TformerTrajEncoder(TrajEncoder):
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 3,
        dropout_ff: float = 0.05,
        dropout_emb: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
        head_scaling: bool = True,
        attention_type: type[transformer.SelfAttention] = transformer.FlashAttention,
    ):


# amago/nets/transformer.py
@gin.configurable
class SlidingWindowFlexAttention(FlexAttention):
    def __init__(
        self,
        causal: bool = True,
        dropout: float = 0.0,
        window_size: int = gin.REQUIRED,
    ):
```

`gin.REQUIRED` is reserved for settings that are not commonly used but would be so important and task-specific that it makes no sense to set a default. You'll get an error if you use one but forget to configure it.

```python
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.nets.transformer import SlidingWindowFlexAttention
from amago.cli_utils import use_config

tformer_config = "amago.nets.traj_encoders.TformerTrajEncoder"
config = {
 f"{tformer_config}.num_heads" : 16,
 f"{tformer_config}.d_model" 512,
 f"{tformer_config}.d_ff" : 2048,
 f"{tformer_config}.attention_type": SlidingWindowFlexAttention,
 "amago.nets.transformer.SlidingWindowFlexAttention.window_size" : 128,
}
use_config(config)
experiment = Experiment(
  traj_encoder_type=TformerTrajEncoder,
  ...
)
```
</details>

#### Exploration

<details>

Explorative action noise is implemented by `gym.Wrapper`s (`amago.envs.exploration`). Env creation automatically wraps the training envs in `Experiment.exploration_wrapper_type`, and these wrappers are `gin.configurable`. One thing to note that is that if the current exploration noise parameter is epsilon_t, the default behavior is for each actor to sample a value in [0, epsilon_t] on each `reset`. In other words the exploration schedule defines the maximum possible value and AMAGO is randomizing over all the settings beneath it to reduce tuning. This can be disabled by `randomize_eps=False`.

```python
from amago.envs.exploration import EpsilonGreedy
from amago.cli_utils import use_config

config = {
  # exploration steps are measured in terms of timesteps *per actor*
  "EpsilonGreedy.steps_anneal" : 200_000.
  "EpsilonGreedy.eps_start" : 1.0,
  "EpsilonGreedy.eps_end" : .01,
  "EpsilonGreedy.randomize_eps" : False,
}
use_config(config)
experiment = Experiment(
  exploration_wrapper_type=EpsilonGreedy,
  ...
)
```
`EpsilonGreedy` is actually the default. The other built-in option is `BilevelEpsilonGreedy`, which is discussed in Appendix A of the paper and is designed for finite-horizon meta-RL problems.

</details>


#### An Easier Way
<details>

Customizing the built-in `TstepEncoder`s, `TrajEncoders`, `Agent`s, and `ExplorationWrapper`s is so common that there's easier ways to do it in `amago.cli_utils`. For example, we could've made the changes for all the previous examples at the same time with:

```python
from amago.cli_utils import switch_traj_encoder, switch_tstep_encoder, switch_agent, switch_exploration, use_config
from amago.nets.transformer import SlidingWindowFlexAttention
from amago.nets.cnn import IMPALAishCNN


config = {
  # these are niche changes customized a level below the `TstepEncoder` / `TrajEncoder`, so we still have to specify them
  "amago.nets.transformer.SlidingWindowFlexAttention.window_size" : 128,
  "amago.nets.cnn.IMPALAishCNN.cnn_block_depths" : [32, 64, 64],
}
tstep_encoder_type = switch_step_encoder(config, arch="cnn", cnn_type=IMPALAishCNN)
traj_encoder_type = switch_traj_encoder(config, arch="transformer", d_model=512, d_ff=2048, num_heads=16, attention_type=SlidingWindowFlexAttention)
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
```


If we want to combine hardcoded changes like these with genuine `.gin` files, `use_config` will take the paths.

```python
# these changes are applied in order from left to right. if we override the same param
# in multiple configs the final one will count. making gin this complicated is a bad idea.
use_config(config, gin_configs=["environment_config.gin", "rl_config.gin"])
```

</details>

<br>

### **7. Start the Experiment and Run Training**

```python
experiment = amago.Experiment(...)
experiment.start()
experiment.learn()
```
We'll see an overview of some high-level settings and progress bars for each learning/collection cycle. Aside from the `wandb` logging metrics, AMAGO outputs data in the following format:

```bash
{Experiment.dset_root}/
    |
    |-- {Experiment.dset_name}/
        |-- train/
        |    # replay buffer of sequence data stored on disk as `*.traj` files.
        |    {environment_name}_{random_id}_{unix_time}.traj
        |    {environment_name}_{another_random_id}_{later_unix_time}.traj
        |    ...
        |
        |-- {Experiment.run_name}/
            |-- config.txt # stores gin configuration details for reproducibility
            |-- policy.pt # the latest model weights
            |-- ckpts/
                    |-- training_states/
                    |    | # full checkpoint dirs used to restore `accelerate` training runs
                    |    |-- {Experiment.run_name}_epoch_0/
                    |    |-- {Experiment.run_name}_epoch_{Experiment.ckpt_interval}/
                    |    |-- ...
                    |
                    |-- policy_weights/
                        | # standard pytorch weight files
                        |-- policy_epoch_0.pt
                        |-- policy_epoch_{Experiment.ckpt_interval}.pt
                        |-- ...
        -- # other runs that share this replay buffer would be listed here
```

#### Offline RL and Replay Across Experiments
As noted in the directory diagram above, the path to the dataset is determined by `dset_root/dset_name`, not by the `run_name`. So we could share the same replay buffer across multiple experiments, initialize the buffer to the result of a previous experiment, or avoid collecting any new data at all (offline RL: `Experiment.start_collecting_at_epoch = float("inf")`).


<br>

### **Train and Learn Asychronously on Multiple GPUs**

<img src="media/amago_big_logo.png" alt="amagologo" width="200" align="right"/>


#### Multi-GPU DistributedDataParallel

<details>

AMAGO can replicate the same (rollout --> learn) on multiple GPUs in `DistributedDataParallel` (DDP) mode. We simplify DDP setup with [`huggingface/accelerate`](https://huggingface.co/docs/accelerate/en/index). To use accelerate, run `accelerate config` and answer the questions. `accelerate` is mainly used for distributed LLM training and many of its features don't apply here. For our purposes, if the answer isn't obvious the answer is "NO", (e.g. "Do you use Megatron-LLM? NO").

Then, to use the GPUs you requested during `accelerate config`, we'd replace a command that noramlly looks like this:

```bash
python my_training_script.py --run_name agi --env CartPole-v1 ...
```

with:

```bash
accelerate launch my_training_script.py --run_name agi --env CartPole-v1 ...
```

And that's it! Let's say our `Experiment.parallel_actors=32`, `Experiment.train_timesteps_per_epoch=1000`, `Experiment.batch_size=32`, and `Experiment.batches_per_epoch=500`. On a single GPU this means we're collecting 32 x 1000 = 32k timesteps per epoch, and training on 500 batches each with 32 sequences. If we decided to use 4 GPUs during `accelerate config`, these same arguments would lead to 4 x 32 x 1000 = 128k timesteps collected per epoch, and we'd still be doing 500 grad updates per epoch with 32 sequences per GPU, but the effective batch size would now be 4 x 32 = 128. Realistically, we're using multiple GPUs to save memory on long sequences and we'd  want to change the batch size to 8 to recover the original batch size of 4 x 8 = 32 while avoiding OOM errors.

> *NOTE*: Validation metrics (`val/` on `wandb`) average over `accelerate` processes, but the `train/` metrics are only logged from the main process (the lowest GPU index) and would have a sample size of a single GPU's batch dim.

</details>

#### Asynchronous Training/Rollouts
<details>

Each `epoch` alternates between rollouts --> gradient updates. AMAGO saves environment data and checkpoints to disk, so changing some `amago.learning.Experiment` kwargs would let these two steps be completely separate. After we create an `experiment = Experiment()`, but before `experiment.start()`, `cli_utils.switch_async_mode` can override settings to `"learn"`, `"collect"` or do `"both"` (the default). This leads to a very hacky but fun way to add extra data collection or do training/learning asychronously. For example, we can `accelerate launch` a multi-gpu script that only does gradient updates, and collect data for that model to train on with as many collect-only processes as we want. All we need to do is make sure the `dset_root`, `dset_name`, `run_name` are the same (so that checkpoints and buffers are being shared), and the network architecture settings are the same (so that checkpoints load correctly). For example:

```python
#  my_training_script.py
from argparse import ArgumentParser()
from amago.cli_utils import switch_async_mode, use_config

parser = ArgumentParser()
parser.add_argument("--mode", options=["learn", "collect", "both"])
args = parser.parse_args()

config = {
  ...
}
use_config(config)


experiment = Experiment(
  dset_root="~/amago_dsets",
  dset_name="agi_training_data",
  run_name="first_big_run",
  tstep_encoder_type=FFTstepEncoder,
  traj_encoder_type=TformerTrajEncoder,
  agent_type=MultiTaskAgent,
  ...
)
switch_async_mode(experiment, args.mode)
experiment.start()
experiment.learn()
```

`accelerate config` a 4-gpu training process on GPU ids 1, 2, 3, 4
Then:
```bash
CUDA_VISIBLE_DEVICES=5 python my_training_script.py --mode collect # on a free GPU
```

```python
accelerate launch my_training_scrip.py --mode train
```

And now we're collecting data on 1 gpu and doing DDP gradient updates on 4 others. At any time during training we could decide to add another `--mode collect` process to boost our framerate. Would that be reproducible? Nope. But this all just kinda works because the AMAGO learning update is way-off-policy (`Agent`) or fully offline (`MultiTaskAgent`). Of course this could be made less hacky by writing one script that starts the collection process, waits until the replay buffer isn't empty, then starts the training process. We are working on some very large training runs and you can expect these features to be much easier to use in the future.
</details>

<br>

## Examples: TODO

To follow most of the examples you'll need to install the benchmark environments with `pip install amago[envs]`.

You can configure the project and account with environment variables:

```bash
export AMAGO_WANDB_PROJECT="wandb project name"
export AMAGO_WANDB_ENTITY="wandb username"
```
For basic single-GPU agents, use the `CUDA_VISIBLE_DEVICES` environment variable to assign learning to a specific GPU index (`CUDA_VISIBLE_DEVICES=7 python train.py ...`).

Environment setup is the main step in applying our agent to a new problem. The example environments are usually *not* included in the scripts themselves but can be found in `amago/envs/builtin/`.

### 1. **Regular MDPs (Classic Gym)**
   
Many popular benchmarks are MDPs and can be treated as a simple special case of the full agent. By turning *off* most of AMAGO's features, we can create a regular off-policy actor-critic like you've seen before. See `examples/01_basic_gym.py` for an example.

 Try `python 01_basic_gym.py --help` for an explanation of hyperparameters and other command line args that are used in most of the examples below.

<details>
<summary> <b>Example Training Commands</b> </summary>
<br>

Train a memory-free policy on built-in gymnasium benchmarks:

```bash
python 01_basic_gym.py --env LunarLander-v2 --horizon 500 --traj_encoder ff --max_seq_len 32 --memory_layers 2 --no_async --run_name <str> --buffer_dir <path>
```
This examples uses a `TrajEncoder` that is just a feedforward network. Training still depends on sequences of `--max_seq_len` timesteps, which is effectively increasing the training batch size.
</details>

<br>

### 2. **POMDPs and Long-Term Memory (POPGym)**
   
Using a memory-equipped `TrajEncoder` creates an effective POMDP solver. AMAGO is efficient enough to use *entire* trajectories as context, and the `TformerTrajEncoder` is a strong default Transformer tuned specifically for stability in RL. See `examples/02_popgym_suite.py` where the same hyperparameters can lead to state-of-the-art performance across the [POPGym](https://arxiv.org/abs/2303.01859) suite.

<details>
<summary> <b>Example Training Commands</b> </summary>
<br>

```bash
python 02_popgym_suite.py --env AutoencodeMedium --parallel_actors 24 --trials 3 --epochs 650 --dset_max_size 80_000 --memory_layers 3 --memory_size 256 --run_name <str> --buffer_dir <path>
```
</details>
 
<br>

### 3. **Fixed-Horizon Meta-RL (Dark-Key-To-Door)**

Meta-RL problems are just POMDPs that automatically reset the task up until a fixed time limit. `TrajEncoder` sequence models let us remember and improve upon past attempts. `examples/03_dark_key_to_door.py` walks through a toy example from the [Algorithm Distillation](https://arxiv.org/abs/2210.14215) paper.

<details>
<summary> <b>Example Training Commands</b> </summary>
<br>

```bash
python 03_dark_key_to_door.py --memory_layers 3 --memory_size 256 --epochs 650 --room_size 9 --episode_length 50 --meta_horizon 500 --run_name <str> --buffer_dir <path>
```
</details>
 
<br>

### 4. **Zero-Shot Adaptation to Goal-Conditioned Environments (Mazerunner)**

`examples/04_mazerunner.py` uses the hindsight instruction relabeling technique from the AMAGO paper on our MazeRunner navigation domain. The ability to relabel rewards in hindsight is a key advantage of off-policy adaptive agents.

<br>

### 5. and 6. **K-Shot Meta-RL (Metaworld and Alchemy)**

`examples/05_kshot_metaworld.py` uses [Metaworld](https://meta-world.github.io) to show how we can setup a meta-RL problem that ends after a certain number of episodes, rather than a fixed horizon `H`. We can let the environment automatically reset itself `k - 1` times while AMAGO pretends it's a zero-shot problem (as long as the resets are added to the observation space). `examples/06_alchemy.py` shows another example on the symbolic version of [DeepMind Alchemy](https://arxiv.org/abs/2102.02926).

<details>
<summary> <b>Example Training Commands</b> </summary>
<br>

Train a transformer policy with a context length of 128 timesteps on 2-shot (`--k 2`) Reach-v2:
```bash
python 05_kshot_metaworld.py --k 2 --benchmark reach-v2 --max_seq_len 128 --epochs 700 --timesteps_per_epoch 1500 --grads_per_epoch 700 --run_name <str> --buffer_dir <path>
```
</details>
 
<br>

### 7. **Goal-Conditioned Open-Worlds (Crafter)**

AMAGO can adapt to procedurally generated environments while completing multi-step instructions. `examples/07_crafter_with_instructions.py` shows how we turn [Crafter](https://danijar.com/project/crafter/) into an instruction-conditioned environment, and then use AMAGO's hindsight relabeling to explore sparse rewards.

<details>
<summary> <b>Example Training Commands</b> </summary>
<br>

Memory-conservative settings with pixel-based observations:

```bash
python 07_crafter_with_instructions.py --max_seq_len 512 --obs_kind crop --start_learning_at_epoch 5 --memory_size 256 --memory_layers 3 --relabel some --epochs 5000 --timesteps_per_epoch 2000 --batch_size 18 --run_name <str> --buffer_dir <path>
```

The command above is a close replication of the pixel-based version (Appendix C5 Table 2). You can watch gameplay of a pretrained checkpoint on user-specified tasks with the `examples/crafter_pixels_demo.ipynb` notebook.
</details>
  
<br>

### 8. **Super Long-Term Memory (Passive T-Maze)**

AMAGO provides a stable way to train long-sequence Transformers with RL, which can turn traditionally hard memory-based environments into simple problems. `examples/08_tmaze.py` adds a few exploration changes to the TMaze environment from [Ni et al., 2023](https://arxiv.org/abs/2307.03864), which lets us recall information for thousands of timesteps.

<details>
<summary> <b>Example Training Commands</b> </summary>
<br>

Example on a horizon of 400 timesteps:
```bash
python 08_tmaze.py --no_async --memory_size 128 --memory_layers 2 --parallel_actors 36 --horizon 400 --timesteps_per_epoch 800  --batch_size 18 --grads_per_epoch 600 --dset_max_size 5000 --run_name <str> --buffer_dir <path>
```
This command with `--horizon 10000 --timesteps_per_epoch 10000` will also train the extreme 10k sequence length mentioned in the paper, although this takes several days to converge due to the inference cost of generating each trajectory.
</details>

<br>

### 9. **Multi-Task Learning (Atari, MetaWorld ML45)**
Switch from the base update (`amago.agent.Agent`) to the "multi-task" update (`amago.agent.MultiTaskAgent`) using `--agent_type multitask`.  `MultiTaskAgent` is better in situations where you are optimizing multiple reward functions. The multitask agent removes actor/critic loss terms that depend on the scale of returns (Q(s, a)) in favor of classification losses that do not. More details in v2 paper coming soon.

**Multi-Game Atari**

Play multiple Atari games simultaneously with short-term memory and a larger [IMPALA](https://arxiv.org/abs/1802.01561) vision encoder.

<details>
<summary> <b>Example Training Commands</b> </summary>

```bash
python 09_ale.py --run_name <str> --buffer_dir <path> --agent_type multitask --parallel_actors 30 --max_seq_len 32 --val_interval 100 --cnn impala --dset_max_size 60_000 --epochs 10_000 --games Pong Boxing Breakout Gopher MsPacman ChopperCommand CrazyClimber BattleZone Qbert Seaquest
```

</details>

**Metaworld ML45**

Learn all 45 [Metaworld](https://meta-world.github.io/) tasks at the same time. Records metrics for each task separately. 

<details>
<summary> <b>Example Training Commands</b> </summary>

```bash
python 05_kshot_metaworld.py --run_name <str> --benchmark ml45 --buffer_dir <path> --parallel_actors 30 --memory_size 320 --timesteps_per_epoch 1501 --agent_type multitask
```

</details>

<br>
<br>

## Multi-GPU Training and Async Rollouts

<img src="media/amago_big_logo.png" alt="amagologo" width="200" align="right"/>

<br>


### Multi-GPU DistributedDataParallel
AMAGO can replicate the same (rollout --> learn) loop of the basic examples on multiple GPUs in `DistributedDataParallel` (DDP) mode. This can improve environment throughput but is mainly intended for distributing the batch dimension of large policies during training. We simplify DDP setup with [`huggingface/accelerate`](https://huggingface.co/docs/accelerate/en/index), which is a popular library for distributed LLM training. To use accelerate, run `accelerate config` and answer the questions. For our purposes, if the answer isn't obvious (e.g. "Do you use Megatron-LLM?"), the answer is always "NO".

Then, run any of the above commands on the GPUs you requested during `accelerate config` by replacing `python <filename> --args` with `accelerate launch <filename> --args`.

> *NOTE*: Validation metrics (average return, success rate) are the only metrics that sync across processes. Everything else is logged only from the main process (the lowest GPU index). This decreases the sample size of training metrics (loss, Q(s, a), etc.), and shows an environment step count (`total_frames`) that is too low.

<br>

### Asynchronous Training/Rollouts
Each `epoch` alternates between rollouts --> gradient updates. AMAGO saves environment data and checkpoints to disk, so changing some `amago.learning.Experiment` kwargs would let these two steps be completely separate. The `examples/` demonstrate a super simple way to run one or more processes of (vectorized parallel) environment interaction alongside training. All you need to do is run the usual command with `--mode collect`. This process only interacts with the environment (including evals), writes trajectories to disk, and reads new parameters from disk. Once that process has finished an epoch or two, run the same command in another terminal (or [`screen`](https://linuxize.com/post/how-to-use-linux-screen/)) with `--mode learn`. This process only loads data from disk and saves fresh checkpoints.

We have used a combination of async updates and multi-gpu training to unlock large-scale RL training (50M+ parameters, 500+ timestep *image* sequences, 1B+ frames) without relying on GPU-accelerated environments.

<br>

## Advanced Configuration

AMAGO is built around [gin-config](https://github.com/google/gin-config), which makes it easy to customize experiments. `gin` makes hyperparameters a default value for an object's `kwargs`, and lets you set their value without editing the source code. You can read more about gin [here](https://github.com/google/gin-config/blob/master/docs/index.md). The `examples/` avoid any `.gin` config files and let you switch between the most important settings without worrying about any of this.



<br>


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

