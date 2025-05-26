"""
Start and launch training runs (main :class:`Experiment`).
"""

import os
import time
import warnings
import contextlib
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Iterable
import math

import gin
from termcolor import colored
import torch
from torch.utils.data import DataLoader
import numpy as np
from einops import repeat
import gymnasium as gym
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import tqdm

from . import utils
from .envs.env_utils import (
    DummyAsyncVectorEnv,
    AlreadyVectorizedEnv,
)
from .envs.exploration import (
    ExplorationWrapper,
    EpsilonGreedy,
)
from .envs import SequenceWrapper, ReturnHistory, SpecialMetricHistory, EnvCreator
from .loading import (
    Batch,
    RLDataset,
    RLData_pad_collate,
    MAGIC_PAD_VAL,
)
from .agent import Agent
from .nets import TstepEncoder, TrajEncoder


@gin.configurable
@dataclass
class Experiment:
    """
    Build, train, and evaluate an :py:class:`~amago.agent.Agent`.

    .. rubric:: Required

    :param run_name: Name of the experiment. Used to create checkpoint and log directories.
    :param ckpt_base_dir: Base directory to store checkpoints and logs. Checkpoints are saved to ``ckpt_base_dir/run_name``.
    :param max_seq_len: Maximum sequence length for training. Determines effective batch size (Batch Size × Sequence Length).
    :param dataset: :py:class:`~amago.loading.RLDataset` for loading training sequences.
    :param tstep_encoder_type: a type of :py:class:`~amago.nets.tstep_encoders.TstepEncoder` (will be created with default kwargs --- edit via gin).
    :param traj_encoder_type: a type of :py:class:`~amago.nets.traj_encoders.TrajEncoder` (will be created with default kwargs --- edit via gin).
    :param agent_type: a type of :py:class:`~amago.agent.Agent` (will be created with default kwargs --- edit via gin).
    :param make_train_env: Callable returning an :py:class:`~amago.envs.amago_env.AMAGOEnv`. If not a list, repeated ``parallel_actors`` times. List gives manual assignment across actors.
    :param make_val_env: Like ``make_train_env``, but only used for evaluation (trajectories never saved).
    :param val_timesteps_per_epoch: Number of steps per parallel environment for evaluation. Determines metric sample size. Should be enough time for at least one episode to finish per actor.

    .. rubric:: Environment

    :param parallel_actors: Number of parallel envs for batched inference. **Default:** 12.
    :param env_mode: ``"async"`` (default), wraps envs in async pool. ``"already_vectorized"`` for jax/gpu batch envs. ``"sync"`` for debug. **Default:** "async".
    :param exploration_wrapper_type: Exploration wrapper for training envs. **Default:** ``EpsilonGreedy``.
    :param sample_actions: Whether to sample from stochastic actor during eval, or take argmax/mean. **Default:** True.
    :param force_reset_train_envs_every: If set, forces call to ``reset`` every N epochs for already_vectorized envs. **Default:** None.
    :param async_env_mp_context: Multiprocessing spawn method for ``AsyncVectorEnv`` (e.g., ``"forkserver"``). Only relevant for ``env_mode="async"``. Set to None for default method. **Default:** None.


    .. rubric:: Logging

    :param log_to_wandb: Enable or disable wandb logging. **Default:** False.
    :param wandb_project: wandb project. **Default:** ``AMAGO_WANDB_PROJECT`` env var.
    :param wandb_entity: wandb entity (username/team). **Default:** ``AMAGO_WANDB_ENTITY`` env var.
    :param wandb_group_name: Group runs on wandb dashboard. **Default:** None.
    :param verbose: Print tqdm bars and info to console. **Default:** True.
    :param log_interval: Log extra metrics every N batches. **Default:** 300.
    :param padded_sampling: Padding for sampling training subsequences. "none", "left", "right", "both". **Default:** "none".
    :param dloader_workers: Number of DataLoader workers for disk loading. Increase for compressed/large trajs.

    .. note::

        The parameters below are only relevant when doing online data collection. They determine
        how parallel environments write finished trajectories to disk. The :py:class:`~amago.loading.DiskTrajDataset`
        reads these files for training.

    :param traj_save_len: Save trajectory on episode end or after this many steps (whichever comes first). Larger values save whole trajectories. **Default:** large value.
    :param has_dset_edit_rights: Turn off for collect-only runs where another process manages the replay buffer. **Default:** True.
    :param stagger_traj_file_lengths: Randomizes file lengths when ``traj_save_len`` is short snippets. **Default:** False.
    :param save_trajs_as: Format for saved trajectories. "npz", "npz-compressed", or "traj". **Default:** "npz".

    .. rubric:: Learning Schedule

    :param epochs: Epochs (each = one data collection + one training round). **Default:** 500.
    :param start_learning_at_epoch: Number of epochs to skip before gradient updates (for replay buffer warmup). **Default:** 0.
    :param start_collecting_at_epoch: Number of epochs to skip data collection (for offline→online finetune or full offline). **Default:** 0.
    :param train_timesteps_per_epoch: Number of steps in each parallel env per epoch. **Default:** 1000.
    :param train_batches_per_epoch: Number of training batches per epoch. **Default:** 1000.
    :param val_interval: How many epochs between evaluation rollouts. **Default:** 20.
    :param ckpt_interval: How many epochs between saving checkpoints. **Default:** 50.
    :param always_save_latest: Whether to always save the latest weights (for distributed usage). **Default:** True.
    :param always_load_latest: Whether to always load the latest weights (for distributed usage). **Default:** False.

    .. rubric:: Optimization

    :param batch_size: Batch size *per GPU* (in sequences). **Default:** 24.
    :param batches_per_update: Number of batches to accumulate gradients over before optimizer update. **Default:** 1.
    :param learning_rate: Optimizer learning rate. **Default:** 1e-4 (defaults to AdamW).
    :param critic_loss_weight: Weight for critic loss vs actor loss in encoders. **Default:** 10.
    :param lr_warmup_steps: Number of warmup steps for learning rate scheduler. **Default:** 500.
    :param grad_clip: Gradient norm clipping value. **Default:** 1.0.
    :param l2_coeff: L2 regularization coefficient (AdamW). **Default:** 1e-3.
    :param mixed_precision: Mixed precision mode for ``accelerate`` ("no", "fp16", "bf16"). **Default:** "no".
    """

    #############
    ## Required ##
    #############
    run_name: str
    ckpt_base_dir: str
    max_seq_len: int
    dataset: RLDataset
    tstep_encoder_type: type[TstepEncoder]
    traj_encoder_type: type[TrajEncoder]
    agent_type: type[Agent]
    val_timesteps_per_epoch: int

    #################
    ## Environment ##
    #################
    make_train_env: callable | Iterable[callable]
    make_val_env: callable | Iterable[callable]
    parallel_actors: int = 12
    env_mode: str = "async"
    async_env_mp_context: Optional[str] = None
    exploration_wrapper_type: Optional[type[ExplorationWrapper]] = EpsilonGreedy
    sample_actions: bool = True
    force_reset_train_envs_every: Optional[int] = None

    #############
    ## Logging ##
    #############
    log_to_wandb: bool = False
    wandb_project: str = os.environ.get("AMAGO_WANDB_PROJECT")
    wandb_entity: str = os.environ.get("AMAGO_WANDB_ENTITY")
    wandb_group_name: str = None
    verbose: bool = True
    log_interval: int = 300

    ############
    ## Replay ##
    ############
    traj_save_len: int = 1e10
    has_dset_edit_rights: bool = True
    stagger_traj_file_lengths: bool = True
    save_trajs_as: str = "npz"
    padded_sampling: str = "none"
    dloader_workers: int = 6

    #######################
    ## Learning Schedule ##
    #######################
    epochs: int = 1000
    start_learning_at_epoch: int = 0
    start_collecting_at_epoch: int = 0
    train_timesteps_per_epoch: int = 1000
    train_batches_per_epoch: int = 1000
    val_interval: Optional[int] = 20
    ckpt_interval: Optional[int] = 50
    always_save_latest: bool = True
    always_load_latest: bool = False

    ##################
    ## Optimization ##
    ##################
    batch_size: int = 24
    batches_per_update: int = 1
    learning_rate: float = 1e-4
    critic_loss_weight: float = 10.0
    lr_warmup_steps: int = 500
    grad_clip: float = 1.0
    l2_coeff: float = 1e-3
    mixed_precision: str = "no"

    def __post_init__(self):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.batches_per_update,
            device_placement=True,
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
            mixed_precision=self.mixed_precision,
        )

    def start(self):
        """Manual initialization after __init__ to give time for gin configuration.

        Call before Experiment.learn()
        """
        self.init_dsets()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("always", category=utils.AmagoWarning)
            env_summary = self.init_envs()
        self.init_dloaders()
        self.init_model()
        self.init_checkpoints()
        self.init_logger()
        if self.verbose:
            self.summary(env_summary=env_summary)

    @property
    def DEVICE(self):
        """Return the device (cpu/gpu) that the experiment is running on."""
        return self.accelerator.device

    def summary(self, env_summary: str) -> None:
        """Print key hparams to the console for reference."""
        total_params = utils.count_params(self.policy)

        assert (
            self.traj_save_len >= self.max_seq_len
        ), "Save longer trajectories than the model can process"

        expl_str = (
            self.exploration_wrapper_type.__name__
            if self.exploration_wrapper_type is not None
            else "None"
        )
        dset_str = "\n\t\t\t".join(self.dataset.get_description().split("\n"))
        self.accelerator.print(
            f"""\n\n \t\t {colored('AMAGO v3.1', 'green')}
            \t -------------------------
            \t Agent: {self.policy.__class__.__name__}
            \t\t Max Sequence Length: {self.max_seq_len}
            \t\t TstepEncoder Type: {self.tstep_encoder_type.__name__}
            \t\t TrajEncoder Type: {self.traj_encoder_type.__name__}
            \t\t Total Parameters: {total_params:,d}
            \t\t Offline Loss Weight: {self.policy.offline_coeff}
            \t\t Online Loss Weight: {self.policy.online_coeff}
            \t\t Mixed Precision: {self.mixed_precision.upper()}
            \t\t Checkpoint Path: {self.ckpt_dir}
            \t Environment:
            \t\t {env_summary}
            \t\t Exploration Type: {expl_str}
            \t Dataset: {dset_str}
            \t Accelerate Processes: {self.accelerator.num_processes} \n\n"""
        )

    def init_envs(self) -> str:
        """Construct parallel training and validation environments.

        Returns:
            str: Description of the environment setup printed to the console when
                Experiment.verbose is True.
        """
        assert self.traj_save_len >= self.max_seq_len

        if self.env_mode in ["async", "sync"]:
            # default environment mode wrapping individual gym environments in a pool of async processes
            # and handling resets by waiting for the termination signal to reach the highest wrapper level
            to_env_list = lambda e: (
                [e] * self.parallel_actors if not isinstance(e, Iterable) else e
            )
            make_val_envs = to_env_list(self.make_val_env)
            make_train_envs = to_env_list(self.make_train_env)
            if not len(make_train_envs) == self.parallel_actors:
                utils.amago_warning(
                    f"`Experiment.parallel_actors` is {self.parallel_actors} but `make_train_env` is a list of length {len(make_train_envs)}"
                )
            if not len(make_val_envs) == self.parallel_actors:
                utils.amago_warning(
                    f"`Experiment.parallel_actors` is {self.parallel_actors} but `make_val_env` is a list of length {len(make_val_envs)}"
                )
            if self.env_mode == "async":
                Par = gym.vector.AsyncVectorEnv
                par_kwargs = dict(context=self.async_env_mp_context)
            else:
                Par = DummyAsyncVectorEnv
                par_kwargs = dict()
        elif self.env_mode == "already_vectorized":
            # alternate environment mode designed for jax / gpu-accelerated envs that handle parallelization
            # with a batch dimension on the lowest wrapper level. These envs must auto-reset and treat the last
            # timestep of a trajectory as the first timestep of the next trajectory.
            make_train_envs = [self.make_train_env]
            make_val_envs = [self.make_val_env]
            Par = AlreadyVectorizedEnv
            par_kwargs = dict()
        else:
            raise ValueError(f"Invalid `env_mode` {self.env_mode}")

        if self.exploration_wrapper_type is not None and not issubclass(
            self.exploration_wrapper_type, ExplorationWrapper
        ):
            utils.amago_warning(
                f"Implement exploration strategies by subclassing `ExplorationWrapper` and setting the `Experiment.exploration_wrapper_type`"
            )

        if self.max_seq_len < self.traj_save_len and self.stagger_traj_file_lengths:
            """
            If the rollout length of the environment is much longer than the `traj_save_len`,
            almost every datapoint will be exactly `traj_save_len` long and spaced `traj_save_len` apart.
            For example if the `traj_save_len` is 100 the trajectory files will all be snippets from
            [0, 100], [100, 200], [200, 300], etc. This can lead to a problem at test-time because the model
            has never seen a sequence from timesteps [50, 150] or [150, 250], etc. We can mitigate this by
            randomizing the trajectory lengths in a range around `traj_save_len`.
            """
            save_every_low = self.traj_save_len - self.max_seq_len
            save_every_high = self.traj_save_len + self.max_seq_len
        else:
            save_every_low = save_every_high = self.traj_save_len

        # wrap environments to save trajectories to replay buffer
        shared_env_kwargs = dict(
            save_trajs_as=self.save_trajs_as,
            save_every_low=save_every_low,
            save_every_high=save_every_high,
        )
        make_train = [
            EnvCreator(
                make_env=env_func,
                # save trajectories to disk
                save_trajs_to=self.dataset.save_new_trajs_to,
                # adds exploration noise
                exploration_wrapper_type=self.exploration_wrapper_type,
                **shared_env_kwargs,
            )
            for env_func in make_train_envs
        ]
        make_val = [
            EnvCreator(
                make_env=env_func,
                # do not save trajectories to disk
                save_trajs_to=None,
                # no exploration noise
                exploration_wrapper_type=None,
                **shared_env_kwargs,
            )
            for env_func in make_val_envs
        ]

        # make parallel envs
        self.train_envs = Par(make_train, **par_kwargs)
        self.val_envs = Par(make_val, **par_kwargs)
        self.train_envs.reset()
        self.rl2_space = make_train[0].rl2_space
        self.hidden_state = None  # holds train_env hidden state between epochs

        if self.env_mode == "already_vectorized":
            _inner = f"Vectorized Gym Env x{self.parallel_actors}"
            _desc = f"{Par.__name__}({_inner})"
        else:
            _inner = "Gym Env"
            _desc = f"{Par.__name__}({_inner} x {self.parallel_actors})"
        return _desc

    def init_checkpoints(self) -> None:
        """Create ckpts/training_states, ckpts/policy_weights, and ckpts/latest dirs"""
        self.ckpt_dir = os.path.join(self.ckpt_base_dir, self.run_name, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "training_states"), exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "policy_weights"), exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "latest"), exist_ok=True)
        self.epoch = 0

    def load_checkpoint_from_path(
        self, path: str, is_accelerate_state: bool = True
    ) -> None:
        """Load a checkpoint from a given path.

        Args:
            path: Full path to the checkpoint fle to load.
            is_accelerate_state: Whether the checkpoint is a full accelerate state (True) or
                pytorch weights only (False). Defaults to True.
        """
        if not is_accelerate_state:
            ckpt = utils.retry_load_checkpoint(path, map_location=self.DEVICE)
            self.policy.load_state_dict(ckpt)
        else:
            self.accelerator.load_state(path)

    def load_checkpoint(self, epoch: int, resume_training_state: bool = True) -> None:
        """Load a historical checkpoint from the `ckpts` directory of this experiment.

        Args:
            epoch: The epoch number of the checkpoint to load.
            resume_training_state: Whether to resume the entire training process (True) or only
                the policy weights (False). Defaults to True.
        """
        if not resume_training_state:
            path = os.path.join(
                self.ckpt_dir, "policy_weights", f"policy_epoch_{epoch}.pt"
            )
            self.load_checkpoint_from_path(path, is_accelerate_state=False)
        else:
            ckpt_name = f"{self.run_name}_epoch_{epoch}"
            ckpt_path = os.path.join(self.ckpt_dir, "training_states", ckpt_name)
            self.load_checkpoint_from_path(ckpt_path, is_accelerate_state=True)
        self.epoch = epoch

    def save_checkpoint(self) -> None:
        """Save both the training state and the policy weights to the ckpt_dir."""
        ckpt_name = f"{self.run_name}_epoch_{self.epoch}"
        self.accelerator.save_state(
            os.path.join(self.ckpt_dir, "training_states", ckpt_name),
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            # create backup of raw weights unrelated to the more complex process of resuming an accelerate state
            torch.save(
                self.policy.state_dict(),
                os.path.join(
                    self.ckpt_dir, "policy_weights", f"policy_epoch_{self.epoch}.pt"
                ),
            )

    def write_latest_policy(self) -> None:
        """Write absolute latest policy to a hardcoded location used by `read_latest_policy`"""
        ckpt_name = os.path.join(self.ckpt_dir, "latest", "policy.pt")
        torch.save(self.policy.state_dict(), ckpt_name)

    def read_latest_policy(self) -> None:
        """Read the latest policy -- used to communicate weight updates between
        learning/collecting processes"""
        ckpt_name = os.path.join(self.ckpt_dir, "latest", "policy.pt")
        ckpt = utils.retry_load_checkpoint(ckpt_name, map_location=self.DEVICE)
        if ckpt is not None:
            self.accelerator.print("Loading latest policy....")
            self.policy.load_state_dict(ckpt)
        else:
            utils.amago_warning("Latest policy checkpoint was not loaded.")

    def delete_buffer_from_disk(self) -> None:
        """Clear the replay buffer from disk (mainly for `examples/`).

        Calls `self.dataset.delete()` if the current process is the main process.
        """
        if self.accelerator.is_main_process:
            self.dataset.delete()

    def init_dsets(self) -> RLDataset:
        """Modifies the provided RLDataset (in place) to use important info configured by the
        experiment."""
        self.dataset.configure_from_experiment(self)
        return self.dataset

    def init_dloaders(self) -> DataLoader:
        """Create pytorch dataloaders to batch trajectories in parallel."""
        train_dloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        self.train_dloader = self.accelerator.prepare(train_dloader)
        return self.train_dloader

    def init_logger(self) -> None:
        """Configure log dir and wandb compatibility."""
        gin_config = gin.operative_config_str()
        config_path = os.path.join(self.ckpt_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
            # records the gin config on the wandb dashboard
            gin_as_wandb = utils.gin_as_wandb_config()
            log_dir = os.path.join(self.ckpt_base_dir, self.run_name, "wandb_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                config=gin_as_wandb,
                init_kwargs={
                    "wandb": dict(
                        entity=self.wandb_entity,
                        dir=log_dir,
                        name=self.run_name,
                        group=self.wandb_group_name,
                    )
                },
            )

    def init_optimizer(self, policy: Agent) -> torch.optim.Optimizer:
        """Defines the optimizer.

        Override to switch from AdamW.

        Returns:
            torch.optim.Optimizer in charge of updating the Agent's parameters
                (Agent.trainable_params)
        """
        adamw_kwargs = dict(lr=self.learning_rate, weight_decay=self.l2_coeff)
        return torch.optim.AdamW(policy.trainable_params, **adamw_kwargs)

    def init_model(self) -> None:
        """Build an initial policy based on observation shapes"""
        policy_kwargs = {
            "tstep_encoder_type": self.tstep_encoder_type,
            "traj_encoder_type": self.traj_encoder_type,
            "obs_space": self.rl2_space["obs"],
            "rl2_space": self.rl2_space["rl2"],
            "action_space": self.train_envs.single_action_space,
            "max_seq_len": self.max_seq_len,
        }
        policy = self.agent_type(**policy_kwargs)
        assert isinstance(policy, Agent)
        optimizer = self.init_optimizer(policy)
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            policy, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0

    @property
    def policy(self) -> Agent:
        """Returns the current Agent policy free from the accelerator wrapper."""
        return self.accelerator.unwrap_model(self.policy_aclr)

    def interact(
        self,
        envs,
        timesteps: int,
        hidden_state=None,
        render: bool = False,
        save_on_done: bool = False,
        episodes: Optional[int] = None,
    ) -> tuple[ReturnHistory, SpecialMetricHistory]:
        """Main policy loop for interacting with the environment.

        Args:
            envs: The (parallel) environments to interact with.
            timesteps: The number of timesteps to interact with each environment.

        Keyword Args:
            hidden_state: The hidden state of the policy. If None, a fresh hidden state is
                initialized. Defaults to None.
            render: Whether to render the environment. Defaults to False.
            save_on_done: If True, save completed trajectory sequences to disk as soon as they
                are finished. If False, wait until all rollouts are completed. Only applicable
                if the provided envs are configured to save rollouts to disk. Defaults to False.
            episodes: The number of episodes to interact with the environment. If provided, the
                loop will terminate after this many episodes have been completed OR we hit the
                `timesteps` limit, whichever comes first. Defaults to None.

        Returns:
            tuple[ReturnHistory, SpecialMetricHistory]: Objects that keep track of standard
                eval stats (average returns) and any additional eval metrics the envs have been
                configured to record.
        """
        policy = self.policy
        policy.eval()

        if self.verbose:
            iter_ = tqdm(
                range(timesteps),
                desc="Env Interaction",
                total=timesteps,
                leave=False,
                colour="yellow",
            )
        else:
            iter_ = range(timesteps)

        # clear results statistics
        # (can make train-time stats useless depending on horizon vs. `timesteps`)
        utils.call_async_env(envs, "reset_stats")

        if hidden_state is None:
            # init new hidden state
            hidden_state = policy.traj_encoder.init_hidden_state(
                self.parallel_actors, self.DEVICE
            )

        def get_t():
            # fetch `Timestep.make_sequence` from all envs
            par_obs_rl2_time = utils.call_async_env(envs, "current_timestep")
            _obs, _rl2s, _time_idxs = [], [], []
            for _o, _r, _t in par_obs_rl2_time:
                _obs.append(_o)
                _rl2s.append(_r)
                _time_idxs.append(_t)
            # stack all the results
            _obs = utils.stack_list_array_dicts(_obs, axis=0, cat=True)
            _rl2s = np.concatenate(_rl2s, axis=0)
            _time_idxs = np.concatenate(_time_idxs, axis=0)
            # ---> torch --> GPU --> dummy length dim
            _obs = {
                k: torch.from_numpy(v).to(self.DEVICE).unsqueeze(1)
                for k, v in _obs.items()
            }
            _rl2s = torch.from_numpy(_rl2s).to(self.DEVICE).unsqueeze(1)
            _time_idxs = torch.from_numpy(_time_idxs).to(self.DEVICE).unsqueeze(1)
            return _obs, _rl2s, _time_idxs

        obs, rl2s, time_idxs = get_t()
        episodes_finished = 0
        for _ in iter_:
            with torch.no_grad():
                with self.caster():
                    actions, hidden_state = policy.get_actions(
                        obs=obs,
                        rl2s=rl2s,
                        time_idxs=time_idxs,
                        sample=self.sample_actions,
                        hidden_state=hidden_state,
                    )
            *_, terminated, truncated, _ = envs.step(actions.squeeze(1).cpu().numpy())
            done = terminated | truncated
            if done.ndim == 2:
                done = done.squeeze(1)
            if done.any():
                if save_on_done:
                    utils.call_async_env(envs, "save_finished_trajs")
                episodes_finished += done.sum()
            obs, rl2s, time_idxs = get_t()
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)
            if render:
                envs.render()
            if episodes is not None and episodes_finished >= episodes:
                break

        return_history = utils.call_async_env(envs, "return_history")
        special_history = utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history)

    def collect_new_training_data(self) -> None:
        """Generate train_timesteps_per_epoch * parallel_actors timesteps of new environment
        interaction that will be saved to the replay buffer when the rollouts finishes.
        """
        if (
            self.force_reset_train_envs_every is not None
            and self.epoch % self.force_reset_train_envs_every == 0
        ):
            self.train_envs.reset()
            self.hidden_state = None
        self.hidden_state, (returns, specials) = self.interact(
            self.train_envs,
            self.train_timesteps_per_epoch,
            hidden_state=self.hidden_state,
        )
        utils.call_async_env(self.train_envs, "save_finished_trajs")

    def evaluate_val(self) -> None:
        """Evaluate the current policy without exploration noise on the validation environments,
        and averages the performance metrics across `accelerate` processes."""
        # reset envs first
        self.val_envs.reset()
        start_time = time.time()
        # interact from a blank hidden state that is discarded
        _, (returns, specials) = self.interact(
            self.val_envs,
            self.val_timesteps_per_epoch,
            hidden_state=None,
        )
        end_time = time.time()
        fps = (
            self.val_timesteps_per_epoch
            * self.parallel_actors
            / (end_time - start_time)
        )
        logs_per_process = self.policy_metrics(returns, specials=specials)
        logs_per_process["Env FPS (per_process)"] = fps
        # validation metrics are averaged over all processes
        logs_global = utils.avg_over_accelerate(logs_per_process)
        if self.verbose:
            cur_return = logs_global["Average Total Return (Across All Env Names)"]
            self.accelerator.print(f"Average Return : {cur_return}")
            self.accelerator.print(
                f"Env FPS : {fps * self.accelerator.num_processes:.2f}"
            )
        self.log(logs_global, key="val")

    def evaluate_test(
        self,
        make_test_env: callable | Iterable[callable],
        timesteps: int,
        render: bool = False,
        save_trajs_to: Optional[str] = None,
        episodes: Optional[int] = None,
    ) -> dict[str, float]:
        """One-off evaluation of a new environment callable for testing.

        Args:
            make_test_env: A callable or iterable of callables that make and return a test
                environment. If an iterable, it must be of length `Experiment.parallel_actors`.
            timesteps: The number of timesteps to interact with each environment.
            render: Whether to render the environment. Defaults to False.
            save_trajs_to: The directory to save trajectories. Useful when using evaluate_test to
                gather demonstration data for another run. If None, no data is saved. Defaults to
                None.
            episodes: The number of episodes to interact with the environment. If provided, the
                loop will terminate after this many episodes have been completed OR we hit the
                `timesteps` limit, whichever comes first. Defaults to None.

        Returns:
            dict[str, float]: A dictionary of evaluation metrics.
        """
        is_saving = save_trajs_to is not None

        def wrap(m):
            return lambda: SequenceWrapper(
                m(),
                save_trajs_to=save_trajs_to,
                save_every=None,
            )

        if self.env_mode == "already_vectorized":
            Par = AlreadyVectorizedEnv
            env_list = [wrap(make_test_env)]
        else:
            if isinstance(make_test_env, Iterable):
                assert len(make_test_env) == self.parallel_actors
                env_list = [wrap(f) for f in make_test_env]
            else:
                env_list = [wrap(make_test_env) for _ in range(self.parallel_actors)]
            if self.env_mode == "async":
                Par = gym.vector.AsyncVectorEnv
            elif self.env_mode == "sync":
                Par = DummyAsyncVectorEnv
        test_envs = Par(env_list)
        test_envs.reset()
        _, (returns, specials) = self.interact(
            test_envs,
            timesteps,
            hidden_state=None,
            render=render,
            # saves trajectories as soon as they're finished instead of waiting until the end of eval
            save_on_done=is_saving,
            episodes=episodes,
        )
        logs = self.policy_metrics(returns, specials)
        logs_global = utils.avg_over_accelerate(logs)
        self.log(logs_global, key="test")
        test_envs.close()
        if self.verbose:
            cur_return = logs_global["Average Total Return (Across All Env Names)"]
            self.accelerator.print(f"Test Average Return : {cur_return}")
        return logs

    def x_axis_metrics(self) -> dict[str, int | float]:
        """Get current x-axis metrics for wandb."""
        metrics = {}
        if hasattr(self, "train_envs"):
            # overall total frames per process
            total_frames = sum(utils.call_async_env(self.train_envs, "total_frames"))
            # total frames by env_name per process
            frames_by_env_name = utils.call_async_env(
                self.train_envs, "total_frames_by_env_name"
            )
            total_frames_by_env_name = defaultdict(int)
            for env_frames in frames_by_env_name:
                for env_name, frames in env_frames.items():
                    total_frames_by_env_name[f"total_frames-{env_name}"] += frames
            total_frames_by_env_name = dict(total_frames_by_env_name)
            total_frames_by_env_name["total_frames"] = total_frames
            # sum over processes
            total_frames_global = utils.sum_over_accelerate(total_frames_by_env_name)
            metrics.update(total_frames_global)

        # add epoch
        metrics["Epoch"] = self.epoch
        metrics["gradient_steps"] = self.grad_update_counter
        return metrics

    def log(
        self, metrics_dict: dict[str, torch.Tensor | int | float], key: str
    ) -> None:
        """Log a dict of metrics to the `key` panel of the wandb console alongisde current
        x-axis metrics."""
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().float().item()
            else:
                log_dict[k] = v

        if self.log_to_wandb:
            wandb_dict = {
                f"{key}/{subkey}": val for subkey, val in log_dict.items()
            } | self.x_axis_metrics()
            self.accelerator.log(wandb_dict)

    def policy_metrics(
        self,
        returns: Iterable[ReturnHistory],
        specials: Iterable[SpecialMetricHistory],
    ) -> dict:
        """Gather policy performance metrics across parallel environments.

        Args:
            returns: The return history logger froms the environments.
            specials: The special metrics history loggers from the environments.

        Returns:
            dict: A dictionary of policy performance metrics.
        """
        returns_by_env_name = defaultdict(list)
        specials_by_env_name = defaultdict(lambda: defaultdict(list))

        for ret, spe in zip(returns, specials):
            for env_name, scores in ret.data.items():
                returns_by_env_name[env_name].extend(scores)
            for env_name, specials_dict in spe.data.items():
                for special_key, special_val in specials_dict.items():
                    specials_by_env_name[env_name][special_key].extend(special_val)

        avg_ret_per_env = {
            f"Average Total Return in {name}": np.array(scores).mean()
            for name, scores in returns_by_env_name.items()
        }
        bottom_quintile_ret_per_env = {
            f"Bottom Quintile Total Return in {name}": np.percentile(scores, 20)
            for name, scores in returns_by_env_name.items()
        }
        avg_special_per_env = {
            f"Average {special_key} in {name}": np.array(special_vals).mean()
            for name, specials_dict in specials_by_env_name.items()
            for special_key, special_vals in specials_dict.items()
        }
        avg_return_overall = {
            "Average Total Return (Across All Env Names)": np.array(
                list(avg_ret_per_env.values())
            ).mean()
        }
        return (
            avg_ret_per_env
            | avg_return_overall
            | avg_special_per_env
            | bottom_quintile_ret_per_env
        )

    def edit_actor_mask(
        self, batch: Batch, actor_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        """Customize the actor loss mask.

        Args:
            batch: The batch of data.
            actor_loss: The unmasked actor loss. Shape: (Batch, Length, Num Gammas, 1)
            pad_mask: The default mask. True where the sequence was not padded out of the
                dataloader.

        Returns:
            The mask. True where the actor loss should count, False where it should be ignored.
        """
        return pad_mask

    def edit_critic_mask(
        self, batch: Batch, critic_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        """Customize the critic loss mask.

        Args:
            batch: The batch of data.
            critic_loss: The unmasked critic loss. Shape: (Batch, Length, Num Critics, Num
                Gammas, 1)
            pad_mask: The default mask. True where the sequence was not padded out of the
                dataloader.

        Returns:
            The mask. True where the critic loss should count, False where it should be ignored.
        """
        return pad_mask

    def compute_loss(self, batch: Batch, log_step: bool) -> dict:
        """Core computation of the actor and critic RL loss terms from a `Batch` of data.

        Args:
            batch: The batch of data.
            log_step: Whether to compute extra metrics for wandb logging.

        Returns:
            dict: loss terms and any logging metrics. "Actor Loss", "Critic Loss", "Sequence
                Length", "Batch Size (in Timesteps)", "Unmasked Batch Size (in Timesteps)" are
                always provided. Additional keys are determined by what is logged in the
                Agent.forward method.
        """
        # Agent.forward handles most of the work
        critic_loss, actor_loss = self.policy_aclr(batch, log_step=log_step)
        update_info = self.policy.update_info
        B, L_1, G, _ = actor_loss.shape
        C = len(self.policy.critics)

        # mask sequence losses
        state_mask = (~((batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True))).bool()
        critic_state_mask = repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {C} {G} 1")
        actor_state_mask = repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {G} 1")
        # hook to allow custom masks
        actor_state_mask = self.edit_actor_mask(batch, actor_loss, actor_state_mask)
        critic_state_mask = self.edit_critic_mask(batch, critic_loss, critic_state_mask)
        batch_size = B * L_1
        unmasked_batch_size = actor_state_mask[..., 0, 0].sum()
        masked_actor_loss = utils.masked_avg(actor_loss, actor_state_mask)
        if isinstance(critic_loss, torch.Tensor):
            masked_critic_loss = utils.masked_avg(critic_loss, critic_state_mask)
        else:
            assert critic_loss is None
            masked_critic_loss = 0.0

        # all of this is logged
        return {
            "Critic Loss": masked_critic_loss,
            "Actor Loss": masked_actor_loss,
            "Sequence Length": L_1 + 1,
            "Batch Size (in Timesteps)": batch_size,
            "Unmasked Batch Size (in Timesteps)": unmasked_batch_size,
        } | update_info

    def _get_grad_norms(self):
        """Get gradient norms for logging."""
        ggn = utils.get_grad_norm
        pi = self.policy
        grads = {
            "Actor Grad Norm": ggn(pi.actor),
            "Critic Grad Norm": ggn(pi.critics),
            "TrajEncoder Grad Norm": ggn(pi.traj_encoder),
            "TstepEncoder Grad Norm": ggn(pi.tstep_encoder),
        }
        return grads

    def train_step(self, batch: Batch, log_step: bool):
        """Take a single training step on a `batch` of data.

        Args:
            batch: The batch of data.
            log_step: Whether to compute extra metrics for wandb logging.

        Returns:
            dict: metrics from the compute_loss method.
        """
        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            l = self.compute_loss(batch, log_step=log_step)
            loss = l["Actor Loss"] + self.critic_loss_weight * l["Critic Loss"]
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.policy_aclr.parameters(), self.grad_clip
                )
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    l.update(
                        {"Learning Rate": self.lr_schedule.get_last_lr()[0]}
                        | self._get_grad_norms()
                    )
            self.optimizer.step()
            self.lr_schedule.step()
        return l

    def caster(self):
        """Get the context manager for mixed precision training."""
        if self.mixed_precision != "no":
            return torch.autocast(device_type="cuda")
        else:
            return contextlib.suppress()

    def learn(self) -> None:
        """Main training loop for the experiment.

        For every epoch, we:
            1. Load the latest policy checkpoint if `always_load_latest` is True.
            2. Evaluate the policy on the validation set if `val_interval` is not None and the
                current epoch is divisible by `val_interval`.
            3. Collect new training data if `train_timesteps_per_epoch` is not None and the
                current epoch >= to `start_collecting_at_epoch`.
            4. Train the policy on the training data for `train_batches_per_epoch` batches if
                `self.dataset.ready_for_training` is True.
            5. Save the policy checkpoint if `ckpt_interval` is not None and the current epoch
                is divisible by `ckpt_interval`.
            6. Write the latest policy checkpoint if `always_save_latest` is True.

        Experiment be configured so that processes do some or all of the above. For example, an
        actor process might only do steps 1, 2, and 3, while a learner process might only do
        steps 4, 5, and 6.
        """

        def make_pbar(loader, epoch_num):
            if self.verbose:
                return tqdm(
                    enumerate(loader),
                    desc=f"{self.run_name} Epoch {epoch_num} Train",
                    total=self.train_batches_per_epoch,
                    colour="green",
                )
            else:
                return enumerate(loader)

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            if self.always_load_latest:
                self.read_latest_policy()

            # environment interaction
            self.policy_aclr.eval()
            if (
                self.val_interval
                and epoch % self.val_interval == 0
                and self.val_timesteps_per_epoch > 0
            ):
                self.evaluate_val()
            if (
                epoch >= self.start_collecting_at_epoch
                and self.train_timesteps_per_epoch > 0
            ):
                self.collect_new_training_data()
            self.accelerator.wait_for_everyone()

            dset_log = self.dataset.on_end_of_collection(experiment=self)
            self.log(dset_log, key="dataset")
            self.init_dloaders()
            if not self.dataset.ready_for_training:
                utils.amago_warning(
                    f"Skipping training on epoch {epoch} because `dataset.ready_for_training` is False"
                )
                continue

            # training
            elif epoch < self.start_learning_at_epoch:
                continue
            if self.train_batches_per_epoch > 0:
                self.policy_aclr.train()
                for train_step, batch in make_pbar(self.train_dloader, epoch):
                    total_step = (epoch * self.train_batches_per_epoch) + train_step
                    log_step = total_step % self.log_interval == 0
                    loss_dict = self.train_step(batch, log_step=log_step)
                    if log_step:
                        self.log(loss_dict, key="train-update")
            self.accelerator.wait_for_everyone()
            del self.train_dloader

            # end epoch
            self.epoch = epoch
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
            if self.always_save_latest:
                self.write_latest_policy()
