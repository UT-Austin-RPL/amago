import os
import time
import warnings
import contextlib
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Iterable

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
    TrajDset,
    RLData_pad_collate,
    MAGIC_PAD_VAL,
)
from .hindsight import Relabeler
from .agent import Agent
from .nets import TstepEncoder, TrajEncoder


@gin.configurable
@dataclass
class Experiment:
    #############
    ## General ##
    #############
    # the name of the experiment. used to create a directory in `dset_root` to store checkpoints and logs. used for logging to wandb.
    run_name: str
    # the most important hyperparameter: the maximum sequence length that the model will be trained on.
    max_seq_len: int
    # trajectories are saved to disk on `terminated or truncated` or after this many steps have passed since the last save (whichever comes first)
    traj_save_len: int
    # TstepEncoder is created by calling this with default kwargs (use gin)
    tstep_encoder_type: type[TstepEncoder]
    # TrajEncoder is created by calling this with default kwargs (use gin)
    traj_encoder_type: type[TrajEncoder]
    # Agent is created by calling this with default kwargs (use gin)
    agent_type: type[Agent]

    #################
    ## Environment ##
    #################
    # a function that takes no args and returns an AMAGOEnv. If this isn't a list, it will be repeated `parallel_actors` times.
    # You can create the list yourself if you want to manually assign different envs across the parallel actors.
    make_train_env: callable | Iterable[callable]
    # same as `make_train_env`, but these environments are only used for evaluation and their trajectories are not saved to disk.
    make_val_env: callable | Iterable[callable]
    # spawns multiple envs in parallel to speed up data collection with batched inference.
    parallel_actors: int = 10
    # two main options: "async" and "already_vectorized".
    # "async" is the default and wraps individual gym environments in a pool of async processes using `gymnasium.vector.AsyncVectorEnv`.
    # "already_vectorized" is an alternate mode designed for jax / gpu-accelerated envs that handle parallelization with a batch dimension on the lowest wrapper level.
    # "sync" is the same as "async" but doesn't actually run the environment in parallel, which is helpful for debugging or when the env is so fast that this overhead isn't worth it.
    env_mode: str = "async"
    # exploration is implemented with a gym wrapper that is only applied to training environments.
    exploration_wrapper_type: Optional[type[ExplorationWrapper]] = EpsilonGreedy
    # whether to sample from the stochastic actor during eval, or take the argmax action.
    sample_actions: bool = True
    # a safety measure that forces a call to `reset` every _ epochs for cases when `reset` is otherwise never called (already_vectorized).
    force_reset_train_envs_every: Optional[int] = None

    #############
    ## Logging ##
    #############
    # enable/disable wandb logging
    log_to_wandb: bool = False
    # your wandb project
    wandb_project: str = os.environ.get("AMAGO_WANDB_PROJECT")
    # your wandb entity (username or team name)
    wandb_entity: str = os.environ.get("AMAGO_WANDB_ENTITY")
    # group different runs on the wandb dashboard
    wandb_group_name: str = None
    # prints tqdm progress bars and some high-level info to the console
    verbose: bool = True
    # how many batches between forward/backward passes that spend time computing extra metrics for wandb logging.
    log_interval: int = 300

    ############
    ## Replay ##
    ############
    # path to the root directory where your datasets (and checkpoints) are stored
    dset_root: str = None
    # stores all the trajectories in `dset_root/dset_name/buffer`. You can use this to reuse the same dataset across multiple experiments.
    dset_name: str = None
    # turn this off for collect-only runs where we need to assume the replay buffer is being managed by another learner process.
    has_replay_buffer_rights: bool = True
    # maximum number of .traj files to keep in the replay buffer before we start deleting the oldest files
    dset_max_size: int = 15_000
    # an object that can edit trajectory data before it is sent to the agent. useful for hindsight relabeling (temporarily removed) and data augmentation.
    relabel_type: type[Relabeler] = Relabeler
    # randomizes trajectory file lengths when saving snippets from a much longer rollout. please refer to a longer explanation in `amago.Experiment.init_envs`.
    stagger_traj_file_lengths: bool = True
    # how to save trajectory .traj files. three options:
    # "npz" saves data as numpy arrays.
    # "npz-compressed" trades time for disk space by compressing large files.
    # "traj" pickles the full `Trajectory` object.
    save_trajs_as: str = "npz"
    # number of workers to use for the DataLoader that loads trajectories from disk. increase when using npz-compressed or when loading very long trajs from pixel envs.
    dloader_workers: int = 6

    #######################
    ## Learning Schedule ##
    #######################
    # each epoch has one round of data collection and one round of training.
    epochs: int = 1000
    # skip the first _ epochs before beginning gradient updates. can be used to imitate the "replay buffer warmup" common to most off-policy impelmentations.
    start_learning_at_epoch: int = 0
    # skip the first _ epochs of data collection. can be used for offline --> online finetuning or to avoid online interaction entirely.
    start_collecting_at_epoch: int = 0
    # how many `steps` to take in each parallel environment each epoch.
    train_timesteps_per_epoch: int = 1000
    # how many batches to load from disk for training each epoch. gradient updates per epoch is train_batches_per_epoch // batches_per_update
    train_batches_per_epoch: int = 1000
    #  how many epochs to wait between evaluation rollouts
    val_interval: Optional[int] = 10
    # how many `steps to take in each parallel environment for evaluation. determines the sample size of the evaluation metrics.
    val_timesteps_per_epoch: int = 10_000
    # how many epochs to wait between saving checkpoints
    ckpt_interval: Optional[int] = 20
    # always_save_latest and always_load_latest are used to communicate the latest policy weights between multiple processes.
    # A learning-only thread would always_save, while an actor-only thread would always_load.
    always_save_latest: bool = True
    always_load_latest: bool = False

    ##################
    ## Optimization ##
    ##################
    # training batch size *per gpu*
    batch_size: int = 24
    # number of batches to accumulate gradients over before updating the model
    batches_per_update: int = 1
    # learning rate for the optimizer
    learning_rate: float = 1e-4
    # coefficient that balances the actor and critic loss for the TstepEncoder and TrajEncoder -- which optimize both. The actor loss weight is fixed to 1.
    critic_loss_weight: float = 10.0
    # linear warmup steps for the learning rate scheduler
    lr_warmup_steps: int = 500
    # gradient clipping (by norm)
    grad_clip: float = 1.0
    # l2 regularization coefficient (note that we default to AdamW)
    l2_coeff: float = 1e-3
    # mixed precision mode. this is passed directly to `accelerate` and follows its options ("no", "fp16", "bf16").
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
        """
        Manually initialization after __init__ to give time for gin configuration.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("always", category=utils.AmagoWarning)
            env_summary = self.init_envs()
        self.init_dsets()
        self.init_dloaders()
        self.init_model()
        self.init_checkpoints()
        self.init_logger()
        if self.verbose:
            self.summary(env_summary=env_summary)

    @property
    def DEVICE(self):
        return self.accelerator.device

    def summary(self, env_summary: str):
        """
        Print key hparams to the console for reference.
        """
        total_params = utils.count_params(self.policy)

        assert (
            self.traj_save_len >= self.max_seq_len
        ), "Save longer trajectories than the model can process"

        expl_str = (
            self.exploration_wrapper_type.__name__
            if self.exploration_wrapper_type is not None
            else "None"
        )
        self.accelerator.print(
            f"""\n\n \t\t {colored('AMAGO v3', 'green')}
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
            \t Replay Buffer:
            \t\t Buffer Path: {os.path.join(self.dset_root, self.dset_name, "buffer")}
            \t\t FIFO Buffer Max Size: {self.dset_max_size}
            \t\t FIFO Buffer Initial Size: {self.train_dset.count_deletable_trajectories()}
            \t\t Protected Buffer Initial Size: {self.train_dset.count_protected_trajectories()}
            \t\t Trajectory File Max Sequence Length: {self.traj_save_len}
            \t Accelerate Processes: {self.accelerator.num_processes} \n\n"""
        )

    def init_envs(self):
        """
        Construct parallel training and validation environments.
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
            Par = (
                gym.vector.AsyncVectorEnv
                if self.env_mode == "async"
                else DummyAsyncVectorEnv
            )
        elif self.env_mode == "already_vectorized":
            # alternate environment mode designed for jax / gpu-accelerated envs that handle parallelization
            # with a batch dimension on the lowest wrapper level. These envs must auto-reset and treat the last
            # timestep of a trajectory as the first timestep of the next trajectory.
            make_train_envs = [self.make_train_env]
            make_val_envs = [self.make_val_env]
            Par = AlreadyVectorizedEnv
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
            utils.amago_warning(
                f"Note: Partial Context Mode. Randomizing trajectory file lengths in [{save_every_low}, {save_every_high}]"
            )
        else:
            save_every_low = save_every_high = self.traj_save_len

        # wrap environments to save trajectories to replay buffer
        shared_env_kwargs = dict(
            dset_root=self.dset_root,
            dset_name=self.dset_name,
            save_trajs_as=self.save_trajs_as,
            save_every_low=save_every_low,
            save_every_high=save_every_high,
        )
        make_train = [
            EnvCreator(
                make_env=env_func,
                # save trajectories to disk
                make_dset=True,
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
                make_dset=False,
                # no exploration noise
                exploration_wrapper_type=None,
                **shared_env_kwargs,
            )
            for env_func in make_val_envs
        ]

        # make parallel envs
        self.train_envs = Par(make_train)
        self.val_envs = Par(make_val)
        self.train_envs.reset()
        self.rl2_space = make_train[0].rl2_space
        self.hidden_state = None  # holds train_env hidden state between rollouts

        if self.env_mode == "already_vectorized":
            _inner = f"Vectorized Gym Env x{self.parallel_actors}"
            _desc = f"{Par.__name__}({_inner})"
        else:
            _inner = "Gym Env"
            _desc = f"{Par.__name__}({_inner} x {self.parallel_actors})"
        return _desc

    def init_checkpoints(self):
        """
        Create ckpts/training_states and ckpts/policy_weights dirs
        """
        self.ckpt_dir = os.path.join(
            self.dset_root, self.dset_name, self.run_name, "ckpts"
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "training_states"), exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "policy_weights"), exist_ok=True)
        self.epoch = 0

    def load_checkpoint(self, epoch: int, resume_training_state: bool = True):
        """
        Load a historical checkpoint from the `ckpts` directory of this experiment.

        `resume_training_state = True` perfectly resumes the entire training process
        (optimizer, grad scaler, etc.), but will probably only work on the exact same
        accelerate configuration.

        `resume_training_state = False` only loads the policy weights.
        """
        if not resume_training_state:
            # load the weights without worrrying about resuming the accelerate state
            ckpt = utils.retry_load_checkpoint(
                os.path.join(
                    self.ckpt_dir, "policy_weights", f"policy_epoch_{epoch}.pt"
                ),
                map_location=self.DEVICE,
            )
            self.policy_aclr.load_state_dict(ckpt)
        else:
            # loads weights and will set the epoch but otherwise resets training
            # (optimizer, grad scaler, etc.)
            ckpt_name = f"{self.run_name}_epoch_{epoch}"
            ckpt_path = os.path.join(self.ckpt_dir, "training_states", ckpt_name)
            self.accelerator.load_state(ckpt_path)
        self.epoch = epoch

    def save_checkpoint(self):
        """
        Save both the training state and the policy weights to the ckpt_dir.
        """
        ckpt_name = f"{self.run_name}_epoch_{self.epoch}"
        self.accelerator.save_state(
            os.path.join(self.ckpt_dir, "training_states", ckpt_name),
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            # create backup of raw weights unrelated to the more complex process of resuming an accelerate state
            weights_only = torch.save(
                self.policy_aclr.state_dict(),
                os.path.join(
                    self.ckpt_dir, "policy_weights", f"policy_epoch_{self.epoch}.pt"
                ),
            )

    def write_latest_policy(self):
        """
        Write absolute latest policy to a hardcoded location used by `read_latest_policy`
        """
        ckpt_name = os.path.join(
            self.dset_root, self.dset_name, self.run_name, "policy.pt"
        )
        torch.save(self.policy.state_dict(), ckpt_name)

    def read_latest_policy(self):
        """
        Read the latest policy -- used to communicate weight updates between learning/collecting processes
        """
        ckpt_name = os.path.join(
            self.dset_root, self.dset_name, self.run_name, "policy.pt"
        )
        ckpt = utils.retry_load_checkpoint(ckpt_name, map_location=self.DEVICE)
        if ckpt is not None:
            self.accelerator.print("Loading latest policy....")
            self.policy.load_state_dict(ckpt)
        else:
            utils.amago_warning("Latest policy checkpoint was not loaded.")

    def delete_buffer_from_disk(self, delete_protected: bool = False):
        """
        Clear the replay buffer from disk (mainly for `examples/`).
        """
        if self.accelerator.is_main_process:
            self.train_dset.clear(delete_protected=delete_protected)

    def init_dsets(self):
        """
        Create a pytorch dataset to load trajectories from disk.
        """
        self.train_dset = TrajDset(
            relabeler=self.relabel_type(),
            dset_root=self.dset_root,
            dset_name=self.dset_name,
            items_per_epoch=self.train_batches_per_epoch
            * self.batch_size
            * self.accelerator.num_processes,
            max_seq_len=self.max_seq_len,
        )
        return self.train_dset

    def init_dloaders(self):
        """
        Create pytorch dataloaders to batch trajectories in parallel.
        """
        train_dloader = DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        self.train_dloader = self.accelerator.prepare(train_dloader)
        return self.train_dloader

    def init_logger(self):
        """
        Configure log dir and wandb compatibility.
        """
        gin_config = gin.operative_config_str()
        config_path = os.path.join(
            self.dset_root, self.dset_name, self.run_name, "config.txt"
        )
        with open(config_path, "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
            # records the gin config on the wandb dashboard
            gin_as_wandb = utils.gin_as_wandb_config()
            log_dir = os.path.join(self.dset_root, "wandb_logs")
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

    def init_optimizer(self, params):
        """
        Override to switch from AdamW
        """
        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.l2_coeff,
        )

    def init_model(self):
        """
        Build an initial policy based on observation shapes
        """
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
        optimizer = self.init_optimizer(policy.trainable_params)
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            policy, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0

    @property
    def policy(self):
        return self.accelerator.unwrap_model(self.policy_aclr)

    def interact(
        self,
        envs,
        timesteps: int,
        hidden_state=None,
        render: bool = False,
    ) -> tuple[ReturnHistory, SpecialMetricHistory]:
        """
        Main policy loop for interacting with the environment.
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
            obs, rl2s, time_idxs = get_t()
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)
            if render:
                envs.render()

        return_history = utils.call_async_env(envs, "return_history")
        special_history = utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history)

    def collect_new_training_data(self):
        """
        Generates train_timesteps_per_epoch * parallel_actors timesteps of new environment interaction that
        will be saved to the replay buffer when the rollouts finishes.
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

    def evaluate_val(self):
        """
        Evaluates the current policy without exploration noise on the validation environments, and averages
        the performance metrics across `accelerate` processes.
        """
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
        self, make_test_env: callable, timesteps: int, render: bool = False
    ):
        """
        One-off evaluation of a new environment callable for testing.
        """
        make = lambda: SequenceWrapper(
            make_test_env(), save_every=None, make_dset=False
        )
        if self.env_mode == "already_vectorized":
            Par = AlreadyVectorizedEnv
            env_list = [make]
        elif self.env_mode == "async":
            Par = gym.vector.AsyncVectorEnv
            env_list = [make for _ in range(self.parallel_actors)]
        elif self.env_mode == "sync":
            Par = DummyAsyncVectorEnv
            env_list = [make for _ in range(self.parallel_actors)]
        test_envs = Par(env_list)
        test_envs.reset()
        _, (returns, specials) = self.interact(
            test_envs,
            timesteps,
            hidden_state=None,
            render=render,
        )
        logs = self.policy_metrics(returns, specials)
        logs_global = utils.avg_over_accelerate(logs)
        self.log(logs_global, key="test")
        test_envs.close()
        if self.verbose:
            cur_return = logs_global["Average Total Return (Across All Env Names)"]
            self.accelerator.print(f"Test Average Return : {cur_return}")
        return logs

    def x_axis_metrics(self):
        """
        Accumulate x-axis metrics for plotting (total frames by environment name and the current epoch)
        across accelerate processes.
        """
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
        # add epoch
        total_frames_global["Epoch"] = self.epoch
        total_frames_global["gradient_steps"] = self.grad_update_counter
        return total_frames_global

    def log(self, metrics_dict, key):
        """
        Log a dict of metrics to the `key` panel of the wandb console, and record the current epoch and total frames.
        """
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
        returns: ReturnHistory,
        specials: SpecialMetricHistory,
    ) -> dict:
        """
        Gather policy performance metrics across parallel environments and then average over accelerate processes.
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

    def compute_loss(self, batch: Batch, log_step: bool):
        """
        Core computation of the actor and critic RL loss terms from a `Batch` of data.
        """
        critic_loss, actor_loss = self.policy_aclr(batch, log_step=log_step)
        update_info = self.policy.update_info
        B, L_1, G, _ = actor_loss.shape
        C = len(self.policy.critics)
        state_mask = (~((batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True))).float()
        critic_state_mask = repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {C} {G} 1")
        actor_state_mask = repeat(state_mask[:, :-1, ...], f"B L 1 -> B L {G} 1")

        masked_actor_loss = utils.masked_avg(actor_loss, actor_state_mask)
        if isinstance(critic_loss, torch.Tensor):
            masked_critic_loss = utils.masked_avg(critic_loss, critic_state_mask)
        else:
            assert critic_loss is None
            masked_critic_loss = 0.0

        return {
            "critic_loss": masked_critic_loss,
            "actor_loss": masked_actor_loss,
            "seq_len": L_1 + 1,
            "mask": state_mask,
        } | update_info

    def _get_grad_norms(self):
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
        """
        Take a single training step on a `batch` of data.

        `log_step = True` computes (many) extra metrics for wandb logging.
        """
        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            l = self.compute_loss(batch, log_step=log_step)
            loss = l["actor_loss"] + self.critic_loss_weight * l["critic_loss"]
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.policy_aclr.parameters(), self.grad_clip
                )
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    l.update(self._get_grad_norms())
            self.optimizer.step()
            self.lr_schedule.step()
        return l

    def caster(self):
        if self.mixed_precision != "no":
            return torch.autocast(device_type="cuda")
        else:
            return contextlib.suppress()

    def manage_replay_buffer(self):
        """
        Find new trajectory files saved to disk and delete old ones to imitate a fixed-size replay buffer.
        Also logs buffer stats to wandb.
        """
        self.train_dset.refresh_files()
        if not self.has_replay_buffer_rights:
            return
        old_size = self.train_dset.count_trajectories()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # the main process edits its file list during the filter but this is currently
            # wasted effort because the other processes don't see the changes...
            self.train_dset.filter(new_size=self.dset_max_size)
        self.accelerator.wait_for_everyone()
        self.train_dset.refresh_files()  # ... so check the current files again
        dset_size = self.train_dset.count_trajectories()
        fifo_size = self.train_dset.count_deletable_trajectories()
        protected_size = self.train_dset.count_protected_trajectories()
        self.log(
            {
                "Trajectory Files Saved in FIFO Replay Buffer": fifo_size,
                "Trajectory Files Saved in Protected Replay Buffer": protected_size,
                "Total Trajectory Files in Replay Buffer": dset_size,
                "Trajectory Files Deleted": old_size - dset_size,
                "Buffer Disk Space (GB)": self.train_dset.disk_usage,
            },
            key="buffer",
        )

    def learn(self):
        """
        Main training loop for the experiment.
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

            self.manage_replay_buffer()
            self.init_dloaders()
            if self.train_dset.count_trajectories() == 0:
                utils.amago_warning(
                    f"Skipping epoch {epoch} because no training trajectories have been saved yet..."
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

            # end epoch
            self.epoch = epoch
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
            if self.always_save_latest:
                self.write_latest_policy()
