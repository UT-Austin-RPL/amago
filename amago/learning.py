import os
import warnings
import contextlib
from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from typing import Optional

import gin
import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
from einops import repeat
import gymnasium as gym
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import tqdm, release_memory

from . import utils
from .agent import Agent
from amago.envs.env_utils import (
    ReturnHistory,
    SuccessHistory,
    ExplorationWrapper,
    SequenceWrapper,
    GPUSequenceBuffer,
    DummyAsyncVectorEnv,
    MakeEnvSaveToDisk,
)
from .loading import Batch, TrajDset, RLData_pad_collate, MAGIC_PAD_VAL
from .hindsight import Relabeler, RelabelWarning


@gin.configurable
@dataclass
class Experiment:
    # General
    run_name: str
    max_seq_len: int
    traj_save_len: int
    agent_type: type[Agent]

    # Environment
    make_train_env: callable
    make_val_env: callable
    parallel_actors: int = 10
    async_envs: bool = True
    exploration_wrapper_Cls: Optional[type[ExplorationWrapper]] = ExplorationWrapper
    sample_actions: bool = True

    # Logging
    log_to_wandb: bool = False
    wandb_project: str = os.environ.get("AMAGO_WANDB_PROJECT")
    wandb_entity: str = os.environ.get("AMAGO_WANDB_ENTITY")
    wandb_group_name: str = None
    verbose: bool = True

    # Replay
    dset_root: str = None
    dset_name: str = None
    dset_max_size: int = 15_000
    dset_filter_pct: float = 0.1
    relabel: str = "none"
    goal_importance_sampling: bool = False
    stagger_traj_file_lengths: bool = True
    save_trajs_as: str = "npz"

    # Learning Schedule
    epochs: int = 1000
    start_learning_at_epoch: int = 0
    start_collecting_at_epoch: int = 0
    train_timesteps_per_epoch: int = 1000
    train_grad_updates_per_epoch: int = 1000
    val_interval: int = 10
    val_timesteps_per_epoch: int = 10_000
    val_checks_per_epoch: int = 50
    log_interval: int = 250
    ckpt_interval: int = 20

    # Optimization
    batch_size: int = 24
    batches_per_update: int = 1
    dloader_workers: int = 6
    learning_rate: float = 1e-4
    critic_loss_weight: float = 10.0
    lr_warmup_steps: int = 500
    grad_clip: float = 1.0
    l2_coeff: float = 1e-3
    fast_inference: bool = True
    mixed_precision: str = "no"

    def start(self):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.batches_per_update,
            device_placement=True,
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
            mixed_precision=self.mixed_precision,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.init_envs()
        self.init_dsets()
        self.init_dloaders()
        self.init_model()
        self.init_checkpoints()
        self.init_logger()
        if self.verbose:
            self.summary()

    @property
    def DEVICE(self):
        return self.accelerator.device

    def summary(self):
        total_params = utils.count_params(self.policy)

        assert (
            self.traj_save_len >= self.max_seq_len
        ), "Save longer trajectories than the model can process"

        if self.horizon <= self.max_seq_len and self.horizon <= self.traj_save_len:
            mode = "Maximum Context (Perfect Meta-RL / Long-Term Memory)"
        elif self.horizon > self.max_seq_len and self.horizon <= self.traj_save_len:
            mode = "Fixed Context with Valid Relabeling (Approximate Meta-RL / POMDPs)"
        elif self.horizon > self.max_seq_len and self.horizon > self.traj_save_len:
            mode = (
                "Fixed Context with Invalid Relabeling (Approximate Meta-RL / POMDPs)"
            )

        self.accelerator.print(
            f"""\n\n \t\t AMAGO (v2)
            \t -------------------------
            \t Agent Type: {self.policy.__class__.__name__}
            \t Environment Horizon: {self.horizon}
            \t Policy Max Sequence Length: {self.max_seq_len}
            \t Trajectory File Sequence Length: {self.traj_save_len}
            \t Mode: {mode}
            \t Mixed Precision: {self.mixed_precision.upper()}
            \t Fast Inference: {self.fast_inference}
            \t Total Parameters: {total_params:,d} \n\n"""
        )

    def init_envs(self):
        assert self.traj_save_len >= self.max_seq_len
        shared_env_kwargs = dict(
            dset_root=self.dset_root,
            dset_name=self.dset_name,
            save_trajs_as=self.save_trajs_as,
            traj_save_len=self.traj_save_len,
            max_seq_len=self.max_seq_len,
            stagger_traj_file_lengths=self.stagger_traj_file_lengths,
        )
        make_train = MakeEnvSaveToDisk(
            make_env=self.make_train_env,
            dset_split="train",
            exploration_wrapper_Cls=self.exploration_wrapper_Cls,
            **shared_env_kwargs,
        )
        make_val = MakeEnvSaveToDisk(
            make_env=self.make_val_env,
            dset_split="val",
            exploration_wrapper_Cls=None,
            **shared_env_kwargs,
        )
        Par = gym.vector.AsyncVectorEnv if self.async_envs else DummyAsyncVectorEnv
        self.train_envs = Par([make_train for _ in range(self.parallel_actors)])
        self.val_envs = Par([make_val for _ in range(self.parallel_actors)])
        self.gcrl2_space = make_train.gcrl2_space
        self.horizon = make_train.horizon
        # self.train_buffers holds the env state between rollout cycles
        # that are shorter than the horizon length
        self.train_envs.reset()
        self.val_envs.reset()
        self.train_buffers = None
        self.hidden_state = None

    def init_checkpoints(self):
        self.ckpt_dir = os.path.join(self.dset_root, self.dset_name, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.epoch = 0

    def load_checkpoint(self, epoch: int):
        ckpt_name = f"{self.run_name}_epoch_{epoch}"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        self.accelerator.load_state(ckpt_path)
        self.epoch = epoch

    def save_checkpoint(self):
        ckpt_name = f"{self.run_name}_epoch_{self.epoch}"
        self.accelerator.save_state(os.path.join(self.ckpt_dir, ckpt_name))

    def write_latest_policy(self):
        ckpt_name = os.path.join(self.dset_root, self.dset_name, "policy.pt")
        torch.save(self.policy_aclr.state_dict(), ckpt_name)

    def read_latest_policy(self):
        ckpt_name = os.path.join(self.dset_root, self.dset_name, "policy.pt")
        ckpt = utils.retry_load_checkpoint(ckpt_name, map_location=self.DEVICE)
        if ckpt is not None:
            self.policy_aclr.load_state_dict(ckpt)
        else:
            warnings.warn("Latest policy checkpoint was not loaded.")

    def init_dsets(self):
        if self.save_trajs_as != "trajectory" and self.relabel != "none":
            warnings.warn(
                "Saving data in efficient ('frozen') format... these files will be skipped by the Relabeler",
                category=RelabelWarning,
            )
        warnings.filterwarnings("ignore", category=RelabelWarning)
        self.train_dset = TrajDset(
            relabeler=Relabeler(self.relabel, self.goal_importance_sampling),
            dset_root=self.dset_root,
            dset_name=self.dset_name,
            dset_split="train",
            items_per_epoch=self.train_grad_updates_per_epoch * self.batch_size,
            max_seq_len=self.max_seq_len,
        )
        self.val_dset = TrajDset(
            relabeler=Relabeler(self.relabel, self.goal_importance_sampling),
            dset_root=self.dset_root,
            dset_name=self.dset_name,
            dset_split="val",
            items_per_epoch=self.val_checks_per_epoch * self.batch_size,
            max_seq_len=self.max_seq_len,
        )

    def init_dloaders(self):
        self.train_dset.refresh_files()
        self.val_dset.refresh_files()
        train_dloader = DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        val_dloader = DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        # accelerator.free_memory clears everything.
        # https://github.com/huggingface/accelerate/blob/2471eacdd648446f2f63eccb9b18a7731304f401/src/accelerate/accelerator.py#L3189
        self.train_dloader, self.val_dloader = self.accelerator.prepare(
            train_dloader, val_dloader
        )

    def init_logger(self):
        gin_config = gin.operative_config_str()
        config_path = os.path.join(self.dset_root, self.dset_name, "config.txt")
        with open(config_path, "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
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

    def init_model(self):
        policy_kwargs = {
            "obs_space": self.gcrl2_space["obs"],
            "goal_space": self.gcrl2_space["goal"],
            "rl2_space": self.gcrl2_space["rl2"],
            "action_space": self.train_envs.single_action_space,
            "max_seq_len": self.max_seq_len,
            "horizon": self.horizon,
        }
        policy = self.agent_type(**policy_kwargs)
        assert isinstance(policy, Agent)
        optimizer = torch.optim.AdamW(
            policy.trainable_params,
            lr=self.learning_rate,
            weight_decay=self.l2_coeff,
        )
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            policy, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)

    @property
    def policy(self):
        return self.accelerator.unwrap_model(self.policy_aclr)

    def interact(
        self,
        envs,
        timesteps: int,
        buffers=None,
        hidden_state=None,
        render: bool = False,
    ) -> tuple[ReturnHistory, SuccessHistory]:
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

        if buffers is None:
            # start of training or new eval cycle
            envs.reset()
            make_buffer = partial(
                GPUSequenceBuffer, self.DEVICE, self.max_seq_len, self.parallel_actors
            )
            obs_seqs = make_buffer()
            goal_seqs = make_buffer()
            rl2_seqs = make_buffer()
        else:
            # continue interaction from previous epoch
            obs_seqs, goal_seqs, rl2_seqs = buffers

        if hidden_state is None:
            # init new hidden state
            hidden_state = policy.traj_encoder.init_hidden_state(
                self.parallel_actors, self.DEVICE
            )

        def get_t(_dones=None):
            par_obs_goal_rl2 = utils.call_async_env(envs, "current_timestep")
            _obs = utils.stack_list_array_dicts(
                [obs_goal_rl2[0] for obs_goal_rl2 in par_obs_goal_rl2], axis=0
            )
            _goal = np.stack(
                [obs_goal_rl2[1] for obs_goal_rl2 in par_obs_goal_rl2], axis=0
            )
            _rl2 = np.stack(
                [obs_goal_rl2[2] for obs_goal_rl2 in par_obs_goal_rl2], axis=0
            )
            obs_seqs.add_timestep(_obs, _dones)
            goal_seqs.add_timestep(_goal, _dones)
            rl2_seqs.add_timestep(_rl2, _dones)

        if buffers is None:
            get_t()

        for _ in iter_:
            obs_tc_t = obs_seqs.sequences
            goals_tc_t = goal_seqs.sequences["_"]
            rl2_tc_t = rl2_seqs.sequences["_"]
            seq_lengths = obs_seqs.sequence_lengths
            time_idxs = obs_seqs.time_idxs

            with torch.no_grad():
                with self.caster():
                    actions, hidden_state = policy.get_actions(
                        obs=obs_tc_t,
                        goals=goals_tc_t,
                        rl2s=rl2_tc_t,
                        seq_lengths=seq_lengths,
                        time_idxs=time_idxs,
                        sample=self.sample_actions,
                        hidden_state=hidden_state if self.fast_inference else None,
                    )
            *_, terminated, truncated, _ = envs.step(actions)
            done = terminated | truncated
            get_t(done)
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)

            if render:
                envs.render()

        return_history = utils.call_async_env(envs, "return_history")
        success_history = utils.call_async_env(envs, "success_history")
        return (
            (obs_seqs, goal_seqs, rl2_seqs),
            hidden_state,
            return_history,
            success_history,
        )

    def collect_new_training_data(self):
        if self.train_timesteps_per_epoch > 0:
            self.train_buffers, self.hidden_state, returns, successes = self.interact(
                self.train_envs,
                self.train_timesteps_per_epoch,
                buffers=self.train_buffers,
            )

    def evaluate_val(self):
        if self.val_timesteps_per_epoch > 0:
            *_, returns, successes = self.interact(
                self.val_envs,
                self.val_timesteps_per_epoch,
            )
            logs = self.policy_metrics(returns, successes)
            cur_return = logs["Average Total Return (Across All Env Names)"]
            if self.verbose:
                self.accelerator.print(f"Average Return : {cur_return}")
            self.log(logs, key="val")

    def evaluate_test(
        self, make_test_env: callable, timesteps: int, render: bool = False
    ):
        make = lambda: SequenceWrapper(
            make_test_env(), save_every=None, make_dset=False
        )
        Par = gym.vector.AsyncVectorEnv if self.async_envs else DummyAsyncVectorEnv
        test_envs = Par([make for _ in range(self.parallel_actors)])
        *_, returns, successes = self.interact(
            test_envs,
            timesteps,
            render=render,
        )
        logs = self.policy_metrics(returns, successes)
        self.log(logs, key="test")
        test_envs.close()
        return logs

    def log(self, metrics_dict, key):
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().float().item()
            else:
                log_dict[k] = v

        total_frames = sum(utils.call_async_env(self.train_envs, "total_frames"))
        frames_by_env_name = utils.call_async_env(
            self.train_envs, "total_frames_by_env_name"
        )
        total_frames_by_env_name = defaultdict(int)
        for env_frames in frames_by_env_name:
            for env_name, frames in env_frames.items():
                total_frames_by_env_name[f"total_frames-{env_name}"] += frames
        if self.log_to_wandb:
            progress = {
                "epoch": self.epoch,
                "total_frames": total_frames,
            }
            self.accelerator.log(
                {f"{key}/{subkey}": val for subkey, val in log_dict.items()}
                | progress
                | dict(total_frames_by_env_name)
            )

    def make_figures(self, loss_info) -> dict[str, wandb.Image]:
        """
        Override this to create polished figures from raw logging
        info and automatically dump them to wandb.
        """
        return {}

    def policy_metrics(self, returns: ReturnHistory, successes: SuccessHistory):
        return_by_env_name = {}
        success_by_env_name = {}
        for ret, suc in zip(returns, successes):
            for env_name, scores in ret.data.items():
                if env_name in return_by_env_name:
                    return_by_env_name[env_name] += scores
                else:
                    return_by_env_name[env_name] = scores
            for env_name, scores in suc.data.items():
                if env_name in success_by_env_name:
                    success_by_env_name[env_name] += scores
                else:
                    success_by_env_name[env_name] = scores

        avg_ret_per_env = {
            f"Average Total Return in {name}": np.array(scores).mean()
            for name, scores in return_by_env_name.items()
        }
        avg_suc_per_env = {
            f"Average Success Rate in {name}": np.array(scores).mean()
            for name, scores in success_by_env_name.items()
        }
        avg_return_overall = {
            "Average Total Return (Across All Env Names)": np.array(
                list(avg_ret_per_env.values())
            ).mean()
        }
        return avg_ret_per_env | avg_suc_per_env | avg_return_overall

    def compute_loss(self, batch: Batch, log_step: bool):
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
            "TstepEncoder Goal Emb. Grad Norm": ggn(pi.tstep_encoder.goal_emb),
        }
        return grads

    def train_step(self, batch: Batch, log_step: bool):
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

    def val_step(self, batch):
        with torch.no_grad():
            return self.compute_loss(batch, log_step=True)

    def learn(self):
        def make_pbar(loader, training, epoch):
            if training:
                desc = f"{self.run_name} Epoch {epoch} Train"
                steps = self.train_grad_updates_per_epoch
                c = "green"
            else:
                desc = f"{self.run_name} Epoch {epoch} Val"
                steps = self.val_checks_per_epoch
                c = "red"

            if self.verbose:
                return tqdm(enumerate(loader), desc=desc, total=steps, colour=c)
            else:
                return enumerate(loader)

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            # environment interaction
            self.policy_aclr.eval()
            if epoch % self.val_interval == 0:
                self.evaluate_val()
            if epoch >= self.start_collecting_at_epoch:
                self.collect_new_training_data()

            self.accelerator.wait_for_everyone()
            # make dataloaders aware of new .traj files
            self.init_dloaders()
            if self.train_dset.count_trajectories() == 0:
                warnings.warn(
                    f"Skipping epoch {epoch} because no training trajectories have been saved yet...",
                    category=Warning,
                )
                continue

            # training
            elif epoch < self.start_learning_at_epoch:
                continue
            self.policy_aclr.train()
            for train_step, batch in make_pbar(self.train_dloader, True, epoch):
                total_step = (epoch * self.train_grad_updates_per_epoch) + train_step
                log_step = total_step % self.log_interval == 0
                loss_dict = self.train_step(batch, log_step=log_step)
                if log_step:
                    self.log(loss_dict, key="train-update")
            self.accelerator.wait_for_everyone()

            # validation
            if (
                epoch % self.val_interval == 0
                and self.val_dset.count_trajectories() > 0
            ):
                self.policy_aclr.eval()
                for _, batch in make_pbar(self.val_dloader, False, epoch):
                    loss_dict = self.val_step(batch)
                    self.log(loss_dict, key="val-update")
                figures = self.make_figures(loss_dict)
                self.log(figures, key="val-update")
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.val_dset.clear()

            # buffer management
            dset_size = self.train_dset.count_trajectories()
            dset_gb = self.train_dset.disk_usage
            needs_filter = (
                dset_size > self.dset_max_size and self.dset_filter_pct is not None
            )
            self.accelerator.wait_for_everyone()
            if needs_filter and self.accelerator.is_main_process:
                self.train_dset.filter(self.dset_filter_pct)
            self.log(
                {
                    "Trajectory Files Saved in Replay Buffer": dset_size,
                    "Train Buffer Disk Space (GB)": dset_gb,
                },
                key="buffer",
            )
            self.accelerator.wait_for_everyone()

            # end epoch
            self.epoch = epoch
            if epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
