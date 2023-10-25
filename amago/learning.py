import os
import warnings
import contextlib
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from tqdm import tqdm
from einops import repeat
import gymnasium as gym
import gin

from . import utils
from .agent import Agent
from amago.envs.env_utils import (
    ReturnHistory,
    SuccessHistory,
    ExplorationWrapper,
    SequenceWrapper,
    GPUSequenceBuffer,
)
from .loading import Batch, TrajDset, RLData_pad_collate, MAGIC_PAD_VAL
from .hindsight import Relabeler, RelabelWarning


@gin.configurable
@dataclass
class Experiment:
    # General
    make_train_env: Callable
    make_val_env: Callable
    parallel_actors: int
    max_seq_len: int
    traj_save_len: int
    run_name: str
    gpu: int

    # Logging
    log_to_wandb: bool = False
    wandb_project: str = os.environ.get("AMAGO_WANDB_PROJECT")
    wandb_entity: str = os.environ.get("AMAGO_WANDB_ENTITY")
    wandb_group_name: str = None
    wandb_log_dir: str = None
    verbose: bool = True

    # Replay
    dset_root: str = None
    dset_name: str = None
    dset_max_size: int = 15_000
    dset_filter_pct: float = 0.1
    relabel: str = "none"
    goal_importance_sampling: bool = False
    stagger_traj_file_lengths: bool = True

    # Learning Schedule
    epochs: int = 1000
    start_learning_after_epoch: int = 0
    train_timesteps_per_epoch: int = 1000
    train_grad_updates_per_epoch: int = 1000
    val_interval: int = 10
    val_timesteps_per_epoch: int = 10_000
    val_checks_per_epoch: int = 50
    ckpt_interval: int = 20
    log_interval: int = 250

    # Optimization
    batch_size: int = 24
    dloader_workers: int = 8
    init_learning_rate: float = 1e-4
    critic_loss_weight: float = 10.0
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    l2_coeff: float = 1e-3
    half_precision: bool = False
    fast_inference: bool = True

    # Exploration
    exploration_wrapper_Cls: Callable | None = ExplorationWrapper
    sample_actions: bool = True

    def start(self):
        self.DEVICE = torch.device(f"cuda:{self.gpu}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.init_envs()
        self.init_dsets()
        self.init_dloaders()
        self.init_model()
        self.init_optimizer()
        self.init_checkpoints()
        self.init_logger()
        if self.verbose:
            self.summary()

    def summary(self):
        total_params = 0
        for name, parameter in self.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params

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

        print(
            f"""\n\n \t\t AMAGO
        \t -------------------------
        \t Environment Horizon: {self.horizon}
        \t Policy Max Sequence Length: {self.max_seq_len}
        \t Trajectory File Sequence Length: {self.traj_save_len}
        \t Mode: {mode}
        \t Half Precision: {self.half_precision}
        \t Fast Inference: {self.fast_inference}
        \t Total Parameters: {total_params:,d} \n\n"""
        )

    def init_envs(self):
        assert self.traj_save_len >= self.max_seq_len
        if self.max_seq_len < self.traj_save_len and self.stagger_traj_file_lengths:
            # staggered traj file lengths fix a potential bug where, for example,
            # a policy with a max_seq_len of 10 trained on trajectory files
            # of length 100 has never seen the sequence of [t=95,...,t=105]
            # during training.
            save_every_low = self.traj_save_len - self.max_seq_len
            save_every_high = self.traj_save_len + self.max_seq_len
            if self.verbose:
                print(
                    f"Note: Partial Context Mode. Randomizing trajectory file lengths in [{save_every_low}, {save_every_high}]"
                )
        else:
            save_every_low = save_every_high = self.traj_save_len

        self.horizon = -float("inf")

        def _make_env(fn, split):
            env = fn()
            self.horizon = max(self.horizon, env.horizon)
            if split == "train" and self.exploration_wrapper_Cls:
                # exploration is handled by a wrapper
                env = self.exploration_wrapper_Cls(env)
            # SequenceWrapper handles all the disk writing
            env = SequenceWrapper(
                env,
                save_every=(save_every_low, save_every_high),
                make_dset=True,
                dset_root=self.dset_root,
                dset_name=self.dset_name,
                dset_split=split,
            )
            # save gcrl2 space here to make model later
            self.gcrl2_space = env.gcrl2_space
            return env

        make_train_env = partial(_make_env, self.make_train_env, "train")
        make_val_env = partial(_make_env, self.make_val_env, "val")
        self.train_envs = gym.vector.AsyncVectorEnv(
            [make_train_env for _ in range(self.parallel_actors)]
        )
        self.train_envs.reset()
        self.val_envs = gym.vector.AsyncVectorEnv(
            [make_val_env for _ in range(self.parallel_actors)]
        )
        self.val_envs.reset()
        self.discrete = isinstance(
            self.train_envs.single_action_space, gym.spaces.Discrete
        )
        if not self.discrete:
            assert (self.train_envs.single_action_space.low >= -1).all()
            assert (self.train_envs.single_action_space.high <= 1.0).all()
        # self.train_buffers holds the env state between rollout cycles
        # that are shorter than the horizon length
        self.train_buffers = None
        self.hidden_state = None

    def init_checkpoints(self):
        self.best_return = -float("inf")
        self.ckpt_dir = os.path.join(self.dset_root, self.dset_name, "ckpts")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.epoch = 0

    def load_checkpoint(self, epoch: int = None, loading_best: bool = False):
        assert epoch is not None or loading_best is True
        if epoch is not None:
            assert not loading_best
        if not loading_best:
            ckpt_name = f"{self.run_name}_epoch_{epoch}.pt"
        else:
            ckpt_name = f"{self.run_name}_BEST.pt"

        ckpt = torch.load(os.path.join(self.ckpt_dir, ckpt_name))
        self.policy.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]
        self.grad_scaler.load_state_dict(ckpt["grad_scaler"])
        self.best_return = ckpt["best_return"]

    def save_checkpoint(self, saving_best: bool = False):
        state_dict = {
            "model_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "grad_scaler": self.grad_scaler.state_dict(),
            "best_return": self.best_return,
        }
        if not saving_best:
            ckpt_name = f"{self.run_name}_epoch_{self.epoch}.pt"
        else:
            ckpt_name = f"{self.run_name}_BEST.pt"
        torch.save(state_dict, os.path.join(self.ckpt_dir, ckpt_name))

    def init_dsets(self):
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

        self.train_dloader = DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        self.val_dloader = DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )

    def init_logger(self):
        utils.init_plt()
        gin_config = gin.operative_config_str()
        config_path = os.path.join(self.dset_root, self.dset_name, "config.txt")
        with open(config_path, "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
            log_dir = self.wandb_log_dir or os.path.join(self.dset_root, "wandb_logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
                group=self.wandb_group_name,
                reinit=True,
            )
            wandb.save(config_path)

    def init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_params,
            lr=self.init_learning_rate,
            weight_decay=self.l2_coeff,
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.train_grad_updates_per_epoch * self.warmup_epochs,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.half_precision,
        )

    def init_model(self):
        obs_shape = self.gcrl2_space["obs"].shape
        goal_shape = self.gcrl2_space["goal"].shape
        rl2_shape = self.gcrl2_space["rl2"].shape
        if self.discrete:
            action_dim = self.train_envs.single_action_space.n
        else:
            action_dim = self.train_envs.single_action_space.shape[-1]
        policy_kwargs = {
            "obs_shape": obs_shape,
            "goal_shape": goal_shape,
            "rl2_shape": rl2_shape,
            "action_dim": action_dim,
            "max_seq_len": self.max_seq_len,
            "horizon": self.horizon,
            "discrete": self.discrete,
        }
        self.policy = Agent(**policy_kwargs)
        self.policy.to(self.DEVICE)

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

        self.policy.eval()

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
        envs.call_async("reset_stats")
        envs.call_wait()

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
            hidden_state = self.policy.traj_encoder.init_hidden_state(
                self.parallel_actors, self.DEVICE
            )

        def get_t(_dones=None):
            envs.call_async("current_timestep")
            par_obs_goal_rl2 = envs.call_wait()
            _obs = np.stack(
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

        for step in iter_:
            obs_tc_t = obs_seqs.sequences
            goals_tc_t = goal_seqs.sequences
            rl2_tc_t = rl2_seqs.sequences
            seq_lengths = obs_seqs.sequence_lengths
            time_idxs = obs_seqs.time_idxs

            with torch.no_grad():
                with self.caster():
                    actions, hidden_state = self.policy.get_actions(
                        obs=obs_tc_t,
                        goals=goals_tc_t,
                        rl2s=rl2_tc_t,
                        seq_lengths=seq_lengths,
                        time_idxs=time_idxs,
                        sample=self.sample_actions,
                        hidden_state=hidden_state if self.fast_inference else None,
                    )

            actions = actions.float().cpu().numpy()
            if self.discrete:
                actions = actions.astype(np.uint8)
            else:
                actions = actions.astype(np.float32)
            _, ext_rew, terminated, truncated, info = envs.step(actions)
            done = terminated | truncated
            get_t(done)
            hidden_state = self.policy.traj_encoder.reset_hidden_state(
                hidden_state, done
            )

            if render:
                envs.render()

        envs.call_async("return_history")
        return_history = envs.call_wait()
        envs.call_async("success_history")
        success_history = envs.call_wait()
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
            if self.verbose:
                print(f"Average Return : {logs['avg_return']}")
            if logs["avg_return"] >= self.best_return:
                self.save_checkpoint(saving_best=True)
                self.best_return = logs["avg_return"]
            self.log(logs, key="val")

    def evaluate_test(
        self, make_test_env: callable, timesteps: int, render: bool = False
    ):
        make = lambda: SequenceWrapper(
            make_test_env(), save_every=None, make_dset=False
        )
        test_envs = gym.vector.AsyncVectorEnv(
            [make for _ in range(self.parallel_actors)]
        )
        *_, returns, successes = self.interact(
            test_envs,
            timesteps,
            render=render,
        )
        logs = self.policy_metrics(returns, successes)
        self.log(logs, key="test")
        return logs

    def log(self, metrics_dict, key):
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().float().item()
            else:
                log_dict[k] = v

        self.train_envs.call_async("total_frames")
        total_frames = sum(self.train_envs.call_wait())
        if self.log_to_wandb:
            wandb.log(
                {f"{key}/{subkey}": val for subkey, val in log_dict.items()}
                | {"total_frames": total_frames}
            )

    def make_figures(self, loss_info):
        figs = {}
        for key in loss_info.keys():
            if "q_seq_mean" in key:
                q_curves = utils.q_curve_plot(loss_info)
                figs.update(q_curves)
                break
        return figs

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
            f"{name}-return": np.array(scores).mean()
            for name, scores in return_by_env_name.items()
        }
        avg_suc_per_env = {
            f"{name}-success": np.array(scores).mean()
            for name, scores in success_by_env_name.items()
        }
        avg_return_overall = {
            "avg_return": np.array(list(avg_ret_per_env.values())).mean()
        }
        return avg_ret_per_env | avg_suc_per_env | avg_return_overall

    def compute_loss(self, batch: Batch, log_step: bool):
        batch.to(self.DEVICE)
        with self.caster():
            critic_loss, actor_loss = self.policy(batch, log_step=log_step)

        update_info = self.policy.update_info
        B, L_1, G, _ = actor_loss.shape
        C = len(self.policy.critics)
        state_mask = (~((batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True))).float()
        critic_state_mask = repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {C} {G} 1")
        actor_state_mask = repeat(state_mask[:, :-1, ...], f"B L 1 -> B L {G} 1")

        masked_actor_loss = (
            actor_state_mask * actor_loss
        ).sum() / actor_state_mask.sum()
        if isinstance(critic_loss, torch.Tensor):
            masked_critic_loss = (
                critic_state_mask * critic_loss
            ).sum() / critic_state_mask.sum()
            optimize_critic = True
        else:
            assert critic_loss is None
            masked_critic_loss = 0.0
            optimize_critic = False

        return {
            "critic_loss": masked_critic_loss,
            "optimize_critic": optimize_critic,
            "actor_loss": masked_actor_loss,
            "mask": state_mask,
        } | update_info

    def _get_grad_norms(self):
        ggn = utils.get_grad_norm
        grads = dict(
            actor_grad_norm=ggn(self.policy.actor),
            crtic_grad_norm=ggn(self.policy.critics),
            traj_encoder_grad_norm=ggn(self.policy.traj_encoder),
            tstep_encoder_grad_norm=ggn(self.policy.tstep_encoder),
            goal_emb_grad_norm=ggn(self.policy.tstep_encoder.goal_emb),
        )
        return grads

    def train_step(self, batch: Batch, log_step: bool):
        """
        See a simplified example of how this changes the classic
        actor-critic update to support a shared sequnce model
        optimized on the actor and critic losses simultaneously
        without a target model:

        https://colab.research.google.com/drive/1XO8MNd_DNArGyoAbroXoQeGblv7Jb9ND#scrollTo=xQe5yry0b6Hl
        """
        l = self.compute_loss(batch, log_step=log_step)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(l["actor_loss"]).backward(retain_graph=True)
        self.policy.critics.zero_grad(set_to_none=True)
        if l["optimize_critic"]:
            self.grad_scaler.scale(
                self.critic_loss_weight * l["critic_loss"]
            ).backward()
        self.grad_scaler.unscale_(self.optimizer)
        if log_step:
            l.update(self._get_grad_norms())
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.trainable_params, max_norm=self.grad_clip
        )
        l["global_grad_norm"] = grad_norm
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.policy.soft_sync_targets()
        if self.half_precision and log_step:
            l["grad_scaler_scale"] = self.grad_scaler.get_scale()
        return l

    def caster(self):
        if self.half_precision:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
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
            self.policy.eval()
            if epoch % self.val_interval == 0:
                self.evaluate_val()
            self.collect_new_training_data()

            # make dataloaders aware of new .traj files
            self.init_dloaders()
            self.policy.train()
            if self.train_dset.count_trajectories() == 0:
                warnings.warn(
                    f"Skipping epoch {epoch} because no training trajectories have been saved yet...",
                    category=Warning,
                )
                continue
            elif epoch < self.start_learning_after_epoch:
                # lets us skip early epochs to prevent overfitting on small datasets
                continue
            for train_step, batch in make_pbar(self.train_dloader, True, epoch):
                total_step = (epoch * self.train_grad_updates_per_epoch) + train_step
                log_step = total_step % self.log_interval == 0
                loss_dict = self.train_step(batch, log_step=log_step)
                if log_step:
                    self.log(loss_dict, key="train-update")
                # lr scheduling done here so we can see epoch/step
                self.warmup_scheduler.step(total_step)

            if epoch % self.val_interval == 0:
                # the "training" metrics on validation data could help identify
                # overfitting or highlight distribution shift between (approx.)
                # on-policy data and way-off-policy data in the disk buffer.
                self.policy.eval()
                for val_step, batch in make_pbar(self.val_dloader, False, epoch):
                    loss_dict = self.val_step(batch)
                    self.log(loss_dict, key="val-update")
                figures = self.make_figures(loss_dict)
                self.log(figures, key="val-update")
                self.val_dset.clear()

            dset_size = self.train_dset.count_trajectories()
            if dset_size > self.dset_max_size:
                self.train_dset.filter(self.dset_filter_pct)
            self.log({"trajectories": dset_size}, key="buffer")

            # end epoch
            self.epoch = epoch
            if epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
