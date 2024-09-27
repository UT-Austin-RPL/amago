from argparse import ArgumentParser
from functools import partial

import wandb
import torch
from torch import nn
import gymnasium as gym
from einops import rearrange

import amago
from amago.envs import AMAGOEnv
from amago.envs.builtin.xland_minigrid import XLandMiniGridEnv
from amago.nets.utils import add_activation_log, symlog
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--benchmark",
        type=str,
        default="trivial-1m",
        choices=["trivial-1m", "small-1m", "medium-1m", "high-1m", "high-3m"],
    )
    parser.add_argument("--xland_device", type=int, default=None)
    parser.add_argument("--k_shots", type=int, default=5)
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=13)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    return parser


class XLandMiniGridAMAGO(AMAGOEnv):
    def __init__(self, env: XLandMiniGridEnv):
        assert isinstance(env, XLandMiniGridEnv)
        super().__init__(
            env=env,
            env_name=f"XLandMiniGrid-{env.ruleset_benchmark}-R{env.rooms}-{env.grid_size}x{env.grid_size}",
            batched_envs=env.parallel_envs,
        )


class XLandMGTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    def __init__(
        self,
        obs_space,
        rl2_space,
        grid_id_dim: int = 8,
        ff_dim: int = 384,
        out_dim: int = 256,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.embedding = nn.Embedding(15, embedding_dim=grid_id_dim)
        self.img_processor = amago.nets.cnn.GridworldCNN(
            img_shape=obs_space["grid"].shape,
            channels_first=False,
            activation="leaky_relu",
            channels=[32, 48, 64],
        )
        img_out_dim = self.img_processor(
            torch.zeros((1, 1) + obs_space["grid"].shape, dtype=torch.uint8)
        ).shape[-1]
        self.merge = nn.Sequential(
            nn.Linear(img_out_dim + 4 + 1 + self.rl2_space.shape[-1], ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, out_dim),
        )
        self.out_norm = amago.nets.ff.Normalization("layer", out_dim)
        self.out_dim = out_dim

    @property
    def emb_dim(self):
        return self.out_dim

    def inner_forward(self, obs, rl2s, log_dict=None):
        grid_rep = self.embedding(obs["grid"].long())
        grid_rep = rearrange(grid_rep, "... h w layers emb -> ... h w (layers emb)")
        grid_rep = self.img_processor(obs["grid"])
        add_activation_log("encoder-grid-rep", grid_rep, log_dict)
        extras = torch.cat((obs["direction_done"], symlog(rl2s)), dim=-1)
        merged_rep = torch.cat((grid_rep, extras), dim=-1)
        merged_rep = self.merge(merged_rep)
        add_activation_log("encoder-merged-rep", merged_rep, log_dict)
        out = self.out_norm(merged_rep)
        return out


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()
    config = {
        "amago.agent.Agent.reward_multiplier": 100.0,
        "amago.agent.Agent.tstep_encoder_Cls": XLandMGTstepEncoder,
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -100_000,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 100_000,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 64,
    }

    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    xland_kwargs = {
        "parallel_envs": args.parallel_actors,
        "rooms": args.rooms,
        "grid_size": args.grid_size,
        "k_shots": args.k_shots,
        "ruleset_benchmark": args.benchmark,
        "jax_device": args.xland_device,
    }

    args.env_mode = "already_vectorized"
    make_train_env = lambda: XLandMiniGridAMAGO(
        XLandMiniGridEnv(**xland_kwargs, train_test_split="train"),
    )
    make_val_env = lambda: XLandMiniGridAMAGO(
        XLandMiniGridEnv(**xland_kwargs, train_test_split="test"),
    )
    traj_len = make_train_env().suggested_max_seq_len

    group_name = f"{args.run_name}_xlandmg_{args.benchmark}_R{args.rooms}_{args.grid_size}x{args.grid_size}"
    args.start_learning_at_epoch = traj_len // args.timesteps_per_epoch

    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=traj_len,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=traj_len,
            save_trajs_as="npz",
        )
        switch_async_mode(experiment, args)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_val_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
