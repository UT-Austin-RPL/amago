from argparse import ArgumentParser
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import wandb
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import gin

import amago
from amago.envs import AMAGOEnv
from amago.envs.builtin.xland_minigrid import XLandMinigridVectorizedGym
from amago.nets.utils import add_activation_log, symlog
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--benchmark",
        type=str,
        default="small-1m",
        choices=["trivial-1m", "small-1m", "medium-1m", "high-1m", "high-3m"],
    )
    parser.add_argument("--k_shots", type=int, default=15)
    parser.add_argument("--rooms", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=9)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    return parser


class XLandMiniGridAMAGO(AMAGOEnv):
    def __init__(self, env: XLandMinigridVectorizedGym):
        assert isinstance(env, XLandMinigridVectorizedGym)
        super().__init__(
            env=env,
            env_name=f"XLandMiniGrid-{env.ruleset_benchmark}-R{env.rooms}-{env.grid_size}x{env.grid_size}",
            batched_envs=env.parallel_envs,
        )


@gin.configurable
class XLandMGTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    def __init__(
        self,
        obs_space,
        rl2_space,
        grid_id_dim: int = 8,
        grid_emb_dim: int = 128,
        goal_id_dim: int = 8,
        goal_emb_dim: int = 32,
        ff_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        # grid world embedding
        num_tokens = lambda space: (space.high.max() - space.low.min() + 1).item()
        grid_tokens = num_tokens(obs_space["grid"])
        self.grid_embedding = nn.Embedding(grid_tokens, embedding_dim=grid_id_dim)
        self.grid_processor = amago.nets.cnn.GridworldCNN(
            img_shape=obs_space["grid"].shape,
            channels_first=False,
            activation="leaky_relu",
            channels=[32, 48, 64],
        )
        grid_out_dim = self.grid_processor(self.grid_processor.blank_img).shape[-1]
        self.grid_rep_ff = nn.Linear(grid_out_dim, grid_emb_dim)

        # goal token embedding
        goal_tokens = num_tokens(obs_space["goal"])
        self.goal_embedding = nn.Embedding(goal_tokens, embedding_dim=goal_id_dim)
        goal_inp_dim = goal_id_dim * obs_space["goal"].shape[0]
        self.goal_rep_ff = nn.Sequential(
            nn.Linear(goal_inp_dim, goal_inp_dim),
            nn.LeakyReLU(),
            nn.Linear(goal_inp_dim, goal_emb_dim),
        )

        # merge grid, goal, and other array features
        self.merge = nn.Sequential(
            nn.Linear(
                grid_emb_dim + goal_emb_dim + 5 + self.rl2_space.shape[-1], ff_dim
            ),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, out_dim),
        )
        self.out_norm = amago.nets.ff.Normalization("layer", out_dim)
        self.out_dim = out_dim

    @property
    def emb_dim(self):
        return self.out_dim

    def inner_forward(self, obs, rl2s, log_dict=None):
        grid_rep = self.grid_embedding(obs["grid"].long())
        grid_rep = rearrange(grid_rep, "... h w layers emb -> ... h w (layers emb)")
        grid_rep = self.grid_processor(obs["grid"])
        add_activation_log("encoder-grid-rep", grid_rep, log_dict)
        grid_rep = F.leaky_relu(self.grid_rep_ff(grid_rep))
        add_activation_log("encoder-grid-rep-ff", grid_rep, log_dict)

        goal_rep = self.goal_embedding(obs["goal"].long())
        goal_rep = rearrange(goal_rep, "... length emb -> ... (length emb)")
        goal_rep = F.leaky_relu(self.goal_rep_ff(goal_rep))
        add_activation_log("encoder-goal-rep-ff", grid_rep, log_dict)

        extras = torch.cat((obs["direction_done"], symlog(rl2s)), dim=-1)
        merged_rep = torch.cat((grid_rep, goal_rep, extras), dim=-1)
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
        "amago.envs.exploration.EpsilonGreedy.steps_anneal": 1_000_000,
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -args.k_shots * 10.0 * 10,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": args.k_shots * 10.0 * 10,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 32,
    }

    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    agent_type = switch_agent(config, args.agent_type, reward_multiplier=10.0)
    use_config(config, args.configs)

    xland_kwargs = {
        "parallel_envs": args.parallel_actors,
        "rooms": args.rooms,
        "grid_size": args.grid_size,
        "k_shots": args.k_shots,
        "ruleset_benchmark": args.benchmark,
    }

    args.env_mode = "already_vectorized"
    make_train_env = lambda: XLandMiniGridAMAGO(
        XLandMinigridVectorizedGym(**xland_kwargs, train_test_split="train"),
    )
    make_val_env = lambda: XLandMiniGridAMAGO(
        XLandMinigridVectorizedGym(**xland_kwargs, train_test_split="test"),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        traj_len = make_train_env().suggested_max_seq_len

    group_name = f"{args.run_name}_xlandmg_{args.benchmark}_R{args.rooms}_{args.grid_size}x{args.grid_size}"
    args.start_learning_at_epoch = traj_len // args.timesteps_per_epoch
    args.max_seq_len = min(args.max_seq_len, traj_len)

    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=traj_len,
            stagger_traj_file_lengths=False,
            run_name=run_name,
            tstep_encoder_type=XLandMGTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=traj_len,
            save_trajs_as="npz-compressed",
            grad_clip=2.0,
        )
        switch_async_mode(experiment, args)
        amago_device = experiment.DEVICE.index or torch.cuda.current_device()
        env_device = jax.devices("gpu")[amago_device]
        with jax.default_device(env_device):
            experiment.start()
            if args.ckpt is not None:
                experiment.load_checkpoint(args.ckpt)
            experiment.learn()
            experiment.evaluate_test(make_val_env, timesteps=20_000, render=False)
            experiment.delete_buffer_from_disk()
            wandb.finish()
