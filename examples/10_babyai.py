from argparse import ArgumentParser
from functools import partial

import wandb
import torch
from torch import nn
import gymnasium as gym

import amago
from amago.envs.builtin.babyai import MultitaskMetaBabyAI, ALL_BABYAI_TASKS
from amago.envs import AMAGOEnv
from amago.nets.utils import add_activation_log, symlog
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--obs_kind",
        choices=["partial-grid", "full-grid", "partial-image", "full-image"],
        default="partial-grid",
    )
    parser.add_argument("--k_episodes", type=int, default=2)
    parser.add_argument("--train_seeds", type=int, default=5_000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    return parser


TRAIN_TASKS = [
    "BabyAI-GoToLocalS7N5-v0",
    "BabyAI-GoToObjMaze-v0",
    "BabyAI-KeyCorridor-v0",
    "BabyAI-KeyCorridorS3R3-v0",
    "BabyAI-GoToRedBall-v0",
    "BabyAI-KeyCorridorS3R2-v0",
    "BabyAI-KeyCorridorS3R1-v0",
    "BabyAI-Unlock-v0",
    "BabyAI-GoToLocalS8N4-v0",
    "BabyAI-GoToObjMazeOpen-v0",
    "BabyAI-KeyCorridorS4R3-v0",
    "BabyAI-UnlockLocal-v0",
    "BabyAI-GoToObjMazeS5-v0",
    "BabyAI-GoToObjMazeS4R2-v0",
    "BabyAI-GoToLocal-v0",
    "BabyAI-PickupLoc-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-GoTo-v0",
    "BabyAI-FindObjS6-v0",
    "BabyAI-BlockedUnlockPickup-v0",
    "BabyAI-KeyCorridorS5R3-v0",
    "BabyAI-GoToObjS6-v0",
    "BabyAI-KeyInBox-v0",
    "BabyAI-Open-v0",
    "BabyAI-GoToOpen-v0",
    "BabyAI-GoToDoor-v0",
    "BabyAI-FindObjS7-v0",
    "BabyAI-OpenRedDoor-v0",
    "BabyAI-PickupDist-v0",
    "BabyAI-GoToImpUnlock-v0",
    "BabyAI-UnblockPickup-v0",
    "BabyAI-OpenDoor-v0",
    "BabyAI-GoToObjMazeS4-v0",
    "BabyAI-OneRoomS12-v0",
    "BabyAI-GoToObjMazeS6-v0",
    "BabyAI-GoToRedBallNoDists-v0",
    "BabyAI-OpenDoorDebug-v0",
    "BabyAI-GoToLocalS8N5-v0",
    "BabyAI-OneRoomS20-v0",
    "BabyAI-Pickup-v0",
    "BabyAI-GoToRedBlueBall-v0",
    "BabyAI-OpenDoorColor-v0",
    "BabyAI-PickupAbove-v0",
    "BabyAI-GoToObjDoor-v0",
    "BabyAI-OpenRedBlueDoors-v0",
    "BabyAI-UnlockToUnlock-v0",
    "BabyAI-OneRoomS16-v0",
    "BabyAI-GoToLocalS8N6-v0",
    "BabyAI-OneRoomS8-v0",
    "BabyAI-PickupDistDebug-v0",
]
TEST_TASKS = ALL_BABYAI_TASKS


class BabyAIAMAGOEnv(AMAGOEnv):
    def __init__(self, env: gym.Env):
        assert isinstance(env, MultitaskMetaBabyAI)
        super().__init__(
            env=env,
        )

    @property
    def env_name(self):
        return self.env.current_task


class BabyTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    def __init__(
        self,
        obs_kind: str,
        obs_space,
        rl2_space,
        extras_dim: int = 16,
        mission_dim: int = 48,
        emb_dim: int = 300,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        breakpoint()
        self.obs_kind = obs_kind
        if obs_kind in ["partial-image", "full-image"]:
            cnn_type = amago.nets.cnn.NatureishCNN
        else:
            cnn_type = amago.nets.cnn.GridworldCNN
        self.img_processor = cnn_type(
            img_shape=obs_space["image"].shape,
            channels_first=False,
            activation="leaky_relu",
        )
        img_out_dim = self.img_processor(self.img_processor.blank_img).shape[-1]

        low_token = obs_space["mission"].low.min()
        high_token = obs_space["mission"].high.max()
        self.mission_processor = amago.nets.goal_embedders.TokenGoalEmb(
            goal_length=9,
            goal_dim=1,
            min_token=low_token,
            max_token=high_token,
            goal_emb_dim=mission_dim,
            embedding_dim=18,
            hidden_size=96,
        )
        self.extras_processor = nn.Sequential(
            nn.Linear(obs_space["extra"].shape[-1] + rl2_space.shape[-1], 32),
            nn.LeakyReLU(),
            nn.Linear(32, extras_dim),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(img_out_dim + mission_dim + extras_dim, emb_dim)
        self.out_norm = amago.nets.ff.Normalization("layer", emb_dim)
        self._emb_dim = emb_dim

    @property
    def emb_dim(self):
        return self._emb_dim

    def inner_forward(self, obs, rl2s, log_dict=None):
        rl2s = symlog(rl2s)
        extras = torch.cat((rl2s, obs["extra"]), dim=-1)
        extras_rep = self.extras_processor(extras)
        add_activation_log("encoder-extras-rep", extras_rep, log_dict)
        mission_rep = self.mission_processor(obs["mission"].unsqueeze(-1))
        add_activation_log("encoder-mission-rep", extras_rep, log_dict)
        img_rep = self.img_processor(obs["image"])
        add_activation_log("encoder-img-rep", extras_rep, log_dict)
        merged_rep = torch.cat((img_rep, mission_rep, extras_rep), dim=-1)
        out = self.out_norm(self.out(merged_rep))
        return out


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": None,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": None,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 64,
    }
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    agent_type = switch_agent(config, args.agent_type, reward_multiplier=1000.0)
    use_config(config, args.configs)

    make_train_env = lambda: BabyAIAMAGOEnv(
        MultitaskMetaBabyAI(
            task_names=TRAIN_TASKS,
            seed_range=(0, args.train_seeds),
            k_episodes=args.k_episodes,
            observation_type=args.obs_kind,
        )
    )

    make_val_env = lambda: BabyAIAMAGOEnv(
        MultitaskMetaBabyAI(
            task_names=TEST_TASKS,
            seed_range=(args.train_seeds + 1, 1_000_000),
            k_episodes=args.k_episodes,
            observation_type=args.obs_kind,
        )
    )

    group_name = f"{args.run_name}_babyai_{args.obs_kind}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 3,
            stagger_traj_file_lengths=True,
            run_name=run_name,
            tstep_encoder_type=partial(BabyTstepEncoder, obs_kind=args.obs_kind),
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=6000,
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
