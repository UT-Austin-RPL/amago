from argparse import ArgumentParser
import os
import multiprocessing as mp

import wandb
import gymnasium as gym
import gym as og_gym
import numpy as np
import memory_maze

from amago.envs import AMAGOEnv
from amago.envs.env_utils import space_convert
from amago.nets.cnn import NatureishCNN, IMPALAishCNN
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--maze_size",
        type=int,
        default=9,
        choices=[9, 11, 13, 15],
        help="n x n dimension of the maze.",
    )
    parser.add_argument(
        "--policy_train_seq_len",
        type=int,
        default=256,
        help="Sequence length used to train the memory policy.",
    )
    parser.add_argument(
        "--cnn",
        type=str,
        choices=["nature", "impala"],
        default="impala",
        help="CNN architecture that embeds images to memory seq",
    )
    return parser


MAZE_SIZE_TO_EP_LENGTH = {
    9: 1000,
    11: 2000,
    13: 3000,
    15: 4000,
}


class MinimalMakeGymPlayGymnasium(gym.Env):
    """
    Create an original gymnasium env and use a gymnasium
    interface from this point upwards in the wrapper stack.
    """

    def __init__(self, gym_env_name):
        super().__init__()
        self.env = og_gym.make(gym_env_name)
        self.observation_space = space_convert(self.env.observation_space)
        self.action_space = space_convert(self.env.action_space)

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, done, info


if __name__ == "__main__":
    # apparently needed for async mujoco rendernig
    mp.set_start_method("spawn")

    # CLI
    parser = ArgumentParser()
    add_cli(parser)
    add_common_cli(parser)
    args = parser.parse_args()

    # Configuration
    config = {}
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    if args.cnn == "nature":
        cnn_type = NatureishCNN
    elif args.cnn == "impala":
        cnn_type = IMPALAishCNN
    tstep_encoder_type = switch_tstep_encoder(
        config,
        arch="cnn",
        cnn_type=cnn_type,
        channels_first=False,
        drqv2_aug=True,
    )
    agent_type = switch_agent(
        config, args.agent_type, tau=0.005, gamma=0.995, reward_multiplier=100.0
    )
    use_config(config, args.configs)

    # Env Creation
    ep_len = MAZE_SIZE_TO_EP_LENGTH[args.maze_size]
    env_name = f"MemoryMaze-{args.maze_size}x{args.maze_size}"

    def make_maze_env():
        return AMAGOEnv(
            MinimalMakeGymPlayGymnasium(f"memory_maze:{env_name}-v0"),
            batched_envs=1,
            env_name=env_name,
        )

    group_name = f"{args.run_name}_memory_maze_{args.maze_size}x{args.maze_size}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_maze_env,
            make_val_env=make_maze_env,
            max_seq_len=args.policy_train_seq_len,
            # we load entire files from disk to then sample
            # (up to) `max_seq_len` subseqs from those files.
            # so when we're training on image envs we're loading
            # large video files and prefer them to be shorter
            # (and use more --dloader_workers)
            traj_save_len=args.policy_train_seq_len * 2,
            stagger_traj_file_lengths=True,
            run_name=run_name,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=ep_len + 1,
            # if disk space is a concern, there's an "npz-compressed"
            # option here. It saves space for video data but is a big
            # slowdown during dataloading.
            save_trajs_as="npz",  # or npz-compressed
            local_time_optimizer=True,
            learning_rate=2e-4,
            grad_clip=2.0,
        )
        switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(env_funcs, timesteps=ep_len * 5, render=False)
        # experiment.delete_buffer_from_disk()
        wandb.finish()
