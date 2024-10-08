from argparse import ArgumentParser

import gymnasium as gym
import wandb

import amago
from amago.envs import AMAGOEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--eval_timesteps", type=int, default=1000)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # dictionary that sets default value for kwargs of classes that are marked as `gin.configurable`
        # see https://github.com/google/gin-config for more information. For example:
        "amago.nets.traj_encoders.TformerTrajEncoder.attention": "flash",  # or "vanilla" if flash attention is not available
    }
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    group_name = f"{args.run_name}_{args.env}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        make_train_env = lambda: AMAGOEnv(
            gym.make(args.env),
            env_name=args.env,
        )
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_timesteps,
        )
        experiment = switch_async_mode(experiment, args)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
