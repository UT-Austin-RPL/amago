from argparse import ArgumentParser

import gymnasium as gym
import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--reward_scale", type=int, default=1.0)
    parser.add_argument("--no_popart", action="store_true")
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="The horizon (H) is the maximum length of a rollout. In most cases this should be >= the maximum length of an environment's episodes.",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.agent.Agent.reward_multiplier": args.reward_scale,
        "amago.agent.Agent.popart": not args.no_popart,
    }
    turn_off_goal_conditioning(config)
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

        make_train_env = lambda: GymEnv(
            gym.make(args.env),
            env_name=args.env,
            horizon=args.horizon,
            zero_shot=True,
        )
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=args.horizon * 5,
        )
        experiment = switch_mode_load_ckpt(experiment, args)
        experiment.start()
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
        wandb.finish()
