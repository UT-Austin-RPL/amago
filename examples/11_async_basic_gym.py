from argparse import ArgumentParser

import gymnasium as gym
import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="The horizon (H) is the maximum length of a rollout. In most cases this should be >= the maximum length of an environment's episodes.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="collect",
        choices=["learn", "collect", "eval", "both"],
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {}
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    group_name = run_name = f"{args.run_name}_{args.env}"

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
        save_trajs_as="npz",
    )

    if args.mode == "collect":
        experiment = make_experiment_collect_only(experiment)
    elif args.mode == "learn":
        experiment = make_experiment_learn_only(experiment)

    experiment.start()
    if args.mode == "eval":
        assert args.ckpt
        experiment.load_checkpoint(args.ckpt)
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
    else:
        experiment.learn()

    wandb.finish()
