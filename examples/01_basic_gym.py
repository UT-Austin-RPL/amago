from argparse import ArgumentParser

import gymnasium as gym
import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="The horizon (H) is the maximum length of a rollout. In most cases this should be >= the maximum length of an environment's episodes.",
    )
    parser.add_argument("--slow_inference", action="store_true")
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

    group_name = f"{args.run_name}_{args.env}_context_length_{args.max_seq_len}"
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"

        make_train_env = lambda: GymEnv(
            gym.make(args.env),
            env_name=args.env,
            horizon=args.horizon,
            zero_shot=True,
        )

        experiment = amago.Experiment(
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            dset_max_size=args.dset_max_size,
            run_name=dset_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dset_name=dset_name,
            log_to_wandb=not args.no_log,
            epochs=args.epochs,
            half_precision=args.half_precision,
            fast_inference=not args.slow_inference,
            parallel_actors=args.parallel_actors,
            train_timesteps_per_epoch=args.timesteps_per_epoch,
            train_grad_updates_per_epoch=args.grads_per_epoch,
            val_interval=args.val_interval,
            val_timesteps_per_epoch=args.horizon * 5,
            ckpt_interval=args.ckpt_interval,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
        wandb.finish()
