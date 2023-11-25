from argparse import ArgumentParser

import wandb
import gym

import amago
from amago.envs.builtin.ale_retro import ALE, AtariAMAGOWrapper
from example_utils import *


def add_cli(parser):
    parser.add_argument("--games", nargs="+")
    parser.add_argument("--max_seq_len", type=int, default=256)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    add_common_cli(parser)
    args = parser.parse_args()

    config = {}
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="cnn", channels_first=True)
    use_config(config, args.configs)

    # ALE (and retro) stack can use MultiBinary actions, but this is not currently
    # supported by the open-source version of the core agent.
    make_env = lambda: AtariAMAGOWrapper(ALE(args.games, use_discrete_actions=True))

    group_name = f"{args.run_name}_{''.join(args.games)}_atari_l_{args.max_seq_len}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=10_000,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_env, timesteps=50_000, render=False)
        wandb.finish()
