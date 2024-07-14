from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.gym_envs import POPGymEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, default="AutoencodeEasy")
    parser.add_argument("--max_seq_len", type=int, default=2000)
    parser.add_argument("--traj_save_len", type=int, default=2000)
    parser.add_argument("--naive", action="store_true")
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # no need to risk numerical instability when returns are this bounded
        "amago.agent.Agent.reward_multiplier": 100.0,
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        # NOTE: paper (and original POPGym results) use `memory_size=256`
        memory_size=args.memory_size,
        # NOTE: paper used layers=3
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="ff", n_layers=2, d_hidden=512, d_output=200)
    if args.naive:
        naive(config)
    use_config(config, args.configs)

    group_name = f"{args.run_name}_{args.env}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        make_train_env = lambda: POPGymEnv(f"popgym-{args.env}-v0")
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            # For this one script to work across every environment,
            # these are arbitrary sequence limits we'll never each.
            max_seq_len=args.max_seq_len,
            traj_save_len=args.traj_save_len,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=2000,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        wandb.finish()
