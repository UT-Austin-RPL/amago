from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.procgen_envs import (
    TwoShotMTProcgen,
    ProcgenAMAGO,
    ALL_PROCGEN_GAMES,
)
from amago.nets.cnn import IMPALAishCNN
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--distribution",
        type=str,
        default="easy",
        choices=["easy", "easy-rescaled", "memory-hard"],
    )
    parser.add_argument("--train_seeds", type=int, default=10_000)
    return parser


PROCGEN_SETTINGS = {
    "easy": {
        "games": ["climber", "coinrun", "jumper", "ninja", "leaper"],
        "reward_scales": {},
        "distribution_mode": "easy",
    },
    "easy-rescaled": {
        "games": ["climber", "coinrun", "jumper", "ninja", "leaper"],
        "reward_scales": {"coinrun": 100.0, "climber": 0.1},
        "distribution_mode": "easy",
    },
    "memory-hard": {
        "games": ALL_PROCGEN_GAMES,
        "reward_scales": {},
        "distribution_mode": "memory-hard",
    },
}

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

    switch_tstep_encoder(config, arch="cnn", cnn_Cls=IMPALAishCNN, channels_first=False)
    use_config(config, args.configs)

    procgen_kwargs = PROCGEN_SETTINGS[args.distribution]
    horizon = 2000 if "easy" in args.distribution else 5000
    make_train_env = lambda: ProcgenAMAGO(
        TwoShotMTProcgen(**procgen_kwargs, seed_range=(0, args.train_seeds)),
        horizon=horizon,
    )
    make_test_env = lambda: ProcgenAMAGO(
        TwoShotMTProcgen(
            **procgen_kwargs, seed_range=(args.train_seeds + 1, 10_000_000)
        ),
        horizon=horizon,
    )

    group_name = f"{args.run_name}_{args.distribution}_procgen_l_{args.max_seq_len}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_test_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=5 * horizon + 1,
        )
        switch_mode_load_ckpt(experiment, args)
        experiment.start()
        experiment.learn()
        experiment.evaluate_test(make_test_env, timesteps=horizon * 20, render=False)
        wandb.finish()
