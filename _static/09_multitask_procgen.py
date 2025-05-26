from argparse import ArgumentParser

import wandb

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
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    tstep_encoder_type = switch_tstep_encoder(
        config,
        arch="cnn",
        cnn_type=IMPALAishCNN,
        channels_first=False,
        drqv2_aug=True,
    )
    agent_type = switch_agent(config, args.agent_type)
    use_config(config, args.configs)

    procgen_kwargs = PROCGEN_SETTINGS[args.distribution]
    horizon = 2000 if "easy" in args.distribution else 5000
    make_train_env = lambda: ProcgenAMAGO(
        TwoShotMTProcgen(**procgen_kwargs, seed_range=(0, args.train_seeds)),
    )
    make_test_env = lambda: ProcgenAMAGO(
        TwoShotMTProcgen(
            **procgen_kwargs, seed_range=(args.train_seeds + 1, 10_000_000)
        ),
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
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=5 * horizon + 1,
        )
        switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_test_env, timesteps=horizon * 20, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
