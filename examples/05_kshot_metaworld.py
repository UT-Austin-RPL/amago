from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.metaworld_ml import Metaworld
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--benchmark",
        type=str,
        default="reach-v2",
        help="`name-v2` for ML1, or `ml10`/`ml45`",
    )
    parser.add_argument("--k", type=int, default=3, help="K-Shots")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--hide_rl2s",
        action="store_true",
        help="hides the 'rl2 info' (previous actions, rewards, current time)",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.envs.env_utils.ExplorationWrapper.steps_anneal": 2_000_000,
        "amago.nets.tstep_encoders.FFTstepEncoder.hide_rl2s": args.hide_rl2s,
        "amago.agent.Agent.reward_multiplier": 1.0,
        # delete the next three lines to use the paper settings, which were
        # intentionally left wide open to avoid reward-specific tuning.
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -100.0,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 5000 * args.k,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 96,
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    make_train_env = lambda: Metaworld(args.benchmark, "train", k_shots=args.k)
    make_test_env = lambda: Metaworld(args.benchmark, "test", k_shots=args.k)

    group_name = (
        f"{args.run_name}_metaworld_{args.benchmark}_K_{args.k}_L_{args.max_seq_len}"
    )
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=min(500 * args.k + 1, args.max_seq_len * 4),
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=10 * args.k * 500 + 1,
            learning_rate=5e-4,
            grad_clip=2.0,
        )

        experiment = switch_mode_load_ckpt(experiment, args)
        experiment.start()
        experiment.learn()
        experiment.evaluate_test(make_test_env, timesteps=20_000, render=False)
        wandb.finish()
