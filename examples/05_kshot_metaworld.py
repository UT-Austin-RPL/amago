from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.metaworld_ml import Metaworld
from example_utils import *


def add_cli(parser):
    parser.add_argument(
        "--benchmark",
        type=str,
        default="reach-v2",
        help="`name-v2` for ML1, or `ml10`/`ml45`",
    )
    parser.add_argument("--k", type=int, default=5, help="K-Shots")
    parser.add_argument("--max_seq_len", type=int, default=2000)
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
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    """
    The easiest way to do k-shot meta-learning (where the end of meta-testing is
    defined by a fixed number of episodes, rather than a fixed number of timesteps)
    is to handle that logic in an environment wrapper. 
    See `amago.envs.builtin.metaworld_ml.KShotMetaworld`. The environment auto-resets
    k times, which lets amago treat it as if it was zero-shot. The only trick is to add
    the resets to the observation space.

    Metaworld actually always has `500 * args.k` timesteps - so it could use the AMAGOEnv
    `soft_reset_kwargs` (see dark_key_to_door example) - but we do it this way as an
    example.
    """

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
            val_timesteps_per_epoch=2 * args.k * 500 + 1,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_test_env, timesteps=20_000, render=False)
        wandb.finish()
