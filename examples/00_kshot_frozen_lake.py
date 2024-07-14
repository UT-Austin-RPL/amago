import gymnasium as gym
import numpy as np

import amago
from amago.envs.builtin.gym_envs import GymEnv, MetaFrozenLake
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--experiment",
        choices=["no-memory", "memory-rnn", "memory-transformer"],
        required=True,
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--buffer_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--lake_size", type=int, default=5)
    parser.add_argument("--k_shots", type=int, default=15)
    parser.add_argument("--hard_mode", action="store_true")
    parser.add_argument("--recover_mode", action="store_true")
    parser.add_argument("--max_rollout_length", type=int, default=512)
    parser.add_argument("--naive", action="store_true")
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()

    if args.log:
        import wandb

    config = {}
    turn_off_goal_conditioning(config)

    # configure trajectory encoder (seq2seq memory model)
    if args.experiment == "memory-rnn":
        traj_encoder = "rnn"
    elif args.experiment == "memory-transformer":
        traj_encoder = "transformer"
    else:
        traj_encoder = "ff"

    # configure timestep encoder
    switch_tstep_encoder(
        config, arch="ff", n_layers=1, d_hidden=128, d_output=64, normalize_inputs=False
    )
    switch_traj_encoder(
        config,
        arch=traj_encoder,
        memory_size=128,
        layers=3,
    )
    if args.naive:
        naive(config, turn_off_fbc=True)

    use_config(config)

    group_name = f"{args.run_name}_{args.experiment}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        # wrap environment
        make_env = lambda: GymEnv(
            MetaFrozenLake(
                k_shots=args.k_shots,
                size=args.lake_size,
                hard_mode=args.hard_mode,
                recover_mode=args.recover_mode,
            ),
            env_name=f"meta_frozen_lake_k{args.k_shots}_{args.lake_size}x{args.lake_size}"
            + ("_hard" if args.hard_mode else "_easy")
            + ("_recover" if args.recover_mode else "_reset"),
            horizon=args.max_rollout_length,
            # "zero-shot" from the *wrapper's perspective*; the MetaFrozenLake
            # is handling k-shots on its own!
            zero_shot=True,
        )

        # create `Experiment`
        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_rollout_length,
            traj_save_len=args.max_rollout_length,
            dset_max_size=10_000,
            run_name=run_name,
            dset_name=run_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dloader_workers=10,
            log_to_wandb=args.log,
            wandb_group_name=group_name,
            epochs=500 if not args.hard_mode else 900,
            parallel_actors=24,
            train_timesteps_per_epoch=350,
            train_grad_updates_per_epoch=700,
            val_interval=20,
            val_timesteps_per_epoch=args.max_rollout_length * 2,
            ckpt_interval=50,
            async_envs=False,
        )

        # start experiment (build envs, policies, etc.)
        experiment.start()
        # run training
        experiment.learn()
        experiment.evaluate_test(make_env, timesteps=10_000)
        wandb.finish()
