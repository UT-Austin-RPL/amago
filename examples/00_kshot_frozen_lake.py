import gymnasium as gym
import numpy as np

import amago
from amago.envs.builtin.gym_envs import GymEnv, RandomFrozenLake
from example_utils import *


def add_cli(parser):
    parser.add_argument(
        "--experiment",
        choices=["no-memory", "memory-rnn", "memory-transformer"],
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--buffer_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--lake_size", type=int, default=5)
    parser.add_argument("--k_shots", type=int, default=10)
    parser.add_argument("--max_rollout_length", type=int, default=1000)
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
    use_config(config)

    group_name = f"{args.run_name}_{args.experiment}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        # wrap environment
        make_env = lambda: GymEnv(
            RandomFrozenLake(k_shots=args.k_shots, size=args.lake_size),
            env_name="random_frozen_lake_k{args.k_shots}_{args.lake_size}x{args.lake_size}",
            horizon=args.max_rollout_length,
            # "zero-shot" from the *wrapper's perspective*; the RandomFrozenLake
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
            epochs=300,
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
        # load the best checkpoint (highest validation env return)
        experiment.load_checkpoint(loading_best=True)
        # evaluate best policy
        experiment.evaluate_test(make_val_env, timesteps=10_000)
        wandb.finish()
