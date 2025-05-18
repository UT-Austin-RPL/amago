import os

import amago
from amago.envs.builtin.toy_gym import MetaFrozenLake
from amago.envs import AMAGOEnv
from amago.loading import DiskTrajDataset
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--seq_model",
        type=str,
        choices=["ff", "transformer", "rnn", "mamba"],
        required=True,
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--buffer_dir", type=str, required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--lake_size", type=int, default=5)
    parser.add_argument("--k_shots", type=int, default=15)
    parser.add_argument("--hard_mode", action="store_true")
    parser.add_argument("--recover_mode", action="store_true")
    parser.add_argument("--max_rollout_length", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()

    if args.log:
        import wandb

    config = {}
    # configure trajectory encoder (seq2seq memory model)
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.seq_model,
        memory_size=128,
        layers=3,
    )
    # configure timestep encoder
    tstep_encoder_type = switch_tstep_encoder(
        config, arch="ff", n_layers=1, d_hidden=128, d_output=64, normalize_inputs=False
    )

    # we're using the default exploration strategy but being overly verbose about it for the example
    exploration_wrapper_type = switch_exploration(
        config,
        strategy="egreedy",
        eps_start=1.0,
        eps_end=0.05,
        steps_anneal=1_000_000,
        randomize_eps=True,
    )
    use_config(config)

    group_name = f"{args.run_name}_{args.seq_model}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        # create a dataset on disk. envs will write finished episodes here
        dset = DiskTrajDataset(
            dset_root=args.buffer_dir, dset_name=run_name, dset_max_size=12_500
        )
        # save checkpoints alongside the buffer
        ckpt_dir = args.buffer_dir

        # wrap environment
        make_env = lambda: AMAGOEnv(
            MetaFrozenLake(
                k_shots=args.k_shots,
                size=args.lake_size,
                hard_mode=args.hard_mode,
                recover_mode=args.recover_mode,
            ),
            env_name=f"meta_frozen_lake_k{args.k_shots}_{args.lake_size}x{args.lake_size}"
            + ("_hard" if args.hard_mode else "_easy")
            + ("_recover" if args.recover_mode else "_reset"),
        )

        # create `Experiment`
        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_rollout_length,
            dataset=dset,
            ckpt_base_dir=ckpt_dir,
            agent_type=amago.agent.Agent,
            exploration_wrapper_type=exploration_wrapper_type,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            run_name=run_name,
            dloader_workers=10,
            log_to_wandb=args.log,
            wandb_group_name=group_name,
            epochs=700 if not args.hard_mode else 900,
            parallel_actors=24,
            train_timesteps_per_epoch=350,
            train_batches_per_epoch=800,
            val_interval=20,
            val_timesteps_per_epoch=args.max_rollout_length * 2,
            ckpt_interval=50,
            env_mode="sync",
        )

        # start experiment (build envs, policies, etc.)
        experiment.start()
        # run training
        experiment.learn()
        experiment.evaluate_test(make_env, timesteps=10_000)
        experiment.delete_buffer_from_disk()
        wandb.finish()
