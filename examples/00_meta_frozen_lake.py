from argparse import ArgumentParser

import amago
from amago.envs.builtin.toy_gym import MetaFrozenLake
from amago.envs import AMAGOEnv
from amago.loading import DiskTrajDataset
from amago import cli_utils


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
    parser.add_argument("--k_episodes", type=int, default=10)
    parser.add_argument("--hard_mode", action="store_true")
    parser.add_argument("--recover_mode", action="store_true")
    parser.add_argument("--slip_chance", type=float, default=0.0)
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Max steps per attempt. Default: N² (standard) or 2*N² (hard).",
    )
    parser.add_argument(
        "--hide_k_progress",
        action="store_true",
        help="Hide current_k/k_episodes from observations (for length extrapolation tests).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Training sequence length. Default: max_episode_steps * k_episodes (full trajectory).",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()

    if args.log:
        import wandb

    lake_kwargs = dict(
        size=args.lake_size,
        k_episodes=args.k_episodes,
        hard_mode=args.hard_mode,
        recover_mode=args.recover_mode,
        max_episode_steps=args.max_episode_steps,
        show_k_progress=not args.hide_k_progress,
        slip_chance=args.slip_chance,
    )
    max_ep_steps = MetaFrozenLake(**lake_kwargs).max_episode_steps
    max_rollout_length = max_ep_steps * args.k_episodes
    max_seq_len = args.max_seq_len or max_rollout_length

    config = {}
    # configure trajectory encoder (seq2seq memory model)
    traj_encoder_type = cli_utils.switch_traj_encoder(
        config,
        arch=args.seq_model,
        memory_size=128,
        layers=3,
    )
    # configure timestep encoder
    tstep_encoder_type = cli_utils.switch_tstep_encoder(
        config, arch="ff", n_layers=1, d_hidden=128, d_output=64, normalize_inputs=False
    )
    # we're using the default exploration strategy but being overly verbose about it for the example
    exploration_wrapper_type = cli_utils.switch_exploration(
        config,
        strategy="egreedy",
        eps_start=1.0,
        eps_end=0.05,
        steps_anneal=1_000_000,
        randomize_eps=True,
    )
    agent_type = cli_utils.switch_agent(config, "agent", tau=0.004)
    cli_utils.use_config(config)

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
            MetaFrozenLake(**lake_kwargs),
            env_name=f"meta_frozen_lake_k{args.k_episodes}_{args.lake_size}x{args.lake_size}"
            + ("_hard" if args.hard_mode else "_easy")
            + ("_recover" if args.recover_mode else "_reset"),
        )

        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=max_seq_len,
            traj_save_len=max_rollout_length,
            dataset=dset,
            ckpt_base_dir=args.buffer_dir,
            agent_type=agent_type,
            exploration_wrapper_type=exploration_wrapper_type,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            run_name=run_name,
            dloader_workers=10,
            log_to_wandb=args.log,
            wandb_group_name=group_name,
            epochs=700 if not args.hard_mode else 900,
            parallel_actors=32,
            train_timesteps_per_epoch=max_rollout_length,
            train_batches_per_epoch=1000,
            val_interval=20,
            val_timesteps_per_epoch=max_rollout_length * 2,
            ckpt_interval=200,
            env_mode="sync",
        )

        experiment.start()
        experiment.learn()
        experiment.evaluate_test(make_env, timesteps=10_000)
        experiment.delete_buffer_from_disk()
        if args.log:
            wandb.finish()
