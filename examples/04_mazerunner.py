from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.mazerunner import MazeRunnerEnv
from utils import *


def add_cli(parser):
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--maze_dim", type=int, default=15)
    parser.add_argument("--min_goals", type=int, default=1)
    parser.add_argument("--max_goals", type=int, default=3)
    parser.add_argument(
        "--relabel", choices=["some", "none", "all", "all_or_nothing"], default="some"
    )
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
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="ff", n_layers=2, d_hidden=128, d_output=64)
    use_config(config, args.configs)

    make_env = lambda: MazeRunnerEnv(
        maze_dim=args.maze_dim,
        min_num_goals=args.min_goals,
        max_num_goals=args.max_goals,
        max_timesteps=args.horizon,
    )

    traj_save_len = max_seq_len = args.horizon
    group_name = (
        f"{args.run_name}_mazerunner_{args.maze_dim}x{args.maze_dim}_k{args.max_goals}"
    )
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"
        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=max_seq_len + 1,
            traj_save_len=traj_save_len + 1,
            dset_max_size=args.dset_max_size,
            run_name=dset_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dset_name=dset_name,
            log_to_wandb=not args.no_log,
            epochs=args.epochs,
            parallel_actors=args.parallel_actors,
            train_timesteps_per_epoch=args.timesteps_per_epoch,
            train_grad_updates_per_epoch=args.grads_per_epoch,
            val_interval=args.val_interval,
            val_timesteps_per_epoch=2_000,
            ckpt_interval=args.ckpt_interval,
            # Hindsight Relabeling!
            relabel=args.relabel,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_env, timesteps=20_000, render=False)
        wandb.finish()
