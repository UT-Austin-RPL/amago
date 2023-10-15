from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.envs.builtin.alchemy import SymbolicAlchemy
from utils import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hide_rl2s",
        action="store_true",
        help="hides the 'rl2 info' (previous actions, rewards, current time)",
    )
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
    use_config(config, args.configs)
    # this is another example where the environment handles the
    # k-shot learning for us, and amago acts as if it is zero-shot.
    make_train_env = lambda: GymEnv(
        gym_env=SymbolicAlchemy(),
        env_name="dm_symbolic_alchemy",
        horizon=201,
        zero_shot=True,
    )
    group_name = f"{args.run_name}_symbolic_dm_alchemy"
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"
        experiment = amago.Experiment(
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=201,
            traj_save_len=201 * 4,
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
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        wandb.finish()
