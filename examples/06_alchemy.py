from argparse import ArgumentParser

import wandb

from amago.envs import AMAGOEnv
from amago.envs.builtin.alchemy import SymbolicAlchemy
from amago.cli_utils import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hide_rl2s",
        action="store_true",
        help="hides the 'rl2 info' (previous actions, rewards, current time)",
    )
    add_common_cli(parser)
    args = parser.parse_args()

    config = {"amago.nets.tstep_encoders.FFTstepEncoder.hide_rl2s": args.hide_rl2s}
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)
    make_train_env = lambda: AMAGOEnv(
        gym_env=SymbolicAlchemy(),
        env_name="dm_symbolic_alchemy",
    )
    group_name = f"{args.run_name}_symbolic_dm_alchemy"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=201,
            traj_save_len=201,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=2000,
        )
        switch_async_mode(experiment, args)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
