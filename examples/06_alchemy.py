from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
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

        switch_mode_load_ckpt(experiment, args)
        experiment.start()
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        wandb.finish()
