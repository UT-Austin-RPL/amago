from argparse import ArgumentParser

import wandb

from amago.envs import AMAGOEnv
from amago.envs.builtin.alchemy import SymbolicAlchemy
from amago.cli_utils import *


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    args = parser.parse_args()

    config = {}
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    exploration_wrapper_type = switch_exploration(
        config, "bilevel", rollout_horizon=200, steps_anneal=2_500_000
    )
    agent_type = switch_agent(config, args.agent_type, reward_multiplier=100.0)
    tstep_encoder_type = switch_tstep_encoder(
        config, arch="ff", n_layers=2, d_hidden=256, d_output=256
    )

    use_config(config, args.configs)
    make_train_env = lambda: AMAGOEnv(
        env=SymbolicAlchemy(),
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
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_wrapper_type,
            agent_type=agent_type,
            val_timesteps_per_epoch=2000,
        )
        switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
