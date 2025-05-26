from argparse import ArgumentParser

import gymnasium as gym
import wandb

import amago
from amago.envs import AMAGOEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--env", type=str, required=True, help="Environment name for `gym.make`"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Policy sequence length."
    )
    parser.add_argument(
        "--eval_timesteps",
        type=int,
        default=1000,
        help="Timesteps per actor per evaluation. Tune based on the episode length of the environment (to be at least one full episode).",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # setup environment
    env_name = args.env.replace("/", "_")
    make_train_env = lambda: AMAGOEnv(
        gym.make(args.env),
        env_name=env_name,
    )
    config = {
        # dictionary that sets default value for kwargs of classes that are marked as `gin.configurable`
        # see `tutorial.md` for more information. For example:
        "amago.nets.policy_dists.Discrete.clip_prob_high": 1.0,
        "amago.nets.policy_dists.Discrete.clip_prob_low": 1e-6,
    }
    # switch sequence model
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    # switch agent
    agent_type = switch_agent(
        config,
        args.agent_type,
        reward_multiplier=1.0,
    )
    use_config(config, args.configs)

    group_name = f"{args.run_name}_{env_name}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 8,
            run_name=run_name,
            tstep_encoder_type=amago.nets.tstep_encoders.FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_timesteps,
        )
        experiment = switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
