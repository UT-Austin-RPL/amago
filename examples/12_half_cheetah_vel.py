from argparse import ArgumentParser

import wandb

import amago
from amago.envs import AMAGOEnv
from amago.envs.builtin.half_cheetah_v4_vel import HalfCheetahV4_MetaVelocity
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--policy_seq_len", type=int, default=32, help="Policy sequence length."
    )
    parser.add_argument(
        "--eval_episodes_per_actor",
        type=int,
        default=1,
        help="Validation episodes per parallel actor.",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # setup environment
    make_train_env = lambda: AMAGOEnv(
        HalfCheetahV4_MetaVelocity(),
        env_name="HalfCheetahV4_MetaVelocity",
    )
    config = {}
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
        reward_multiplier=1.0,  # gym locomotion returns are already large
        gamma=0.99,  # locomotion policies don't need long horizons - fall back to the default
        tau=0.005,
    )
    exploration_type = switch_exploration(config, "egreedy", steps_anneal=500_000)
    use_config(config, args.configs)

    group_name = args.run_name
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.policy_seq_len,
            traj_save_len=args.policy_seq_len * 6,
            run_name=run_name,
            tstep_encoder_type=amago.nets.tstep_encoders.FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_episodes_per_actor * 1001,
            grad_clip=2.0,
            learning_rate=3e-4,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
