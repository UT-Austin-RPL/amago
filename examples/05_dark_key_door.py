from argparse import ArgumentParser

import wandb

from amago.envs.builtin.toy_gym import RoomKeyDoor
from amago.envs import AMAGOEnv
from amago.envs.exploration import BilevelEpsilonGreedy
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--meta_horizon",
        type=int,
        default=500,
        help="Total meta-adaptation timestep budget for the agent to explore the same room layout.",
    )
    parser.add_argument(
        "--room_size",
        type=int,
        default=8,
        help="Size of the room. Exploration is sparse and difficulty scales quickly with room size.",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=50,
        help="Maximum length of a single episode in the environment.",
    )
    parser.add_argument(
        "--light_room_observation",
        action="store_true",
        help="Demonstrate how meta-RL relies on partial observability by revealing the goal location as part of the observation. This version of the environment can be solved without memory!",
    )
    parser.add_argument(
        "--randomize_actions",
        action="store_true",
        help="Randomize the agent's action space to make the task harder.",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "BilevelEpsilonGreedy.steps_anneal": 500_000,
        "BilevelEpsilonGreedy.rollout_horizon": args.meta_horizon,
    }
    tstep_encoder_type = switch_tstep_encoder(
        config, arch="ff", n_layers=2, d_hidden=128, d_output=64
    )
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    agent_type = switch_agent(config, args.agent_type, reward_multiplier=100.0)
    use_config(config, args.configs)

    group_name = f"{args.run_name}_dark_key_door"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        make_train_env = lambda: AMAGOEnv(
            env=RoomKeyDoor(
                size=args.room_size,
                max_episode_steps=args.episode_length,
                meta_rollout_horizon=args.meta_horizon,
                dark=not args.light_room_observation,
                randomize_actions=args.randomize_actions,
            ),
            env_name=f"Dark-Key-To-Door-{args.room_size}x{args.room_size}",
        )
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.meta_horizon,
            traj_save_len=args.meta_horizon,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=args.meta_horizon * 4,
            # the fancier exploration schedule mentioned in the appendix can help
            # when the domain is a true meta-RL problem and the "horizon" time limit
            # (above) is actually relevant for resetting the task.
            exploration_wrapper_type=BilevelEpsilonGreedy,
        )
        switch_async_mode(experiment, args)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
