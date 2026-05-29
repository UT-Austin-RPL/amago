from argparse import ArgumentParser

import wandb

from amago.envs.builtin.toy_gym import RoomKeyDoor
from amago.envs import AMAGOEnv
from amago import cli_utils


def add_cli(parser):
    parser.add_argument(
        "--k_episodes",
        type=int,
        default=8,
        help="Number of episodes per meta-rollout. Effective sequence length = k_episodes * episode_length.",
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
    parser.add_argument(
        "--finite_horizon",
        action="store_true",
        help="Use finite-horizon mode: include time in observations and signal meta-done as terminated. Default is infinite-horizon (no time in obs, meta-done as truncated).",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    cli_utils.add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {}
    tstep_encoder_type = cli_utils.switch_tstep_encoder(
        config,
        arch="ff",
        n_layers=2,
        d_hidden=128,
        d_output=64,
        specify_obs_keys=["observed", "prev_action", "prev_reward"],
        hide_rl2s=True,
        normalize_inputs=False,
    )
    traj_encoder_type = cli_utils.switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
        pos_emb="rope",
    )
    agent_type = cli_utils.switch_agent(
        config, args.agent_type, reward_multiplier=100.0
    )
    horizon_type = "finite" if args.finite_horizon else "infinite"
    dummy_env = RoomKeyDoor(
        size=args.room_size,
        max_episode_steps=args.episode_length,
        k_episodes=args.k_episodes,
        horizon_type=horizon_type,
    )
    meta_horizon = dummy_env.meta_horizon
    args.timesteps_per_epoch = meta_horizon
    # the fancier exploration schedule mentioned in the appendix can help
    # when the domain is a true meta-RL problem and the "horizon" time limit
    # (above) is actually relevant for resetting the task.
    exploration_type = cli_utils.switch_exploration(
        config, "bilevel", steps_anneal=500_000, rollout_horizon=meta_horizon
    )
    cli_utils.use_config(config, args.configs)

    group_name = f"{args.run_name}_dark_key_door"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        make_train_env = lambda: AMAGOEnv(
            env=RoomKeyDoor(
                size=args.room_size,
                max_episode_steps=args.episode_length,
                k_episodes=args.k_episodes,
                dark=not args.light_room_observation,
                randomize_actions=args.randomize_actions,
                horizon_type=horizon_type,
            ),
            env_name=f"Dark-Key-To-Door-{args.room_size}x{args.room_size}-{horizon_type}",
        )
        experiment = cli_utils.create_experiment_from_cli(
            args,
            agent_type=agent_type,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=meta_horizon,
            traj_save_len=meta_horizon * 10,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=meta_horizon * 4,
            exploration_wrapper_type=exploration_type,
            stagger_traj_file_lengths=False,
            wandb_project="z-room-key-door",
        )
        experiment = cli_utils.switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
