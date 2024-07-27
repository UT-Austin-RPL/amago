from argparse import ArgumentParser
from copy import deepcopy
from functools import partial

import wandb
import gym

import amago
from amago.envs.builtin.ale_retro import AtariAMAGOWrapper, AtariGame
from amago.nets.cnn import NatureishCNN, IMPALAishCNN
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--max_seq_len", type=int, default=80)
    parser.add_argument(
        "--cnn", type=str, choices=["nature", "impala"], default="nature"
    )
    return parser


DEFAULT_MULTIGAME_LIST = [
    "Pong",
    "Boxing",
    "Breakout",
    "Gopher",
    "MsPacman",
    "ChopperCommand",
    "CrazyClimber",
    "BattleZone",
    "Qbert",
    "Seaquest",
]

ATARI_TIME_LIMIT = (30 * 60 * 60) // 5  # (30 minutes of game time)


def make_atari_game(game_name):
    return AtariAMAGOWrapper(
        AtariGame(game=game_name, use_discrete_actions=True), horizon=ATARI_TIME_LIMIT
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    add_common_cli(parser)
    args = parser.parse_args()

    config = {"amago.agent.Agent.reward_multiplier": 0.25}
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )

    if args.cnn == "nature":
        cnn_type = NatureishCNN
    elif args.cnn == "impala":
        cnn_type = IMPALAishCNN

    switch_tstep_encoder(config, arch="cnn", cnn_Cls=cnn_type, channels_first=True)
    use_config(config, args.configs)

    # Episode lengths in Atari vary widely across games, so we manually set actors
    # to a specific game so that all games are always played in parallel.
    games = args.games or DEFAULT_MULTIGAME_LIST
    assert (
        args.parallel_actors % len(games) == 0
    ), "Number of actors must be divisible by number of games."
    env_funcs = []
    for actor in range(args.parallel_actors):
        game_name = games[actor % len(games)]
        env_funcs.append(partial(make_atari_game, game_name))

    group_name = (
        f"{args.run_name}_{','.join(games)}_atari_l_{args.max_seq_len}_cnn_{args.cnn}"
    )
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=env_funcs,
            make_val_env=env_funcs,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 3,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=ATARI_TIME_LIMIT,
            save_trajs_as="npz-compressed",
        )
        switch_mode_load_ckpt(experiment, args)
        experiment.start()
        experiment.learn()
        experiment.evaluate_test(
            env_funcs, timesteps=ATARI_TIME_LIMIT * 5, render=False
        )
        wandb.finish()
