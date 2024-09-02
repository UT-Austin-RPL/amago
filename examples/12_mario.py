from argparse import ArgumentParser
import random
import math
from functools import partial

import wandb
import gymnasium as gym
import numpy as np

import amago
from amago.envs.builtin.ale_retro import RetroAMAGOWrapper, RetroArcade
from amago.envs.env_utils import EpsilonGreedy, ExplorationWrapper
from amago.nets.cnn import NatureishCNN, IMPALAishCNN
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--game_group",
        type=str,
        choices=["sm1", "sm2", "sm3", "smw", "all"],
        default="all",
    )
    parser.add_argument("--use_discrete_actions", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument(
        "--cnn", type=str, choices=["nature", "impala"], default="impala"
    )
    return parser


# fmt: off
MARIO_TRAIN = {
"SuperMarioBros-Nes": ["Level1-1", "Level2-1-clouds-easy", "Level4-1", "Level5-1", "Level7-1", "Level8-1"],
"SuperMarioBros2Japan-Nes": ["Level1-1", "Level1-2", "Level3-1", "Level4-1", "Level5-1", "Level6-2", "Level7-1", "Level8-1"],
"SuperMarioBros3-Nes": ["1Player.World1.Fortress", "1Player.World1.FortressBoss", "1Player.World1.Level3"],
"SuperMarioWorld-Snes": ["Bridges1", "Bridges2", "ChocolateIsland1", "ChocolateIsland2", "ChocolateIsland3", "DonutPlains1", "DonutPlains2", "DonutPlains3", "DonutPlains4", "DonutPlains5", "Forest1", "Forest2", "Forest3", "Forest4", "Forest5", "Start", "VanillaDome2", "VanillaDome3", "VanillaDome4", "VanillaDome5", "YoshiIsland1", "YoshiIsland2", "YoshiIsland4"], 
"SuperMarioWorld2-Snes": ["Start"]
} 

MARIO_TEST = {
"SuperMarioBros-Nes": ["Level1-4", "Level2-1", "Level2-1-clouds", "Level3-1", "Level6-1"],
"SuperMarioBros2Japan-Nes": ["Level2-1", "Level6-1"],
"SuperMarioBros3-Nes": ["1Player.World1.Castle", "1Player.World1.Level1", "1Player.World1.Level1.HammerBros"],
"SuperMarioWorld-Snes": ["VanillaDome1", "YoshiIsland3"]
}
# fmt: on


@gin.configurable
class MultiBinaryExploration(ExplorationWrapper):
    def __init__(
        self,
        env: gym.Env,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        flip: float = 0.15,
        steps_anneal: int = 500_000,
    ):
        super().__init__(env)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.flip = flip
        self.eps_slope = (eps_start - eps_end) / steps_anneal
        self.multibinary = isinstance(self.env.action_space, gym.spaces.MultiBinary)
        assert self.multibinary
        self.global_step = 0

    def current_eps(self, local_step: int, horizon: int):
        eps = max(self.eps_start - self.eps_slope * self.global_step, self.eps_end)
        current = self.global_multiplier * eps
        return current

    def add_exploration_noise(self, action: np.ndarray, local_step: int, horizon: int):
        noise = self.current_eps(local_step, horizon)
        use_random = random.random() <= noise
        should_flip = np.random.random(*action.shape) <= self.flip
        expl_action = (1 - should_flip) * action + should_flip * np.invert(
            action.astype(bool)
        )
        use_action = ((1 - use_random) * action + use_random * expl_action).astype(
            np.uint8
        )
        return use_action


FRAME_SKIP = 8
TIME_LIMIT_MINUTES = 8

mins_to_steps = lambda m: math.ceil(m * 60 * 60 / FRAME_SKIP)
steps_to_mins = lambda s: math.ceil((s * FRAME_SKIP) / 60 * 60)


def make_mario_games(game_group: str, train: bool, discrete_actions: bool):
    game_dict = MARIO_TRAIN if train else MARIO_TEST
    wrap_dict = lambda *games: {game: game_dict[game] for game in games}
    if game_group == "sm1":
        game_states = wrap_dict(["SuperMarioBros-Nes"])
    elif game_group == "sm2":
        game_states = wrap_dict(["SuperMarioBros2Japan-Nes"])
    elif game_group == "sm3":
        game_states = wrap_dict(["SuperMarioBros3-Nes"])
    elif game_group == "smw":
        game_states = wrap_dict(["SuperMarioWorld-Nes", "SuperMarioWorld2-Snes"])
    elif game_group == "all":
        game_states = game_dict
    return RetroAMAGOWrapper(
        RetroArcade(
            game_start_dict=game_states,
            use_discrete_actions=discrete_actions,
            time_limit_minutes=TIME_LIMIT_MINUTES,
            frame_skip=FRAME_SKIP,
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    add_common_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.agent.Agent.reward_multiplier": 0.5,
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -10_000,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 60_000,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 96,
        "amago.agent.Agent.online_coeff": 0.0,
        "amago.agent.Agent.offline_coeff": 1.0,
        "amago.learning.Experiment.exploration_wrapper_Cls": (
            EpsilonGreedy if args.use_discrete_actions else MultiBinaryExploration
        ),
        "amago.nets.traj_encoders.TformerTrajEncoder.pos_emb": "fixed",
    }
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

    make_train_envs = partial(
        make_mario_games,
        game_group=args.game_group,
        train=True,
        discrete_actions=args.use_discrete_actions,
    )
    make_val_envs = partial(
        make_mario_games,
        game_group=args.game_group,
        train=True,
        discrete_actions=args.use_discrete_actions,
    )
    make_test_envs = partial(
        make_mario_games,
        game_group=args.game_group,
        train=False,
        discrete_actions=args.use_discrete_actions,
    )

    group_name = f"{args.run_name}_mario_{args.game_group}_l_{args.max_seq_len}_cnn_{args.cnn}_{'MT' if args.agent_type == 'multitask' else 'ST'}"
    print(group_name)
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_envs,
            make_val_env=make_val_envs,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 3,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=mins_to_steps(TIME_LIMIT_MINUTES) * 2,
            save_trajs_as="npz-compressed",
            learning_rate=3e-4,
            grad_clip=2.0,
        )
        switch_async_mode(experiment, args)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(
            make_test_envs,
            timesteps=mins_to_steps(TIME_LIMIT_MINUTES) * 5,
            render=False,
        )
        experiment.delete_buffer_from_disk()
        wandb.finish()
