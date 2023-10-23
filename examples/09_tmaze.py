from argparse import ArgumentParser

import gymnasium as gym
import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.envs.builtin.tmaze import TMazeAltPassive
from amago.envs.env_utils import ExplorationWrapper
from utils import *


def add_cli(parser):
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--slow_inference", action="store_true")
    return parser


@gin.configurable
class TMazeExploration(ExplorationWrapper):
    """
    The Tmaze environment is meant to evaluate recall over long context lengths without
    testing exploration, but it does this by requiring horizon - 1 deterministic actions
    to create a gap between the timestep that reveals the correct action and the timestep
    it is taken. This unintentionally creates a worst-case scenario for epsilon greedy
    exploration. We use this epsilon greedy exploration schedule to answer the
    central memory question while fixing the sample efficiency problems it creates.

    https://github.com/twni2016/Memory-RL/issues/1
    """

    def __init__(
        self,
        env,
        start_window=0,
        end_window=3,
        eps_start=1.0,
        eps_end=0.01,
        steps_anneal=100_000,
    ):
        self.start_window = start_window
        self.end_window = end_window
        super().__init__(
            env,
            eps_start_start=eps_start,
            eps_start_end=eps_start,
            eps_end_start=eps_end,
            eps_end_end=eps_end,
            steps_anneal=steps_anneal,
        )

    def current_eps(self, local_step: int, horizon: int):
        current = super().current_eps(local_step, horizon)
        if local_step > self.start_window and local_step < horizon - self.end_window:
            # low exploration during the easy corridor section; regular during the
            # interesting early and late timesteps.
            current = 0.5 / horizon
        return current


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # improve PopArt numerical stability
        "amago.agent.Agent.reward_multiplier": 100.0,
        # high discount
        "amago.agent.Agent.gamma": 0.9999,
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="ff", n_layers=2, d_hidden=128, d_output=128)
    use_config(config, args.configs)

    group_name = f"{args.run_name}_TMazePassive_H{args.horizon}"
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"

        make_env = lambda: GymEnv(
            TMazeAltPassive(corridor_length=args.horizon, penalty=-1.0 / args.horizon),
            env_name=f"TMazePassive-H{args.horizon}",
            horizon=args.horizon + 1,
            zero_shot=True,
        )

        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.horizon + 1,
            traj_save_len=args.horizon * 3,
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
            val_timesteps_per_epoch=args.horizon + 1,
            ckpt_interval=args.ckpt_interval,
            sample_actions=False,  # even softmax prob .999 isn't good enough for this env...
            exploration_wrapper_Cls=TMazeExploration,
            fast_inference=not args.slow_inference,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(
            make_train_env, timesteps=args.horizon * 5, render=False
        )
        wandb.finish()
