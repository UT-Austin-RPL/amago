import random

import gymnasium as gym
import numpy as np

import amago
from amago.envs.builtin.gym_envs import GymEnv
from example_utils import *


class PendulumRandomGravity(gym.Env):
    def __init__(
        self, gravity_range: tuple[float, float], timesteps_per_episode: int = 100
    ):
        self.gravity_range = gravity_range
        self.timesteps_per_episode = timesteps_per_episode
        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *args, **kwargs):
        self.current_gravity = random.uniform(*self.gravity_range)
        self.episode_timestep = 0
        self.env = gym.make("Pendulum-v1", g=self.current_gravity)
        return self.env.reset()

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.episode_timestep += 1
        if self.episode_timestep >= self.timesteps_per_episode or (
            terminated or truncated
        ):
            next_state, info = self.env.reset()
            self.episode_timestep = 0
        return next_state, reward, False, False, info


def add_cli(parser):
    parser.add_argument(
        "--experiment",
        choices=["train-test-gap", "no-memory", "memory-rnn", "memory-transformer"],
    )
    parser.add_argument("--horizon", "-H", type=int, default=200)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--buffer_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()

    if args.log:
        import wandb

    config = {}
    turn_off_goal_conditioning(config)

    if args.experiment == "memory-rnn":
        traj_encoder = "rnn"
    elif args.experiment == "memory-transformer":
        traj_encoder = "transformer"
    else:
        traj_encoder = "ff"
    switch_tstep_encoder(config, arch="ff", n_layers=1, d_hidden=64, d_output=64)
    switch_traj_encoder(
        config,
        arch=traj_encoder,
        memory_size=128,
        layers=3,
    )
    use_config(config)

    gravity_range_train = (2.0, 18.0) if "memory" in args.experiment else (10.0, 10.01)
    gravity_range_test = (2.0, 18.0) if "memory" in args.experiment else (5.0, 5.01)

    group_name = f"{args.run_name}_{args.experiment}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        make_train_env = lambda: GymEnv(
            PendulumRandomGravity(gravity_range_train),
            env_name=f"pendulum_g({gravity_range_train})",
            horizon=args.horizon,
            zero_shot=True,
        )

        make_val_env = lambda: GymEnv(
            PendulumRandomGravity(gravity_range_test),
            env_name=r"pendulum_g({gravity_range_test})",
            horizon=args.horizon,
            zero_shot=True,
        )

        experiment = amago.Experiment(
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            max_seq_len=args.horizon,
            traj_save_len=args.horizon,
            dset_max_size=5_000,
            run_name=run_name,
            dset_name=run_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dloader_workers=8,
            log_to_wandb=args.log,
            wandb_group_name=group_name,
            epochs=150,
            parallel_actors=24,
            train_timesteps_per_epoch=250,
            train_grad_updates_per_epoch=500,
            val_interval=10,
            val_timesteps_per_epoch=1000,
            ckpt_interval=25,
            async_envs=False,
        )

        experiment.start()
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_val_env, timesteps=10_000)
        wandb.finish()
