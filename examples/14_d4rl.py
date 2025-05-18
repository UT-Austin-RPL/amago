from argparse import ArgumentParser
import random

import torch
import gym as og_gym
import d4rl
import gymnasium as gym
import wandb
import numpy as np

import amago
from amago.envs import AMAGOEnv
from amago.cli_utils import *
from amago.loading import RLData, RLDataset
from amago.nets.policy_dists import TanhGaussian, GMM


def add_cli(parser):
    parser.add_argument(
        "--env", type=str, required=True, help="Environment/Dataset name"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=32, help="Policy sequence length."
    )
    parser.add_argument(
        "--policy_dist",
        type=str,
        default="TanhGaussian",
        help="Policy distribution type",
        choices=["TanhGaussian", "GMM"],
    )
    parser.add_argument(
        "--eval_timesteps",
        type=int,
        default=1000,
        help="Number of timesteps to evaluate for each actor. Will be overridden if the environment has a known time limit.",
    )
    return parser


class D4RLDataset(RLDataset):
    def __init__(self, d4rl_dset: dict[str, np.ndarray]):
        super().__init__()
        self.d4rl_dset = d4rl_dset
        self.episode_ends = np.where(d4rl_dset["terminals"] | d4rl_dset["timeouts"])[0]

    def get_description(self) -> str:
        return "D4RL"

    @property
    def save_new_trajs_to(self):
        # disables saving new amago trajectories to disk
        return None

    def sample_random_trajectory(self):
        # pick a random episode
        episode_idx = random.randrange(0, len(self.episode_ends) - 1)
        s = self.episode_ends[episode_idx] + 1
        e = self.episode_ends[episode_idx + 1] + 1
        traj_len = e - s
        """
        Slice an episode, with off-by-one action/reward/terminal alignment,
        from D4RL.

        In D4RL qlearning_dataset, the last transition of an episode is:
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i]) # True
        data = (obs, action, reward, new_obs, done_bool)
        """
        obs_np = self.d4rl_dset["observations"][s:e]
        actions_np = self.d4rl_dset["actions"][s : e - 1]
        rewards_np = self.d4rl_dset["rewards"][s : e - 1]
        terminals_np = self.d4rl_dset["terminals"][s + 1 : e]
        timeouts_np = self.d4rl_dset["timeouts"][s + 1 : e]

        # convert to torch, adding time_idxs and the "rl2" prev action/rew features
        obs = {"observation": torch.from_numpy(obs_np)}
        actions = torch.from_numpy(actions_np).float()
        rewards = torch.from_numpy(rewards_np).float().unsqueeze(-1)
        time_idxs = torch.arange(traj_len).unsqueeze(-1).long()
        dones = torch.from_numpy(terminals_np).bool().unsqueeze(-1)

        return RLData(
            obs=obs,
            actions=actions,
            rews=rewards,
            dones=dones,
            time_idxs=time_idxs,
        )


from amago.envs.env_utils import space_convert
from amago.envs.amago_env import AMAGO_ENV_LOG_PREFIX


class D4RLGymEnv(gym.Env):
    """
    Light wrapper that logs the D4RL normalized return and handles
    the gym/gymnasium conversion while we're at it.
    """

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.env = og_gym.make(env_name)
        self.action_space = space_convert(self.env.action_space)
        self.observation_space = space_convert(self.env.observation_space)
        if isinstance(self.env, og_gym.wrappers.TimeLimit):
            self.time_limit = self.env._max_episode_steps
        else:
            self.time_limit = None

    @property
    def dset(self):
        return self.env.get_dataset()

    def reset(self, *args, **kwargs):
        self.episode_return = 0
        return self.env.reset(*args, **kwargs), {}

    def step(self, action):
        s, r, d, i = self.env.step(action)
        self.episode_return += r
        if d:
            i[f"{AMAGO_ENV_LOG_PREFIX} D4RL Normalized Return"] = (
                d4rl.get_normalized_score(self.env_name, self.episode_return).item()
            )
        return s, r, d, d, i


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # ues env to set some args
    env_name = args.env
    example_env = D4RLGymEnv(args.env)
    assert isinstance(
        example_env.action_space, gym.spaces.Box
    ), "Only supports continuous action spaces"
    if args.timesteps_per_epoch > 0:
        print("WARNING: timesteps_per_epoch is not supported for D4RL, setting to 0")
        args.timesteps_per_epoch = 0

    if example_env.time_limit is not None:
        print(
            f"WARNING: time limit is {example_env.time_limit}, setting eval_timesteps"
        )
        args.eval_timesteps = example_env.time_limit + 1

    # create dataset
    dataset = D4RLDataset(d4rl_dset=example_env.dset)

    # setup environment
    make_train_env = lambda: AMAGOEnv(
        D4RLGymEnv(args.env),
        env_name=env_name,
        batched_envs=1,
    )

    # agent architecture: drop everything down to standard small sizes
    config = {
        "amago.nets.actor_critic.NCritics.d_hidden": 128,
        "amago.nets.actor_critic.NCriticsTwoHot.d_hidden": 128,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 48,
        "amago.nets.actor_critic.Actor.d_hidden": 128,
        "amago.nets.actor_critic.Actor.continuous_dist_type": eval(args.policy_dist),
    }
    tstep_encoder_type = switch_tstep_encoder(
        config,
        arch="ff",
        d_hidden=128,
        d_output=128,
        n_layers=1,
    )
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    agent_type = switch_agent(
        config,
        args.agent_type,
        online_coeff=0.0,
        offline_coeff=1.0,
        gamma=0.99,
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
            run_name=run_name,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_timesteps,
            learning_rate=1.5e-4,
            dataset=dataset,
            sample_actions=False,
        )
        experiment = switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=10_000, render=False)
        wandb.finish()
