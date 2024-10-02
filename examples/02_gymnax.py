from argparse import ArgumentParser
from functools import partial

import gymnax
from gymnax.wrappers import GymnaxToVectorGymWrapper

import jax
import wandb
import numpy as np

from amago.envs import AMAGOEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--eval_timesteps", type=int, default=1000)
    parser.add_argument("--jax_device_idx", type=int, default=None)
    return parser


class GymnaxCompatibility(GymnaxToVectorGymWrapper):
    """
    Gymnax wants to give us the batched observation and action spaces,
    but AMAGO is expecting unbatched spaces.
    It's also going to send out jax arrays, but we need numpy.

    A key point is that this only works because gymnax envs automatically reset.
    """

    @property
    def observation_space(self):
        return self.single_observation_space

    @property
    def action_space(self):
        return self.single_action_space

    def reset(self, *args, **kwargs):
        obs, info = super().reset()
        obs = np.array(obs)
        return obs, info

    def step(self, action):
        obs, rewards, te, tr, info = super().step(action)

        # Convert jax arrays to numpy
        # We could probably get jax arrays to work, but it would
        # fail a lot of assert checks, and the data's going to be
        # saved to disk as numpy and loaded as torch anyway.
        obs = np.array(obs)
        rewards = np.array(rewards)
        te = np.array(te)
        tr = np.array(tr)
        return obs, rewards, te, tr, info


def make_gymnax_amago(env_name, parallel_envs):
    env, params = gymnax.make(env_name)
    vec_env = GymnaxCompatibility(env, num_envs=parallel_envs, params=params)
    # when the environment is already vectorized, alert the AMAGOEnv wrapper with `batched_envs`
    return AMAGOEnv(
        env=vec_env, env_name=f"gymnax_{env_name}", batched_envs=parallel_envs
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # "already_vectorized" will stop the training loop from trying spawn multiple instances of the env
    args.env_mode = "already_vectorized"

    # config
    config = {}
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    # free speedup by doing env computation on a spare GPU
    # e.g. `CUDA_VISIBLE_DEVICES=1,2 python ... --jax_device_idx 1` will put AMAGO training on gpu id 1, gymnax on gpu id 2
    jax_device = (
        jax.devices("cpu")[0]
        if args.jax_device_idx is None
        else jax.devices()[args.jax_device_idx]
    )

    with jax.default_device(jax_device):
        make_env = partial(
            make_gymnax_amago, env_name=args.env, parallel_envs=args.parallel_actors
        )
        group_name = f"{args.run_name}_{args.env}"
        for trial in range(args.trials):
            run_name = group_name + f"_trial_{trial}"
            experiment = create_experiment_from_cli(
                args,
                make_train_env=make_env,
                make_val_env=make_env,
                max_seq_len=args.max_seq_len,
                traj_save_len=args.max_seq_len * 4,
                run_name=run_name,
                group_name=group_name,
                val_timesteps_per_epoch=args.eval_timesteps,
            )
            experiment = switch_async_mode(experiment, args)
            experiment.start()
            if args.ckpt is not None:
                experiment.load_checkpoint(args.ckpt)
            experiment.learn()
            experiment.evaluate_test(make_env, timesteps=10_000, render=False)
            experiment.delete_buffer_from_disk()
            wandb.finish()
