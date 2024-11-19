"""
Support for gymnax is experimental and mainly meant to test the already_vectorized 
env API used by XLand MiniGrid (an unsolved environment) with classic gym envs. 
Many of the gymnax envs appear to be broken by recent versions of jax.
There are a couple memory/meta-RL bsuite envs where AMAGO+Transformer
is significantly better than the gymnax reference scores though.
"""

import os

# stop jax from stealing pytorch's memory, since we're only using it for the envs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
import math
from functools import partial

import gymnax
import torch
import jax
import wandb
import numpy as np

from amago.envs import AMAGOEnv
from amago.envs.builtin.gymnax_envs import GymnaxCompatibility
from amago.nets.cnn import GridworldCNN
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--eval_timesteps", type=int, default=1000)
    return parser


def make_gymnax_amago(env_name, parallel_envs):
    env, params = gymnax.make(env_name)
    vec_env = GymnaxCompatibility(env, num_envs=parallel_envs, params=params)
    # when the environment is already vectorized, alert the AMAGOEnv wrapper with `batched_envs`
    return AMAGOEnv(
        env=vec_env, env_name=f"gymnax_{env_name}", batched_envs=parallel_envs
    )


def guess_tstep_encoder(config, obs_shape):
    """
    We'll move past the somewhat random collection of gymnax envs by making up a simple
    timestep encoder based on a few hacks. If we really cared about gymnax performance we
    could tune this per environment.
    """
    if len(obs_shape) == 3:
        print(f"Guessing CNN for observation of shape {obs_shape}")
        channels_first = np.argmin(obs_shape).item() == 0
        return switch_tstep_encoder(
            config,
            "cnn",
            cnn_type=GridworldCNN,
            channels_first=channels_first,
        )
    else:
        print(f"Guessing MLP for observation of shape {obs_shape}")
        dim = math.prod(obs_shape)  # FFTstepEncoder will flatten the obs on input
        return switch_tstep_encoder(
            config,
            "ff",
            d_hidden=max(dim // 3, 128),
            n_layers=2,
            d_output=max(dim // 4, 96),
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
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        test_env, env_params = gymnax.make(args.env)
        test_obs_shape = test_env.observation_space(env_params).shape
    tstep_encoder_type = guess_tstep_encoder(config, test_obs_shape)
    agent_type = switch_agent(config, args.agent_type)

    use_config(config, args.configs)
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
            traj_save_len=args.max_seq_len * 20,
            run_name=run_name,
            agent_type=agent_type,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_timesteps,
            grad_clip=2.0,
            l2_coeff=1e-4,
            save_trajs_as="npz-compressed",
        )
        experiment = switch_async_mode(experiment, args.mode)
        amago_device = experiment.DEVICE.index or torch.cuda.current_device()
        env_device = jax.devices("gpu")[amago_device]
        with jax.default_device(env_device):
            experiment.start()
            if args.ckpt is not None:
                experiment.load_checkpoint(args.ckpt)
            experiment.learn()
            experiment.evaluate_test(make_env, timesteps=10_000, render=False)
            experiment.delete_buffer_from_disk()
            wandb.finish()
