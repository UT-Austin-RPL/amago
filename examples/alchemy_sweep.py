import shutil
import random
from argparse import ArgumentParser
from functools import partial
import os

import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.envs.builtin.alchemy import SymbolicAlchemy
from example_utils import *


sweep_config = {
    "method": "grid",
    "metric": {"name": "test/avg_return", "goal": "maximize"},
    "parameters": {
        "layers": {"values": [3, 5, 8]},
        "memory_size": {"values": [256, 400]},
        "grads_per_epoch": {"values": [500, 1000, 2000]},
        "exploration_schedule_steps": {"values": [100_000, 1_000_000]},
        "start_learning_at_epoch": {"values": [0, 50]},
        "max_buffer_size": {"values": [10_000, 100_000]},
        "max_lr": {"values": [1e-4, 5e-4]},
        "offline_coeff": {"values": [0.0, 0.1]},
    },
}


def train(cli_args=None):
    with wandb.init(config=None):
        wandb_config = wandb.config

        gin_config = {
            "amago.envs.env_utils.ExplorationWrapper.steps_anneal": wandb_config.exploration_schedule_steps,
            "amago.agent.Agent.offline_coeff": wandb_config.offline_coeff,
        }
        turn_off_goal_conditioning(gin_config)
        switch_traj_encoder(
            gin_config,
            arch="transformer",
            memory_size=wandb_config.memory_size,
            layers=wandb_config.layers,
        )
        use_config(gin_config, finalize=False)
        make_train_env = lambda: GymEnv(
            gym_env=SymbolicAlchemy(),
            env_name="dm_symbolic_alchemy",
            horizon=201,
            zero_shot=True,
        )
        buffer_dir = cli_args.buffer_dir
        run_name = f"symbolic_dm_alchemy_SWEEP_{random.randint(0, 100_000)}"
        experiment = amago.Experiment(
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=200,
            traj_save_len=200,
            dset_max_size=wandb_config.max_buffer_size,
            run_name=run_name,
            dset_name=run_name,
            gpu=cli_args.gpu,
            learning_rate=wandb_config.max_lr,
            start_learning_at_epoch=wandb_config.start_learning_at_epoch,
            dset_root=buffer_dir,
            dloader_workers=8,
            log_to_wandb=True,
            epochs=2,
            parallel_actors=36,
            train_timesteps_per_epoch=201,
            train_grad_updates_per_epoch=wandb_config.grads_per_epoch,
            val_interval=1,
            val_timesteps_per_epoch=201 * 2,
            ckpt_interval=100,
            async_envs=False,
        )
        experiment.start()
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_train_env, timesteps=1_200, render=False)
        shutil.rmtree(os.path.join(buffer_dir, run_name, "train"))
        shutil.rmtree(os.path.join(buffer_dir, run_name, "val"))
        shutil.rmtree(os.path.join(buffer_dir, run_name, "ckpts"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--buffer_dir")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--join_sweep", default=None)
    cli_args = parser.parse_args()

    if cli_args.join_sweep is None:
        sweep_id = wandb.sweep(sweep_config, project="alchemy_sweep")
    else:
        sweep_id = cli_args.join_sweep
    wandb.agent(
        sweep_id, partial(train, cli_args=cli_args), count=100, project="alchemy_sweep"
    )
