from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.gym_envs import POPGymEnv
from utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, default="AutoencodeEasy")
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--slow_inference", action="store_true")
    parser.add_argument("--half_precision", action="store_true")
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # no need to risk numerical instability when returns are this bounded
        "amago.agent.Agent.reward_multiplier": 100.0,
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        # NOTE: paper (and original POPGym results) use `memory_size=256`
        memory_size=args.memory_size,
        # NOTE: paper used layers=3
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="ff", n_layers=2, d_hidden=512, d_output=200)
    if args.naive:
        naive(config)
    use_config(config, args.configs)

    group_name = f"{args.run_name}_{args.env}"
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"
        make_train_env = lambda: POPGymEnv(f"popgym-{args.env}-v0")
        experiment = amago.Experiment(
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            # in POPGym the max_seq_len is an arbitrary limit we'll never reach
            max_seq_len=2000,
            traj_save_len=2000 + 1,
            dset_max_size=args.dset_max_size,  # paper used a larger size of 80_000
            run_name=dset_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dset_name=dset_name,
            log_to_wandb=not args.no_log,
            fast_inference=not args.slow_inference,
            epochs=args.epochs,
            parallel_actors=args.parallel_actors,  # paper used 24
            train_timesteps_per_epoch=args.timesteps_per_epoch,  # paper used 1000
            train_grad_updates_per_epoch=args.grads_per_epoch,  # paper used 1000
            val_interval=args.val_interval,
            val_timesteps_per_epoch=2_000,
            ckpt_interval=args.ckpt_interval,
            half_precision=args.half_precision,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        wandb.finish()
