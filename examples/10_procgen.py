from argparse import ArgumentParser

import wandb
import procgen
import gym

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.envs.builtin.ale_retro import ALE, AtariAMAGOWrapper
from utils import *


def add_cli(parser):
    parser.add_argument("--env", default="coinrun")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--start_after_epochs", type=int, default=25)
    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--slow_inference", action="store_true")
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_cli(parser)
    add_common_cli(parser)
    args = parser.parse_args()

    config = {}
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="cnn", channels_first=True)
    if args.use_aug:
        config.update({"amago.nets.tstep_encoders.CNNTstepEncoder.aug_Cls" : amago.nets.cnn.DrQv2Aug})
    if args.naive:
        naive(config)
    use_config(config, args.configs)

    """
    make_env = lambda: GymEnv(
        gym.make(f"procgen-{args.env}-v0", distribution_mode="easy"),
        env_name=args.env,
        horizon=108_000,
        zero_shot=True,
        convert_from_old_gym=True,
    )
    """
    make_env = lambda : AtariAMAGOWrapper(ALE([args.env], use_discrete_actions=True))

    group_name = f"{args.run_name}_{args.env}_atari_l_{args.max_seq_len}"
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"
        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            dset_max_size=args.dset_max_size,
            run_name=dset_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dset_name=dset_name,
            log_to_wandb=not args.no_log,
            epochs=args.epochs,
            parallel_actors=args.parallel_actors,
            start_learning_after_epoch=args.start_after_epochs,
            train_timesteps_per_epoch=args.timesteps_per_epoch,
            train_grad_updates_per_epoch=args.grads_per_epoch,
            val_interval=args.val_interval,
            val_timesteps_per_epoch=10_000,
            ckpt_interval=args.ckpt_interval,
            fast_inference=not args.slow_inference,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_env, timesteps=50_000, render=False)
        wandb.finish()
