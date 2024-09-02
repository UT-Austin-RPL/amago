from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.popgym_envs import POPGymAMAGO, MultiDomainPOPGymAMAGO
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, default="AutoencodeEasy")
    parser.add_argument("--max_seq_len", type=int, default=2000)
    parser.add_argument("--traj_save_len", type=int, default=2000)
    parser.add_argument(
        "--multidomain",
        action="store_true",
        help="Activate 'MultiDomain' POPGym, where agents play 27 POPGym games at the same time in 1-shot format (2 episodes, second one counts).",
    )
    parser.add_argument("--naive", action="store_true")
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # bins are hard to tune in POPGym. The paper left the settings wide open, but it's actually
        # better to tighten the return limits and set a lower bin count because these envs have rapid
        # swings in Q vals. We don't think it really matters whether you do e.g. rewards x100, returns in [-100, 100] or
        # rewards x1, returns in [-1, 1], but the symlog mapping technically makes these different.
        "amago.agent.Agent.reward_multiplier": (
            1.0 if args.agent_type == "multitask" else 100.0
        ),  # paper: always 100
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -1.0,  # paper: None
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 1.0,  # paper: None
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 32,  # paper: 64
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
        run_name = group_name + f"_trial_{trial}"

        if args.multidomain:
            make_train_env = lambda: MultiDomainPOPGymAMAGO()
        else:
            make_train_env = lambda: POPGymAMAGO(f"popgym-{args.env}-v0")

        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            # For this one script to work across every environment,
            # these are arbitrary sequence limits we'll never each.
            max_seq_len=args.max_seq_len,
            traj_save_len=args.traj_save_len,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=2000,
        )
        experiment = switch_async_mode(experiment, args)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
