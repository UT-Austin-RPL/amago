from argparse import ArgumentParser

import wandb

from amago.envs.builtin.popgym_envs import POPGymAMAGO, MultiDomainPOPGymAMAGO
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, default="AutoencodeEasy")
    parser.add_argument("--max_seq_len", type=int, default=2000)
    parser.add_argument(
        "--multidomain",
        action="store_true",
        help="Activate 'MultiDomain' POPGym, where agents play 27 POPGym games at the same time in 1-shot format (2 episodes, second one counts).",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -1.0,  # paper: None
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 1.0,  # paper: None
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 32,  # paper: 64
    }
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,  # paper: 256
        layers=args.memory_layers,  # paper: 3
    )
    tstep_encoder_type = switch_tstep_encoder(
        config, arch="ff", n_layers=2, d_hidden=512, d_output=200
    )
    agent_type = switch_agent(config, args.agent_type, reward_multiplier=100.0)
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
            max_seq_len=args.max_seq_len,
            traj_save_len=2000,
            group_name=group_name,
            run_name=run_name,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
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
