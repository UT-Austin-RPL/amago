from argparse import ArgumentParser

import wandb

from amago.envs.builtin.popgym_envs import POPGymAMAGO, MultiDomainPOPGymAMAGO
from amago.agent import binary_filter
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

    # whenever we need a "max rollout length" value, we use this arbitrarily large number
    artificial_horizon = max(args.max_seq_len, 2000)

    # fmt: off
    config = {
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": None,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": None,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 64,
        "binary_filter.threshold": 1e-3, # not important
        # learnable position embedding
        "amago.nets.transformer.LearnablePosEmb.max_time_idx": artificial_horizon,
        "amago.nets.traj_encoders.TformerTrajEncoder.pos_emb": "learnable",
        "amago.nets.policy_dists.Discrete.clip_prob_high": 1.0, # not important
        "amago.nets.policy_dists.Discrete.clip_prob_low": 1e-6, # not important
        # paper version defaulted to large set of gamma values
        "amago.agent.Multigammas.discrete": [0.1, 0.7, 0.9, 0.93, 0.95, 0.98, 0.99, 0.992, 0.994, 0.995, 0.997, 0.998, 0.999, 0.9991, 0.9992, 0.9993, 0.9994, 0.9995],
    }
    # fmt: on

    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,  # paper: 256
        layers=args.memory_layers,  # paper: 3
    )
    tstep_encoder_type = switch_tstep_encoder(
        config,
        arch="ff",
        n_layers=2,
        d_hidden=512,
        d_output=200,
    )
    agent_type = switch_agent(
        config,
        args.agent_type,
        reward_multiplier=200.0 if args.multidomain else 100.0,
        tau=0.0025,
    )
    # steps_anneal can safely be set much lower (<500k) in most tasks. More sweeps needed.
    exploration_type = switch_exploration(
        config,
        "egreedy",
        steps_anneal=1_000_000,
    )
    use_config(config, args.configs)

    group_name = f"{args.run_name}_{args.env}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        if args.multidomain:
            make_train_env = lambda: MultiDomainPOPGymAMAGO()
        else:
            # in order to match the pre-gymnasium version of popgym (done instead of terminated/truncated),
            # we need to set terminated = terminated or truncated
            make_train_env = lambda: POPGymAMAGO(
                f"popgym-{args.env}-v0", truncated_is_done=True
            )
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=artificial_horizon,
            group_name=group_name,
            run_name=run_name,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_type,
            agent_type=agent_type,
            val_timesteps_per_epoch=artificial_horizon,
            learning_rate=1e-4,
            grad_clip=1.0,
            lr_warmup_steps=2000,
        )
        experiment = switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
