"""
Wind: a 2D point-robot meta-RL navigation task with hidden wind perturbations.

Tiny, fast meta-RL toy adapted from twni2016/pomdp-baselines. The agent
issues 2D velocity commands; each step the dynamics also add a hidden
per-task wind vector. The goal is fixed at (0, 1) and the reward is
sparse. A single hidden wind persists across ``k`` inner episodes (soft
resets between), so the policy must identify the wind from a few attempts
and exploit it on subsequent ones.

Mirrors the structure of ``12_half_cheetah_vel.py``: distinct
``--k_train_episodes`` and ``--k_eval_episodes`` so you can probe how
well the agent keeps adapting beyond its training horizon.
"""

from argparse import ArgumentParser

import wandb

import amago
from amago.envs import AMAGOEnv
from amago.envs.builtin.wind import WindEnv
from amago import cli_utils


def add_cli(parser):
    parser.add_argument(
        "--policy_seq_len",
        type=int,
        default=128,
        help="Policy sequence length. Default 128 fits a default training trial (4 inner * 75 steps = 300) plus padding.",
    )
    parser.add_argument(
        "--eval_episodes_per_actor",
        type=int,
        default=1,
        help="Validation episodes per parallel actor.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=75,
        help="Inner-episode horizon. 75 matches the original POMDP-baselines setting.",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=80,
        help="Number of unique winds in the discrete task set sampled from each wind seed. The training set and the held-out test set each contain n_tasks winds, drawn from disjoint RNG streams.",
    )
    parser.add_argument(
        "--train_wind_seed",
        type=int,
        default=1337,
        help="RNG seed for the training (and in-loop validation) wind set. 1337 matches the original POMDP-baselines convention.",
    )
    parser.add_argument(
        "--test_wind_seed",
        type=int,
        default=9999,
        help="RNG seed for the held-out test wind set used by experiment.evaluate_test() at the end of training. Should differ from --train_wind_seed for a true train/test split.",
    )
    parser.add_argument(
        "--k_train_episodes",
        type=int,
        default=4,
        help="Inner episodes per meta-trial during training. Default 4 (300 total steps).",
    )
    parser.add_argument(
        "--k_eval_episodes",
        type=int,
        default=10,
        help="Inner episodes per meta-trial during in-loop validation and held-out testing. Default 10 (750 total steps) probes how well the agent keeps adapting beyond its training horizon.",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    cli_utils.add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    def make_train_env():
        return AMAGOEnv(
            WindEnv(
                max_episode_steps=args.max_episode_steps,
                n_tasks=args.n_tasks,
                k_episodes=args.k_train_episodes,
                wind_seed=args.train_wind_seed,
            ),
            env_name="WindEnv-Train",
        )

    def make_val_env():
        # In-loop validation reuses the *training* wind set so the
        # validation curve tracks training progress rather than
        # generalization.
        return AMAGOEnv(
            WindEnv(
                max_episode_steps=args.max_episode_steps,
                n_tasks=args.n_tasks,
                k_episodes=args.k_eval_episodes,
                wind_seed=args.train_wind_seed,
            ),
            env_name="WindEnv-Train",
        )

    def make_test_env():
        # Held-out test set: a fresh, disjoint wind set sampled from a
        # different RNG seed. Used only by experiment.evaluate_test()
        # after training finishes.
        return AMAGOEnv(
            WindEnv(
                max_episode_steps=args.max_episode_steps,
                n_tasks=args.n_tasks,
                k_episodes=args.k_eval_episodes,
                wind_seed=args.test_wind_seed,
            ),
            env_name="WindEnv-Test",
        )

    config = {
        "amago.nets.traj_encoders.TformerTrajEncoder.pos_emb": "rope",
    }
    traj_encoder_type = cli_utils.switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    agent_type = cli_utils.switch_agent(
        config,
        args.agent_type,
        # Wind rewards are sparse 0/1; a successful inner episode tops out
        # at ``max_episode_steps`` reward. Boost the multiplier so the
        # critic sees signal comparable to dense-reward tasks.
        reward_multiplier=10.0,
        gamma=0.99,
        tau=0.005,
    )
    exploration_type = cli_utils.switch_exploration(
        config, "egreedy", steps_anneal=500_000
    )
    cli_utils.use_config(config, args.configs)

    group_name = args.run_name
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = cli_utils.create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            max_seq_len=args.policy_seq_len,
            traj_save_len=args.policy_seq_len * 6,
            run_name=run_name,
            tstep_encoder_type=amago.nets.tstep_encoders.FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_type,
            padded_sampling="center",
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_episodes_per_actor
            * (args.max_episode_steps * args.k_eval_episodes + 1),
            grad_clip=2.0,
            learning_rate=3e-4,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_test_env, timesteps=10_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
