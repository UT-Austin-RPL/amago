from argparse import ArgumentParser

import wandb

import amago
from amago.envs import AMAGOEnv
from amago.envs.builtin.half_cheetah_v4_vel import HalfCheetahV4_MetaVelocity
from amago import cli_utils


def add_cli(parser):
    parser.add_argument(
        "--policy_seq_len", type=int, default=32, help="Policy sequence length."
    )
    parser.add_argument(
        "--eval_episodes_per_actor",
        type=int,
        default=1,
        help="Validation episodes per parallel actor.",
    )
    parser.add_argument(
        "--task_min_velocity",
        type=float,
        default=0.0,
        help="Min running velocity the cheetah needs to be capable of to solve the meta-learning problem. Original benchmark used 0.",
    )
    parser.add_argument(
        "--task_max_velocity",
        type=float,
        default=3.0,
        help="Max running velocity the cheetah needs to be capable of to solve the meta-learning problem. Original benchmark used 3. Agents in the default locomotion env (no reward randomization) reach > 10.",
    )
    parser.add_argument(
        "--inner_episode_steps",
        type=int,
        default=200,
        help="Step horizon of each inner episode. Default 200 (combined with the default --k_train_episodes=3) keeps total trial length at 600 steps. Set 1000 with --k_train_episodes=1 to recover the unwrapped task.",
    )
    parser.add_argument(
        "--k_train_episodes",
        type=int,
        default=3,
        help="Inner episodes per meta-trial during training. Default 3 makes the env a true meta-RL trial: a single hidden target velocity persists across 3 inner episodes (soft resets between). Set 1 to recover the unwrapped task.",
    )
    parser.add_argument(
        "--k_eval_episodes",
        type=int,
        default=None,
        help="Inner episodes per meta-trial at eval time. Defaults to --k_train_episodes. Larger values (e.g. 10) probe how well the agent keeps adapting beyond its training horizon.",
    )
    return parser


"""
Because this task is so similar to the other gymnasium examples, this example script is overly
verbose about showing how you could customize the environment and create a train/test split.

If you don't edit anything, this only becomes a longer way to train/test on the default task
distribution (which is to sample a velocity uniformly between: [args.task_min_velocity, args.task_max_velocity])
"""


class MyCustomHalfCheetahTrain(HalfCheetahV4_MetaVelocity):
    def sample_target_velocity(self) -> float:
        # be sure to use `random` or be careful about np default_rng to ensure
        # tasks are different across async parallel actors!
        vel = super().sample_target_velocity()  # random.uniform(min_vel, max_vel)
        return vel


class MyCustomHalfCheetahEval(HalfCheetahV4_MetaVelocity):
    def sample_target_velocity(self) -> float:
        vel = super().sample_target_velocity()
        # or, to create OOD eval tasks:
        # vel = random.uniform(self.task_min_velocity, self.task_max_velocity * 10.0)
        # or random.choice([0., 1., self.task_max_velocity * 1.2]), etc.
        return vel


if __name__ == "__main__":
    parser = ArgumentParser()
    cli_utils.add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    k_eval = (
        args.k_eval_episodes
        if args.k_eval_episodes is not None
        else args.k_train_episodes
    )

    def make_train_env():
        return AMAGOEnv(
            MyCustomHalfCheetahTrain(
                task_min_velocity=args.task_min_velocity,
                task_max_velocity=args.task_max_velocity,
                max_episode_steps=args.inner_episode_steps,
                k_episodes=args.k_train_episodes,
            ),
            env_name="HalfCheetahV4Velocity",
        )

    def make_val_env():
        return AMAGOEnv(
            MyCustomHalfCheetahEval(
                task_min_velocity=args.task_min_velocity,
                task_max_velocity=args.task_max_velocity,
                max_episode_steps=args.inner_episode_steps,
                k_episodes=k_eval,
            ),
            env_name="HalfCheetahV4Velocity",
        )

    config = {
        "amago.nets.traj_encoders.TformerTrajEncoder.pos_emb": "rope",
    }
    # switch sequence model
    traj_encoder_type = cli_utils.switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    # switch agent
    agent_type = cli_utils.switch_agent(
        config,
        args.agent_type,
        reward_multiplier=1.0,  # gym locomotion returns are already large
        gamma=0.99,  # locomotion policies don't need long horizons - fall back to the default
        tau=0.005,
    )
    # "egreedy" exploration in continuous control is just the epsilon-scheduled random (normal)
    # noise from most TD3/DPPG implementations.
    exploration_type = cli_utils.switch_exploration(
        config, "egreedy", steps_anneal=500_000
    )
    cli_utils.use_config(config, args.configs)

    group_name = args.run_name
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = cli_utils.create_experiment_from_cli(
            args,
            make_train_env=make_train_env,  # different train/val envs
            make_val_env=make_val_env,
            max_seq_len=args.policy_seq_len,
            traj_save_len=args.policy_seq_len * 6,
            run_name=run_name,
            tstep_encoder_type=amago.nets.tstep_encoders.FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.eval_episodes_per_actor
            * (args.inner_episode_steps * k_eval + 1),
            grad_clip=2.0,
            learning_rate=3e-4,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_val_env, timesteps=10_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
