from argparse import ArgumentParser
import math
import random

import wandb

import amago
from amago.envs import AMAGOEnv
from amago.envs.builtin.half_cheetah_v4_vel import HalfCheetahV4_MetaVelocity
from amago.cli_utils import *


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


class AMAGOEnvWithVelocityName(AMAGOEnv):
    """
    Every eval metric gets logged based on the current
    `env_name`. You could use this to log metrics for
    different tasks separately. They get averaged over
    all the evals with the same name, so you want a discrete
    number of names that will get sample sizes > 1.
    """

    @property
    def env_name(self) -> str:
        current_task_vel = self.env.unwrapped.target_velocity
        # need to discretize this somehow; just one example
        low, high = math.floor(current_task_vel), math.ceil(current_task_vel)
        return f"HalfCheetahVelocity-Vel-[{low}, {high}]"


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # setup environment
    make_train_env = lambda: AMAGOEnvWithVelocityName(
        MyCustomHalfCheetahTrain(
            task_min_velocity=args.task_min_velocity,
            task_max_velocity=args.task_max_velocity,
        ),
        # the env_name is totally arbitrary and only impacts logging / data filenames
        env_name=f"HalfCheetahV4Velocity",
    )

    make_val_env = lambda: AMAGOEnvWithVelocityName(
        MyCustomHalfCheetahEval(
            task_min_velocity=args.task_min_velocity,
            task_max_velocity=args.task_max_velocity,
        ),
        # this would get replaced by the env_name property
        # defined above.
        env_name=f"HalfCheetahV4VelocityEval",
    )

    config = {}
    # switch sequence model
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    # switch agent
    agent_type = switch_agent(
        config,
        args.agent_type,
        reward_multiplier=1.0,  # gym locomotion returns are already large
        gamma=0.99,  # locomotion policies don't need long horizons - fall back to the default
        tau=0.005,
    )
    # "egreedy" exploration in continuous control is just the epsilon-scheduled random (normal)
    # noise from most TD3/DPPG implementations.
    exploration_type = switch_exploration(config, "egreedy", steps_anneal=500_000)
    use_config(config, args.configs)

    group_name = args.run_name
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
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
            val_timesteps_per_epoch=args.eval_episodes_per_actor * 1001,
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
