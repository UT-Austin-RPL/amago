from argparse import ArgumentParser
import copy
import random
from functools import partial

import gymnasium as gym
import wandb
import numpy as np

import amago
from amago.envs.builtin.mazerunner import MazeRunnerAMAGOEnv
from amago.hindsight import Relabeler
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--maze_size", type=int, default=15)
    parser.add_argument("--goals", type=int, default=3)
    parser.add_argument("--time_limit", type=int, default=400)
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument(
        "--relabel_strategy", choices=["some", "all", "none"], default="some"
    )
    return parser


class HindsightInstructionReplay(Relabeler):
    def __init__(self, num_goals: int, strategy: str = "all"):
        assert strategy in ["some", "all", "none"]
        self.strategy = strategy
        self.k = num_goals

    def __call__(self, traj):
        # comments align to pseudocode in AMAGO Appendix B Alg 1 (arxiv Page 20)

        if self.strategy == "none":
            del traj.obs["achieved"]
            return traj

        if traj.rews.sum() == self.k:
            og_traj = copy.deepcopy(traj)

        k = self.k
        length = len(traj.rews)

        # line 1-2
        t_g = np.nonzero(traj.rews[:, 0])[0].tolist()
        n = len(t_g)
        goals_completed = [traj.obs["goals"][t][i] for i, t in enumerate(t_g)]

        # line 3
        if self.strategy == "some":
            h = random.randint(0, k - n)
        elif self.strategy == "all":
            h = k - n

        # line 4
        alternative_tsteps = set(
            range(1, length)
        )  # in this env, every timestep has an alternative goal
        candidate_tsteps = list(
            alternative_tsteps - set(t_g)
        )  # but don't sample a timestep that achieved a real goal already
        t_a = random.sample(candidate_tsteps, k=h)
        alternatives = [g[0] for g in traj.obs["achieved"][t_a]]

        # line 5
        a_g = alternatives + goals_completed
        t_ag = t_a + t_g
        r = [g for _, g in sorted(zip(t_ag, a_g))]  # sort a_g by t_ag
        new_goals = np.array(r)[np.newaxis, ...]

        # line 6 "replay the trajectory as if rewards were computed with new_goals"
        traj.obs["goals"] = np.repeat(new_goals, length, axis=0)
        # reset the rewards/dones
        traj.rews[:] = 0.0
        traj.dones[:-1] = False
        traj.dones[-1] = True

        active_goal_idx = 0
        for t in range(length):
            traj.obs["goals"][t][
                :active_goal_idx, :
            ] = -1  # set to "already accomplished"

            if (traj.obs["achieved"][t] == new_goals[0, active_goal_idx]).all():
                # next goal achieved
                traj.rews[t] = 1.0  # give reward
                traj.rl2s[t - 1][
                    0
                ] = 1.0  # a little hacky... prev rewards in are 0 idx of rl2s
                active_goal_idx += 1  # advance active goal

            if active_goal_idx >= k:
                traj.dones[t] = True
                break

        # enforce early termination
        # rule of thumb in amago: inputs to seq models are a timestep
        # longer than RL data used for TD updates
        del traj.obs["achieved"]
        traj.obs = {k: v[: t + 1] for k, v in traj.obs.items()}
        traj.rl2s = traj.rl2s[: t + 1]
        traj.time_idxs = traj.time_idxs[: t + 1]
        traj.rews = traj.rews[:t]
        traj.dones = traj.dones[:t]
        traj.actions = traj.actions[:t]
        return traj


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # setup environment
    make_train_env = lambda: MazeRunnerAMAGOEnv(
        maze_dim=args.maze_size,
        num_goals=args.goals,
        time_limit=args.time_limit,
    )
    config = {
        "amago.nets.tstep_encoders.FFTstepEncoder.d_hidden": 128,
        "amago.nets.tstep_encoders.FFTstepEncoder.specify_obs_keys": ["obs", "goals"],
    }
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
    )

    exploration_type = switch_exploration(
        config, strategy="egreedy", steps_anneal=500_000
    )
    use_config(config, args.configs)

    group_name = args.run_name
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.time_limit,
            traj_save_len=args.time_limit + 1,
            stagger_traj_file_lengths=False,
            relabel_type=partial(
                HindsightInstructionReplay,
                num_goals=args.goals,
                strategy=args.relabel_strategy,
            ),
            run_name=run_name,
            tstep_encoder_type=amago.nets.tstep_encoders.FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.time_limit * 5,
        )
        experiment = switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(
            make_train_env, timesteps=args.time_limit * 20, render=False
        )
        experiment.delete_buffer_from_disk()
        wandb.finish()
