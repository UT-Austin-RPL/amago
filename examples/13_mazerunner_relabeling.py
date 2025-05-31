"""
A demonstration of the hindsight instruction relabeling techinque discussed in the AMAGO paper -
a generalization of Hindsight Experience Replay (HER) to sequences of multiple goals.
The ability to relabel data is another good reason to prefer off-policy RL^2 to on-policy.

This example uses the "MazeRunner" environment. MazeRunner is an adapted version of Memory Maze
(https://arxiv.org/abs/2210.13383) that does not require the DM Lab simulator or learning from pixels.
For more information please refer to the AMAGO Appendix C.4.

There are three steps to using hindsight relabeling:

1. Make the env's observations a dict with keys for the intended goal and alternative goals achieved
   at that timestep. Goals are often subsets of the state space. In this example, observations consist of:

   `obs` : the regular observation from the maze environment (LIDAR-ish depth sensors to the walls, timer, etc.)
   `goals` : a sequence of k goal positions to navigate to.
   `achieved`: the agent's current position.

2. Let the policy network take the observation and goals as input, but ignore the `achieved` data.

3. Use the `achieved` key to relabel data with alternative goal sequences that would lead to higher returns.
"""

from argparse import ArgumentParser
import random
from functools import partial

import gymnasium as gym
import wandb
import numpy as np

import amago
from amago.envs.builtin.mazerunner import MazeRunnerAMAGOEnv
from amago.hindsight import Relabeler, FrozenTraj
from amago.loading import DiskTrajDataset
from amago import cli_utils


def add_cli(parser):
    parser.add_argument(
        "--maze_size",
        type=int,
        default=11,
        help="Dimension of randomly generated mazes (n x n). n must be odd.",
    )
    parser.add_argument(
        "--goals",
        type=int,
        default=3,
        help="Length of the sequence of goal positions to reach during the episode.",
    )
    parser.add_argument(
        "--time_limit", type=int, default=250, help="Episode time limit."
    )
    parser.add_argument(
        "--relabel",
        choices=["some", "all", "none"],
        default="some",
        help="`none` skips relabeling, `all` relabels every trajectory to a success. `some` creates a mixture of varying returns.",
    )
    parser.add_argument(
        "--randomized_actions",
        action="store_true",
        help="Randomize the directions of movement each episode (requires context-based identitifcation of current controls).",
    )
    return parser


class HindsightInstructionReplay(Relabeler):
    """
    Hindsight Experience Replay extended to "instructions" or sequences of multiple goals.

    Relabelers are passed RL trajectory data before it is padded + batched and sent to the agent
    for training.
    """

    def __init__(self, num_goals: int, strategy: str = "all"):
        assert strategy in ["some", "all", "none"]
        self.strategy = strategy
        self.k = num_goals

    def relabel(self, traj: FrozenTraj) -> FrozenTraj:
        """
        Assume observations are a dict with three keys:
        1. `obs` : the regular observation from the env
        2. `achieved` : candidate goals for relabeling (in this case: current (x, y) position)
        3. `goals` : a (k, n) array of the k goals we want to reach

        Agent receives `obs` and `goals` as input, and we use `achieved` to relabel failed trajectories
        with new goals that lead to more reward signal.

        var names are references to pseudocode in AMAGO Appendix B Alg 1 (arXiv page 20)
        """
        if self.strategy == "none":
            del traj.obs["achieved"]
            return traj

        # 1. Find timesteps where original goals were completed
        k = self.k
        length = len(traj.rews)
        tsteps_with_goals = np.nonzero(traj.rews[:, 0])[0].tolist()
        n = len(tsteps_with_goals)
        goals_completed = [
            traj.obs["goals"][t][i] for i, t in enumerate(tsteps_with_goals)
        ]

        # 2. Determine how many relabled goals we'll add
        h = k - n if self.strategy == "all" else random.randint(0, k - n)

        # it's important that this relabeler can recreate the exact reward func
        # and terminal signals of the original env. The best way to check that is to
        # let successful trajs (n == self.k, traj.rews.sum() == self.k) or those
        # where we're not adding goals (h == 0) carry on through relabeling,
        # then check that the "relabeled" verion is the same as the original.
        # Since we've already done this we'll save the relabel time:
        if h == 0:
            del traj.obs["achieved"]
            return traj

        # 3. Pick h goals that were achieved as replacements for relabeling
        # in this env, every timestep has an alternative goal
        alternative_tsteps = set(range(1, length))
        # but don't sample a timestep that achieved a real goal already
        candidate_tsteps = list(alternative_tsteps - set(tsteps_with_goals))
        h = min(h, len(candidate_tsteps))
        tsteps_with_alt_goals = random.sample(candidate_tsteps, k=h)
        alternatives = [g[0] for g in traj.obs["achieved"][tsteps_with_alt_goals]]

        # 4. Sort the (new) "alternative" goals and completed (real) goals in the order
        # they'd occur in the trajectory. Leave uncompleted real goals at the end
        # (in their original order).
        combined_goals = alternatives + goals_completed
        tsteps_with_combined_goals = tsteps_with_alt_goals + tsteps_with_goals
        # sort combined_goals by tsteps
        r = [g for _, g in sorted(zip(tsteps_with_combined_goals, combined_goals))]
        new_goals = traj.obs["goals"][0].copy()
        for i, new_goal in enumerate(r):
            new_goals[i, :] = new_goal

        # 5. Replay the trajectory as if rewards were computed with new_goals.
        # This step requires knowledge of the reward function (like HER). In this
        # case: binary check that "achieved" == "goals"[current_goal_idx]
        traj.obs["goals"] = np.repeat(new_goals[np.newaxis, ...], length + 1, axis=0)
        traj.rews[:] = 0.0
        # note that `rl2s` array is prev_action + prev_reward. The rew is in the 0 index.
        traj.rl2s[:, 0] = 0.0
        active_goal_idx = 0
        for t in range(length + 1):
            # the format of arrays in the `traj` object is:
            # traj.obs = {o_0, o_1, ..., o_length}
            # traj.rl2s = {rl2_0, rl2_1, ..., rl2_length}
            # traj.rews = {r_1, r_2, ..., missing}
            # traj.dones = {d_1, d_2, ..., missing}
            achieved_this_turn = traj.obs["achieved"][t][0]
            # some envs might have multiple goals per timestep
            if np.array_equal(achieved_this_turn, new_goals[active_goal_idx]):
                traj.rews[t - 1] = 1.0
                traj.rl2s[t][0] = 1.0
                active_goal_idx += 1
            # -1 is "accomplished"... would change by env, but we need some way to keep goals consistent shape
            traj.obs["goals"][t][:active_goal_idx, ...] = -1
            if active_goal_idx >= k:
                traj.dones[t - 1] = True
                break
        # enforce early termination
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
    cli_utils.add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # setup environment
    # the AMAGO wrapper adds the relabeling info to the obs dict
    make_train_env = lambda: MazeRunnerAMAGOEnv(
        maze_dim=args.maze_size,
        num_goals=args.goals,
        time_limit=args.time_limit,
        randomized_action_space=args.randomized_actions,
    )
    config = {}
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
        # reward_multiplier=100.,
    )
    tstep_encoder_type = cli_utils.switch_tstep_encoder(
        config,
        arch="ff",
        n_layers=2,
        d_hidden=128,
        # ignore "achieved" obs dict in the policy net.
        specify_obs_keys=["obs", "goals"],
    )
    exploration_type = cli_utils.switch_exploration(
        config,
        strategy="egreedy",
        steps_anneal=500_000,  # needs re-tuning; paper used an older version
    )
    cli_utils.use_config(config, args.configs)

    group_name = args.run_name
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        dataset = DiskTrajDataset(
            dset_root=args.buffer_dir,
            dset_name=run_name,
            dset_max_size=args.dset_max_size,
            relabeler=HindsightInstructionReplay(
                num_goals=args.goals,
                strategy=args.relabel,
            ),
        )

        experiment = cli_utils.create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            # paper made a point of using maximum context length; in practice this can be shortened with similar results
            max_seq_len=args.time_limit,
            # make sure the entire trajectory is contained in one file that will be sent to the relabeler
            traj_save_len=args.time_limit + 1,
            stagger_traj_file_lengths=False,
            # provide the dataset explicitly to use our relabeler instead of the default.
            # create_experiment_from_cli creates the default dataset otherwise.
            dataset=dataset,
            run_name=run_name,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            exploration_wrapper_type=exploration_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=args.time_limit * 5,
        )
        experiment = cli_utils.switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(
            make_train_env, timesteps=args.time_limit * 20, render=False
        )
        experiment.delete_buffer_from_disk()
        wandb.finish()
