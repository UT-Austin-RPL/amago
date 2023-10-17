import random
import math
import warnings
import copy
import pickle
from dataclasses import dataclass

import torch
import numpy as np
import gin


@gin.configurable
@dataclass
class GoalSeq:
    seq: list[np.ndarray]
    active_idx: int
    hide_full_plan: bool = False

    @property
    def current_goal(self) -> np.ndarray:
        if self.active_idx < len(self.seq):
            return self.seq[self.active_idx]
        return None

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return self.seq[i]

    def __setitem__(self, i, item):
        assert isinstance(item, np.ndarray)
        self.seq[i] = item

    @property
    def on_last_goal(self) -> bool:
        return self.active_idx >= len(self.seq) - 1

    def make_array(
        self, pad_to_k_goals=None, pad_val=-1.0, completed_val=0.0
    ) -> np.ndarray:
        goal_array = []
        for i, subgoal in enumerate(self.seq):
            if i < self.active_idx:
                goal_i = (
                    subgoal * 0.0 + completed_val
                )  # = np.full_like(subgoal, completed_val)
                goal_array.append(goal_i)
            elif i == self.active_idx:
                goal_array.append(subgoal)
            else:
                if self.hide_full_plan:
                    continue
                else:
                    goal_array.append(subgoal)

        if pad_to_k_goals is not None:
            pad = pad_to_k_goals - len(goal_array)
            pad_subgoal = (
                self.seq[0] * 0.0 + pad_val
            )  # = np.full_like(self.seq[0], pad_val)
            goal_array = [pad_subgoal] * pad + goal_array

        goal_array = np.array(goal_array).astype(np.float32)
        return goal_array

    def __eq__(self, other):
        if len(other) != len(self):
            return False
        for g_self, g_other in zip(self.seq, other.seq):
            if (g_self != g_other).any():
                return False
        return other.active_idx == self.active_idx

    def __repr__(self):
        if self.active_idx + 1 < len(self.seq):
            next_goal = self.seq[self.active_idx + 1]
        else:
            next_goal = "Completed"
        return f"Current Goal {self.current_goal}, Next Goal: {next_goal}"


class RelabelWarning(Warning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class Timestep:
    obs: np.ndarray
    prev_action: np.ndarray
    achieved_goal: list[np.ndarray]
    goal_seq: GoalSeq
    time: float
    reset: bool
    real_reward: float
    terminal: bool = False

    @property
    def reward(self):
        if self.real_reward is not None:
            return self.real_reward
        elif self.goal_seq.current_goal is None:
            return 0.0
        for achieved in self.achieved_goal:
            rew = float(all(abs(achieved - self.goal_seq.current_goal) < 1e-3))
            if rew > 0:
                return rew
        return 0.0

    @property
    def goal_completed(self):
        if self.real_reward is not None:
            return False

        return self.reward > 0

    @property
    def all_goals_completed(self):
        if self.real_reward is not None:
            return False

        return self.goal_seq.on_last_goal and self.reward > 0

    def __eq__(self, other):
        if len(self.achieved_goal) != len(other.achieved_goal):
            return False
        if self.real_reward != other.real_reward:
            return False
        if self.time != other.time:
            return False
        if self.reset != other.reset:
            return False
        if self.terminal != other.terminal:
            return False
        for goal, other_goal in zip(self.achieved_goal, other.achieved_goal):
            if (goal != other_goal).any():
                return False
        if (self.prev_action != other.prev_action).any():
            return False
        if (self.obs != other.obs).any():
            return False
        return self.goal_seq == other.goal_seq

    def __deepcopy__(self, memo):
        warnings.warn(
            "python shenanigans. `Timestep` deepcopies return *shallow* copies of raw data but *deep* copies of goal sequences (for relabeling).",
            category=RelabelWarning,
        )
        new = self.__class__(
            obs=self.obs,
            prev_action=self.prev_action,
            achieved_goal=self.achieved_goal,
            time=self.time,
            reset=self.reset,
            real_reward=self.real_reward,
            terminal=self.terminal,
            goal_seq=GoalSeq(
                seq=[g.copy() for g in self.goal_seq.seq],
                active_idx=self.goal_seq.active_idx,
            ),
        )
        memo[id(self)] = new
        return new


@gin.configurable(denylist=["max_goals", "timesteps"])
class Trajectory:
    def __init__(
        self,
        max_goals: int,
        timesteps=None,
        goal_pad_val: float = -1.0,
        goal_completed_val: float = -3.0,
    ):
        self.max_goals = max_goals
        self.goal_pad_val = goal_pad_val
        self.goal_completed_val = goal_completed_val
        self.timesteps = timesteps or []

    def add_timestep(self, timestep: Timestep):
        assert isinstance(timestep, Timestep)
        self.timesteps.append(timestep)

    @property
    def total_return(self):
        rews = [t.reward for t in self.timesteps]
        return sum(rews)

    @property
    def is_success(self):
        for t in reversed(self.timesteps):
            if t.all_goals_completed:
                return True
        return False

    def __getitem__(self, i):
        return self.timesteps[i]

    def _make_sequence(self, timesteps) -> np.ndarray:
        make_array = lambda t: t.goal_seq.make_array(
            pad_to_k_goals=self.max_goals,
            pad_val=self.goal_pad_val,
            completed_val=self.goal_completed_val,
        )
        goals = map(make_array, timesteps)
        goals = np.stack(list(goals), axis=0)
        obs = np.stack([t.obs for t in timesteps], axis=0)
        actions = np.stack([t.prev_action for t in timesteps], axis=0)
        resets = np.array([t.reset for t in timesteps], dtype=np.float32)[:, np.newaxis]
        time = np.array([t.time for t in timesteps], dtype=np.float32)[:, np.newaxis]
        rews = np.stack([t.reward for t in timesteps], axis=0)[:, np.newaxis]
        rl2 = np.concatenate((resets, rews, time, actions), axis=-1).astype(np.float32)
        return obs, goals, rl2

    def make_sequence(self, last_only: bool = False):
        if last_only:
            return self._make_sequence([self.timesteps[-1]])
        else:
            return self._make_sequence(self.timesteps)

    def __len__(self):
        return len(self.timesteps)

    def save_to_disk(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(path):
        with open(path, "rb") as f:
            disk = pickle.load(f)
        traj = Trajectory(max_goals=disk.max_goals, timesteps=disk.timesteps)
        return traj

    def __eq__(self, other):
        if len(other) != len(self):
            return False

        for t_self, t_other in zip(self.timesteps, other.timesteps):
            if t_self != t_other:
                return False

        return True

    def __repr__(self):
        str = ""
        for i, t in enumerate(self.timesteps):
            str += f"Achieved: {t.achieved_goal}, GoalSeq: {t.goal_seq}, Reward: {t.reward}, t={i}\n"
        return str


class GoalCounter:
    """
    How likely is this goal to occur on any given timestep?
    """

    def __init__(self, weight_exponent: int = 4):
        self.counts = {}
        self.i = 0
        self.weight_exponent = weight_exponent

    def __call__(self, traj: Trajectory):
        achieved = []
        for t in traj:
            for g in t.achieved_goal:
                key = str(g)
                if key in self.counts:
                    self.counts[key] += 1e-3
                else:
                    self.counts[key] = 1e-7
        self.i += 1
        if self.i > 1e3:
            self.resync()
        self.total_count = sum(self.counts.values())

    def resync(self):
        max_ = max(self.counts.values())
        min_ = min(self.counts.values())
        if max_ > 100.0 or min_ > 1.0:
            self.counts = {k: v / 10.0 for k, v in self.counts.items()}
        self.i = 0

    @property
    def frequencies(self):
        total = self.total_count
        return {k: v / total for k, v in self.counts.items()}

    @property
    def priorities(self):
        freqs = self.frequencies
        ps = {k: (-math.log(v, 10)) ** self.weight_exponent for k, v in freqs.items()}
        return ps

    def weight(self, goals: list[np.ndarray]):
        weights = []
        for goal in goals:
            key = str(goal)
            if key in self.counts:
                weights.append(self.counts[key])
            else:
                weights.append(1e-3)

        weights = (
            -np.log10((np.array(weights) / self.total_count))
        ) ** self.weight_exponent
        weights = np.clip(weights, 1e-3, 1e5)
        return weights.tolist()


class EpisodeCounter(GoalCounter):
    """
    How likely is this goal to occur in any given trajectory?
    """

    def __call__(self, traj: Trajectory):
        # binary tracker of a goal occuring in a given traj
        this_traj = []
        for t in traj:
            for g in t.achieved_goal:
                this_traj.append(str(g))
        this_traj = list(set(this_traj))

        for k in this_traj:
            if k in self.counts:
                self.counts[k] += 1e-3
            else:
                self.counts[k] = 1e-7

        self.i += 1
        if self.i > 1e3:
            self.resync()
        self.total_count = sum(self.counts.values())


class Relabeler:
    def __init__(self, relabel: str = "none", goal_importance_sampling: bool = False):
        if relabel not in ["none", "some", "all", "all_or_nothing"]:
            raise ValueError(
                f"Invalid `Relabeler.relabel` scheme `{relabel}`. Options are: 'none', 'some', `all_or_nothing`, 'all'"
            )
        self.relabel = relabel
        self.goal_statistics = GoalCounter()
        self.episode_statistics = EpisodeCounter()
        self.goal_importance_sampling = goal_importance_sampling

    def _norm_min(self, weights: list[float]):
        min_ = min(weights)
        weights = [w - min_ + 1e-5 for w in weights]
        total = sum(weights)
        return [w / total for w in weights]

    def _norm_median(self, weights: list[float]):
        median_ = np.median(weights)
        weights = [max(w - median_, 1e-5) for w in weights]
        total = sum(weights)
        return [w / total for w in weights]

    def _norm_top_k(self, weights: list[float], top_k: int = 5):
        init_weights = copy.deepcopy(weights)
        sorted_weights = sorted(weights)
        try:
            kth_weight = sorted_weights[-top_k]
        except IndexError:
            kth_weight = sorted_weights[-1]
        weights = [w if w >= kth_weight else 1e-5 for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]
        return weights

    def _norm_uniform(self, weights: list[float]):
        weights = [1.0 / len(weights) for w in weights]
        return weights

    def _select_norm(self):
        # TODO:the specifics of how this is implemented are not important, but
        # should probably be gin-config options.
        p = random.random()
        if p < 0.2:
            norm_method = self._norm_uniform
        elif p < 0.3:
            norm_method = self._norm_min
        elif p < 0.4:
            norm_method = self._norm_median
        else:
            norm_method = self._norm_top_k
        return norm_method

    def __call__(self, traj: Trajectory) -> Trajectory:
        """
        Hindsight Experience Replay for binary rewards and
        multi-goal sequences / "instructions".

        Follows the logic in Figure 2. We can pick a number of achieved
        goals for relabeling, add them in the logical order to the real instruction,
        and then replay the trajectory from the start.

        In some environments there are so many possibilities that we need to start
        sampling based on goal rarity. We do not think the specifics of how this is done
        are important, as long as we are still relabeling with uniform selection
        at some frequency to make sure whatever method we come up with isn't forgetting about
        basic goals.
        """
        ############
        ## Step 0 ##
        ############
        if self.relabel == "none" or traj.is_success:
            # can we just skip this?
            return traj
        if traj[-1].real_reward is not None:
            # `real_rewards` are for the natural env reward, not AMAGO's goal-conditioned system.
            raise RuntimeError(
                "Do not try to relabel trajs that use natural environment rewards"
            )
        og_traj = traj  # save original traj for testing
        traj = copy.deepcopy(traj)  # only deepcopies what we will actually edit (goals)

        #########################################
        ## Step 1: Update Goal Frequency Stats ##
        #########################################
        update_stats = random.random() < 0.1 or (
            self.goal_statistics.i == 0 or self.episode_statistics.i == 0
        )
        if self.goal_importance_sampling and update_stats:
            self.goal_statistics(traj)
            self.episode_statistics(traj)

        ####################################################
        ## Step 2: Find all the goals that were completed ##
        ####################################################
        t_success = []
        for t, timestep in enumerate(traj.timesteps[1:]):
            if timestep.goal_completed:
                t_success.append((t + 1, timestep.goal_seq.current_goal))
        successes = len(t_success)

        ####################################################################
        ## Step 3: Pick a number of new / synthetic goals to relabel with ##
        ####################################################################
        num_goals = len(traj[0].goal_seq)
        if self.relabel == "some":
            # relabel a random number of goals
            assert num_goals >= successes
            num_syn_goals = random.randint(0, num_goals - successes)
        elif self.relabel == "all":
            # relabel the trajectory to be a complete success
            num_syn_goals = num_goals - successes
        elif self.relabel == "all_or_nothing":
            # either make the trajectory a complete success or leave it untouched
            num_syn_goals = num_goals - successes if random.random() < 0.8 else 0
        t_options = [
            t + 1
            for t, traj_i in enumerate(traj.timesteps[1:])
            if len(traj_i.achieved_goal) > 0
        ]
        # (that are not already a success!)
        t_options = list(set(t_options) - set([t_[0] for t_ in t_success]))
        num_syn_goals = min(num_syn_goals, len(t_options))

        ###########################################################
        ## Step 4: Rank timesteps by an optional priority metric ##
        ###########################################################
        if self.goal_importance_sampling and num_syn_goals > 0:
            statistic = (
                self.goal_statistics
                if random.random() < 0.5
                else self.episode_statistics
            )
            total_count = statistic.total_count
            # inverse frequency weighting, summed across multiple goals per timstep if applicable
            t_option_weights = [
                sum(statistic.weight(traj[t].achieved_goal)) for t in t_options
            ]
        else:
            t_option_weights = [1.0 for _ in range(len(t_options))]

        ##############################################
        ## Step 5: Sample timesteps to relabel with ##
        ##############################################
        t_syn_goals = []
        for _ in range(num_syn_goals):
            global_norm_method = self._select_norm()
            t_option_weights_norm = global_norm_method(t_option_weights)
            choice_idx = random.choices(
                range(len(t_options)), k=1, weights=t_option_weights_norm
            )[0]
            t_syn_goals.append(t_options[choice_idx])
            del t_options[choice_idx]
            del t_option_weights[choice_idx]

        ###################################################################################
        ## Step 6: Pick goals from the timesteps we chose (if there are several options) ##
        ###################################################################################
        for t_syn in t_syn_goals:
            local_norm_method = self._select_norm()
            alternate_goal_options = traj[t_syn].achieved_goal
            if self.goal_importance_sampling:
                local_weights = statistic.weight(alternate_goal_options)
                alternate_goal_weights = local_norm_method(local_weights)
            else:
                alternate_goal_weights = [
                    1.0 for _ in range(len(alternate_goal_options))
                ]
            alternate_goal = random.choices(
                alternate_goal_options, k=1, weights=alternate_goal_weights
            )[0]
            t_success.append((t_syn, alternate_goal))

        ###############################################################
        ## Step 7: Replay the trajectory with the newly chosen goals ##
        ###############################################################
        t_success = sorted(
            t_success, key=lambda x: x[0]
        )  # put goals in chronological order
        syn_goal_seq = traj[0].goal_seq.seq
        for i, (_, goal) in enumerate(t_success):
            syn_goal_seq[i] = goal
        active_idx = 0
        end = len(traj)
        for i, timestep in enumerate(traj.timesteps):
            timestep.goal_seq = GoalSeq(seq=syn_goal_seq, active_idx=active_idx)
            timestep.terminal = False
            if timestep.reward > 0:
                active_idx += 1
            if active_idx >= len(syn_goal_seq):
                end = i
                if i + 1 < len(traj):
                    # fill last timestep used in agent update
                    traj.timesteps[i + 1].goal_seq = GoalSeq(
                        seq=syn_goal_seq, active_idx=len(syn_goal_seq)
                    )
                break
        traj.timesteps = traj[: end + 2]
        traj.timesteps[-1].terminal = True

        if end < len(traj):
            assert traj.is_success
        if num_syn_goals == 0:
            # check to be sure the hindsight reward logic matches the real environment logic
            assert traj == og_traj
        return traj
