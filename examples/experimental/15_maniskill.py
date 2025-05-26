"""Offline-To-Online FineTuning Example w/ ManiSkill 2-Finger Tabletop Tasks

This example is much more experimental than the others. Setup is more involved:

1. Download the demo data for a task:

python -m mani_skill.utils.download_demo "PushCube-v1"

2. Decide on the demonstrator to imitate (most tasks seem to have `rl` and `motionplanning`). 

3. Decide on the controller. Online PPO baselines for some of these tasks appear to use `pd_joint_delta_pos`, while motionplanning demos list `pd_ee_delta_pos` as the official baseline dataset.

4. Decide on the sim backend ("physx_cpu" or "physx_cuda"). Most of the motionplanning datasets use cpu, while most the rl ones use cuda.
   We'll try to convert the dataset to the desired backend, but the docs make it pretty clear this is a bad idea.

5. Then run this command with `--controller`, `--demonstrator` and `--backend` set accordingly.
"""

import os
import collections
import functools
import random
import warnings
import copy
from typing import Optional

import torch
import wandb
import h5py
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import mani_skill.envs
import mani_skill
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.io_utils import load_json

import amago
from amago.envs import AMAGOEnv, AMAGO_ENV_LOG_PREFIX
from amago.cli_utils import *
from amago.loading import RLData, RLDataset, MixtureOfDatasets, DiskTrajDataset


#######################
## Setup Environment ##
#######################


class VectorizedNumpyWrapper(gym.Env):
    """
    Step the envs in parallel, but we expect
    numpy arrays and unbatched observation/action spaces.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = env.num_envs

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return obs.cpu().numpy(), info

    def step(self, action):
        # fmt: off
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated.cpu().numpy()
        truncated = truncated.cpu().numpy()
        reward = reward.cpu().numpy()
        done = terminated | truncated
        obs = obs.cpu().numpy()

        # build info dict with eval stats
        out_info = collections.defaultdict(list)
        if done.any():
            for i in range(self.num_envs):
                done_i = done[i]
                if not done_i:
                    continue
                # if terminal, add the success metric to the info dict with a special key
                # that tells AMAGO to log the average of that metric over the entire eval
                if "final_info" in info:
                    success_i = info["final_info"]["episode"]["success_once"][i]
                else:
                    success_i = info["episode"]["success_once"][i]
                out_info[f"{AMAGO_ENV_LOG_PREFIX} Success [0, 1]"].append(success_i.cpu().numpy().item())

        # fmt: on
        return obs, reward, terminated, truncated, dict(out_info)


class ManiSkillCPUSuccessLogger(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            success = info["episode"]["success_once"]
            info[f"{AMAGO_ENV_LOG_PREFIX} Success [0, 1]"] = success
        return obs, reward, terminated, truncated, info


def make_env(
    name: str,
    parallel_envs: int,
    env_kwargs: dict,
    override_max_ep_len: Optional[int] = None,
):
    """
    Create an environment according to the dataset's metadata
    and the reccomendations for RL/IL evals from the ManiSkill docs.

    Note `ignore_terminations=True`
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    env_kwargs = env_kwargs.copy()
    backend = env_kwargs["sim_backend"]
    env_kwargs["reconfiguration_freq"] = 1
    if backend == "physx_cpu":
        # cpu mode --> regular async gym
        env_kwargs["num_envs"] = 1
        env = gym.make(name, render_backend="cpu", **env_kwargs)
        if override_max_ep_len is not None:
            env._max_episode_steps = override_max_ep_len
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        # log success rate to amago wandb console
        env = ManiSkillCPUSuccessLogger(env)
        return AMAGOEnv(env, env_name=name, batched_envs=1)
    elif backend == "physx_cuda":
        # gpu mode --> vectorized env with a numpy wrapper to work with AMAGO
        env_kwargs["num_envs"] = parallel_envs
        env = gym.make(name, **env_kwargs)
        if override_max_ep_len is not None:
            env._max_episode_steps = override_max_ep_len
        env = ManiSkillVectorEnv(
            env, auto_reset=True, ignore_terminations=True, record_metrics=True
        )
        # log success rate to amago wandb console, convert to numpy
        env = VectorizedNumpyWrapper(env)
        return AMAGOEnv(env, env_name=name, batched_envs=parallel_envs)
    else:
        raise ValueError(f"Invalid backend: {backend}")


#################################################
## Create Custom RLDataset for ManiSkill Demos ##
#################################################


def replay_maniskill_trajs_for_learning(
    demonstrator: str, controller: str, env_name: str, backend: str
) -> str:
    """
    Attempt to automate the confusing process of converting the trajectory data that is downloaded from ManiSkil's dataset
    into the RL format (with observations + rewards) for the various controllers, backends, and demonstrators.
    """
    # fmt: off
    assert backend in ["physx_cpu", "physx_cuda"]

    if demonstrator == "motionplanning":
        path = os.path.join(mani_skill.DEMO_DIR, env_name, demonstrator, f"trajectory.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trajectory file not found at {path}")
    else:
        cpu_path = os.path.join(mani_skill.DEMO_DIR, env_name, demonstrator, f"trajectory.none.{controller}.physx_cpu.h5")
        cpu_exists = os.path.exists(cpu_path)
        gpu_path = cpu_path.replace(".physx_cpu", ".physx_cuda")
        gpu_exists = os.path.exists(gpu_path)
        if backend == "physx_cpu" and cpu_exists:
            path = cpu_path
        elif backend == "physx_cpu" and gpu_exists:
            warnings.warn(f"Requested CPU backend, but we only found a GPU backend source file. Maniskill will try to convert but docs discourage this due to sim --> sim issues.")
            path = gpu_path
        elif backend == "physx_cuda" and gpu_exists:
            path = gpu_path
        elif backend == "physx_cuda" and cpu_exists:
            warnings.warn(f"Requested GPU backend, but we only found a CPU backend source file. Maniskill will try to convert but docs discourage this due to sim --> sim issues.")
            path = cpu_path
        else:
            raise FileNotFoundError(f"Trajectory file not found at {cpu_path} or {gpu_path}")

    expected_outcome = os.path.join(mani_skill.DEMO_DIR, env_name, demonstrator, f"trajectory.state.{controller}.{backend}.h5")
    if not os.path.exists(expected_outcome):
        command = f"python -m mani_skill.trajectory.replay_trajectory  --traj-path {path} --use-first-env-state -o state -c {controller} --save-traj --num-envs 10 -b {backend} --record-rewards --reward-mode normalized_dense"
        os.system(command)
    else:
        warnings.warn(f"Trajectory file already exists at {expected_outcome}")
    # fmt: on
    return expected_outcome


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillTrajectoryDataset(RLDataset):
    """
    Create an RLDataset from the ManiSkill demonstration h5 files
    """

    def __init__(
        self,
        dataset_file: str,
        success_only: bool = False,
    ) -> None:
        super().__init__()

        # Load the h5 trajectories and metadata based on starter code from the ManiSkill repo.
        self.dataset_file = dataset_file
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        load_count = len(self.episodes)
        to_torch = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        self.max_ep_len = -float("inf")

        self.rl_datas = []
        with h5py.File(dataset_file, "r") as data:
            for eps_id in tqdm(
                range(load_count), desc="Loading ManiSkillTrajectoryDataset"
            ):
                eps = self.episodes[eps_id]
                if success_only:
                    assert (
                        "success" in eps
                    ), "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                    if not eps["success"]:
                        continue
                trajectory = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(trajectory)
                """
                On Truncation and Termination:

                - ManiSkill reccomends treating done = truncated (not terminated) and measuring "success_once"
                    (did we ever succeed?). Resets stay synced across actors at the cost of letting the robot
                    keep going after it's already done.
                - The envs all have established truncated = True time limits (standard TimeLimit wrapper)
                - But the demos go well beyond the time limit. The first truncated seems to match up with 
                    the expected time limit, but that's not the end of the sequence for some reason. 
                    Truncated signals stay true until the end.
                - So I'd think the correct approach is to cut the episode off after the first truncated.
                    But the rewards don't hit their peak value of 1.0 until well after the first truncated.
                - I tried it this way and IL on successes only completely fails, even on tasks so easy you can't blame
                    a lack of diffusion policy or whatever.
                - I tried completely ignoring the dataset's terminated/truncated flags. The policy trains on longer
                    sequences than it will see at test-time, but if we're supposed to be cutting these demo seqs short
                    it will follow the demo policy until eval terminates without issue. This fails too.
                - Overriding the max episode length of the environment to the length of the longest demo trajectory
                    and then imitating the demos without any truncation hits 100% SR.
                - In conclusion, by pure trial and error I think the demos do not match up with the env's time limit
                    and overriding both the demo truncation and the env episode length is the way to make training 1:1 with eval.
                """
                # make obs a dict that matches the AMAGOEnv default
                ep_len = len(trajectory["actions"])
                self.max_ep_len = max(self.max_ep_len, ep_len)
                obs = {"observation": to_torch(trajectory["obs"]).float()}
                actions = to_torch(trajectory["actions"]).float()
                # NOTE: this seems to happen when i go off the beaten path of the official IL reference datasets.
                # Maybe there is a controller conversion / action space normalization issue that the reference dataset
                # skips over (?). The env will always list the action space as [-1, 1], so standard rescaling doesn't
                # seem like the right solution.
                assert (
                    abs(actions).max() <= 1.0
                ), "Trajectory replay has generated actions outside of [-1, 1] range."
                rewards = to_torch(trajectory["rewards"]).unsqueeze(-1).float()
                # completely ignore the dataset's terminated/truncated flags.
                dones = torch.zeros_like(rewards, dtype=torch.bool)
                dones[-1] = True
                time_idxs = torch.arange(ep_len + 1).long().unsqueeze(-1)
                rl_data = RLData(
                    obs=obs,
                    actions=actions,
                    rews=rewards,
                    dones=dones,
                    time_idxs=time_idxs,
                )
                self.rl_datas.append(copy.deepcopy(rl_data))

    def get_description(self):
        # prints some basic info to the console on startup
        return f"ManiSkillTrajectoryDataset \n \t {self.dataset_file} \n \t Trajs: {len(self.rl_datas)}"

    def sample_random_trajectory(self):
        # just pick a random traj from the list.
        return random.choice(self.rl_datas)

    @property
    def save_new_trajs_to(self):
        # tells Experiment that we don't want to save online trajs to disk
        return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="PushCube-v1", help="ManiSkill environment name"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="pd_ee_delta_pos",
        help="controller / action space",
    )
    parser.add_argument(
        "--demonstrator",
        type=str,
        default="motionplanning",
        choices=["rl", "motionplanning"],
        help="ManiSkill demonstration source.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
        help="ManiSkill backend to use for training.",
    )
    parser.add_argument("--il", action="store_true", help="just do imitation learning")
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="only train on demonstrations from successful episodes",
    )
    parser.add_argument(
        "--online_after_epoch",
        type=int,
        default=None,
        help="start online RL collection after this many epochs of training on the demos. Offline RL only if None.",
    )
    add_common_cli(parser)
    args = parser.parse_args()

    # attempt to create the h5 dataset from the downloaded trajectory data
    demo_path = replay_maniskill_trajs_for_learning(
        args.demonstrator, args.controller, args.env, args.backend
    )

    # create the RL dataset from the replayed trajectory data
    maniskill_dset = ManiSkillTrajectoryDataset(
        dataset_file=demo_path,
        success_only=args.success_only,
    )

    # make an environment that is 1:1 with the dataset's metadata and sequence length
    backend = maniskill_dset.env_kwargs["sim_backend"]
    horizon = max(gym.registry[args.env].max_episode_steps, maniskill_dset.max_ep_len)
    args.eval_timesteps = horizon
    args.timesteps_per_epoch = horizon
    make_train_env = functools.partial(
        make_env,
        name=args.env,
        parallel_envs=args.parallel_actors,
        env_kwargs=maniskill_dset.env_kwargs,
        override_max_ep_len=horizon,
    )

    args.env_mode = "already_vectorized" if backend == "physx_cuda" else "async"

    # setup our agent
    from amago.nets import actor_critic, policy_dists, transformer
    from amago import agent

    config = {
        # continuous policy dist choice becomes super important outside gym mujoco / dmc
        "actor_critic.Actor.continuous_dist_type": policy_dists.Beta,
        # robotics IL often use short-context sequence models for demonstrator multi-modality.
        # We could lower the max_seq_len, but this introduces some padding concerns and k/v cache drift.
        # These episodes are so short that we can go with the better approach: operate on entire
        # trajectories, but use sliding window attention.
        "traj_encoders.TformerTrajEncoder.attention_type": transformer.FlashAttention,
        "transformer.FlashAttention.window_size": (12, 0),
        "agent.exp_filter.clip_weights_high": 1000,
    }
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    tstep_encoder_type = switch_tstep_encoder(
        config,
        arch="ff",
        # guess slightly smaller architecture than default
        d_hidden=256,
        n_layers=2,
    )
    agent_type = switch_agent(
        config,
        args.agent_type,
        # usually safer to scale rewards when they're this small
        # (we use PopArt to renormalize values)
        reward_multiplier=10.0,
        # fake_filter = True --> Imitation Learning
        fake_filter=args.il,
        # disable "online" loss that suffers from Q overestimation
        online_coeff=0.0,
        # rely entirely on filtered/weighted BC
        offline_coeff=1.0,
        # exponential filter (exp(Advantage)) is generally safer
        # when demonstrations are known to be high-quality.
        fbc_filter_func=agent.exp_filter,
        # increase action sampling for value estimates using continuous actions
        num_actions_for_value_in_critic_loss=2,
        num_actions_for_value_in_actor_loss=4,
        # slightly faster target updates (default is super conservative)
        tau=0.005,
    )
    exploration_type = switch_exploration(
        config,
        "egreedy",
        # short/low-noise schedule given highly parallel actor setup & pre-training on demos
        eps_start=0.1,
        eps_end=0.05,
        steps_anneal=5_000,
    )
    use_config(config, args.configs)

    # run training
    group_name = f"{args.run_name}_{args.env}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        online_dset = DiskTrajDataset(
            dset_root=args.buffer_dir,
            dset_name=run_name,
            dset_min_size=500,
            dset_max_size=args.dset_max_size,
        )

        combined_dset = MixtureOfDatasets(
            datasets=[maniskill_dset, online_dset],
            # skew sampling towards the demos 60/40
            sampling_weights=[0.6, 0.4],
            # gradually increase the weight of the online dset
            # over the first 100 epochs *after online collection starts*
            smooth_sudden_starts=100,
        )

        experiment = create_experiment_from_cli(
            args,
            dataset=combined_dset,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            # training and inference on entire episodes
            max_seq_len=horizon,
            traj_save_len=horizon + 1,
            run_name=run_name,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            group_name=group_name,
            val_timesteps_per_epoch=horizon + 1,
            start_collecting_at_epoch=(args.online_after_epoch or float("inf")) + 1,
            learning_rate=1.25e-4,
            grad_clip=2.0,
            force_reset_train_envs_every=1,
            async_env_mp_context="forkserver" if backend == "physx_cpu" else None,
        )

        experiment = switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.evaluate_test(make_train_env, timesteps=horizon * 5, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()
