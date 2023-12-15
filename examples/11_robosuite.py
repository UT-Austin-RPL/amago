from argparse import ArgumentParser
import random
import time
from functools import partial

import wandb
import robosuite
from robosuite.wrappers import DomainRandomizationWrapper, GymWrapper
import numpy as np
import gym
import gymnasium as new_gym
from einops import rearrange

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.nets.tstep_encoders import TstepEncoder
from example_utils import *


def add_cli(parser):
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--from_pixels", action="store_true")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--generalization", action="store_true")
    parser.add_argument("--bc", action="store_true")
    return parser


class _ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, from_pixels : bool):
        super().__init__(env)
        self.from_pixels = from_pixels
        example = self.observation(env.reset())
        obs_dict = {} if not from_pixels else {
            "agentview_image": gym.spaces.Box(
                low=0, high=255, shape=example["agentview_image"].shape
            ),
            "robot0_eye_in_hand_image": gym.spaces.Box(
                low=0, high=255, shape=example["robot0_eye_in_hand_image"].shape
            ),
        }
        for k in example:
            if k not in obs_dict:
                obs_dict[k] = gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=example[k].shape
                )
        self.observation_space = gym.spaces.Dict(obs_dict)
        action_spec = env.action_spec
        self.action_space = gym.spaces.Box(low=action_spec[0], high=action_spec[1])

    def observation(self, raw_obs_dict: dict):
        obs = {
            "robot0_gripper_qpos": raw_obs_dict["robot0_gripper_qpos"],
            "robot0_joint_pos_cos": raw_obs_dict["robot0_joint_pos_cos"],
            "robot0_joint_pos_sin": raw_obs_dict["robot0_joint_pos_sin"],
        }
        if self.from_pixels:
            c_first = lambda img: rearrange(img, "h w c -> c h w")
            obs.update({"agentview_image": c_first(raw_obs_dict["agentview_image"]),
                   "robot0_eye_in_hand_image": c_first(raw_obs_dict["robot0_eye_in_hand_image"])})
        other = []
        for k in sorted(raw_obs_dict.keys()):
            if k not in obs:
                a = raw_obs_dict[k]
                if a.ndim == 0:
                    a = a[np.newaxis]
                other.append(a)
        obs["state_info"] = np.concatenate(other).astype(np.float32)
        return obs


class _SuccessWrapper(gym.Wrapper):
    def step(self, action):
        *other, info = super().step(action)
        info["success"] = self.env._check_success()
        return *other, info


def make_robosuite(
    env_name: str,
    robot: str = "Panda",
    generalization: bool = True,
    from_pixels: bool = False,
):
    controller_config = robosuite.controllers.load_controller_config(
        default_controller="OSC_POSE"
    )
    env = robosuite.make(
        env_name,
        robots=[robot],
        gripper_types="default",
        controller_configs=controller_config,
        env_configuration="single-arm-opposed",
        has_renderer=False,
        render_camera="frontview",
        has_offscreen_renderer=from_pixels,
        control_freq=20,
        use_object_obs=True,
        horizon=500,
        use_camera_obs=from_pixels,
        camera_heights=100,
        camera_widths=100,
        reward_shaping=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        hard_reset=False,
    )
    if generalization:
        env = DomainRandomizationWrapper(
            env,
            randomize_color=True,
            randomize_camera=True,
            randomize_lighting=True,
            randomize_dynamics=True,
            randomize_on_reset=True,
            randomize_every_n_steps=0,
        )

    env = _SuccessWrapper(env)
    env = _ObsWrapper(env, from_pixels=from_pixels)
    return env


class DelayedStart(new_gym.Env):
    def __init__(self, make_env, obs_space, action_space):
        self.make_env = make_env
        self.observation_space = obs_space
        self.action_space = action_space
        self.env = None

    def reset(self, *args, **kwargs):
        if self.env is not None:
            self.env.close()
        self.env = self.make_env()
        obs = self.env.reset()
        for _ in range(5):
            obs, *_ = self.env.step(np.zeros_like(self.action_space.low))
        return obs, {"success": False}

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, done, info


class RobosuiteTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        goal_space,
        rl2_space,
        from_states: bool = False,
        img_features: int = 128,
        out_features: int = 256,
    ):
        super().__init__(obs_space, goal_space, rl2_space)

        obs_keys = list(obs_space.keys())
        self.from_states = from_states
        if from_states:
            # remove images
            obs_keys = [key for key in obs_keys if "image" not in key]
        else:
            # remove priveledged info
            obs_keys.remove("state_info")
        self.obs_keys = sorted(obs_keys)

        if not from_states:
            self.img_norm = amago.nets.ff.Normalization("layer", img_features * 2)
            self.resize = torchvision.transforms.Resize(80)
            self.double_img_features = amago.nets.cnn.NatureishCNN(
                (6, 80, 80), channels_first=True, activation="leaky_relu"
            )
            self.img_mlp = nn.Linear(2304, img_features * 2)

        array_inp_size = (
            sum([obs_space[k].shape[-1] for k in obs_keys if "image" not in k])
            + rl2_space.shape[-1]
        )
        self.array_inp_norm = amago.nets.utils.InputNorm(array_inp_size)
        array_out_size = out_features if from_states else out_features // 2
        self.array_mlp = amago.nets.ff.MLP(
            d_inp=array_inp_size,
            d_hidden=256,
            d_output=array_out_size,
            n_layers=2,
            activation="leaky_relu",
        )

        if not from_states:
            merge_inp_dim = array_out_size + 2 * img_features
            self.merge = amago.nets.ff.MLP(
                d_inp=merge_inp_dim,
                d_hidden=512,
                d_output=out_features,
                n_layers=2,
                activation="leaky_relu",
            )

        self.out_features = out_features
        self.norm = amago.nets.ff.Normalization("layer", out_features)
        self.encoder = encoder

    @property
    def emb_dim(self):
        return self.out_features

    def cast_img(self, img):
        # force consistency between demos and eval
        assert img.dtype == torch.uint8
        img = img.float() / 255.0
        return img

    def inner_forward(self, obs, goal_rep, rl2s):
        if not self.from_states:
            # generate image features
            B, L, *_ = obs["agentview_image"].shape
            before = lambda img: rearrange(img, "b l c h w -> (b l) c h w")
            after = lambda img: rearrange(img, "(b l) c h w -> b l c h w", b=B, l=L)
            agent_img = obs["agentview_image"]
            eye_img = obs["robot0_eye_in_hand_image"]
            double_img = torch.cat((agent_img, eye_img), dim=-3)
            double_img = after(self.resize(before(double_img)))
            img_features = self.double_img_features(double_img)
            img_features = self.img_norm(self.img_mlp(img_features))

        # process everything (we're allowed to use) that isn't an image
        arrays = [rl2s]
        for key in self.obs_keys:
            if "image" not in key:
                arrays.append(obs[key])
        arrays = torch.cat(arrays, dim=-1).float()
        arrays = self.array_inp_norm(arrays)
        if self.training:
            self.array_inp_norm.update_stats(arrays)
        array_features = self.array_mlp(arrays)

        if not self.from_states:
            # add image features
            merge_inp = torch.cat((array_features, img_features), dim=-1)
            out_features = self.merge(merge_inp)
        else:
            out_features = array_features

        return self.norm(out_features)


if __name__ == "__main__":
    # configs
    parser = ArgumentParser()
    add_cli(parser)
    add_common_cli(parser)
    args = parser.parse_args()
    config = {
        "amago.agent.Agent.tstep_encoder_Cls": partial(
            RobosuiteTstepEncoder,
            from_states=not args.from_pixels,
        )
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    example_env = make_robosuite(
        args.env_name, args.robot, args.generalization, from_pixels=args.from_pixels
    )
    obs_space = example_env.observation_space
    action_space = example_env.action_space
    example_env.close()

    make_env = lambda: GymEnv(
        DelayedStart(
            partial(
                make_robosuite,
                env_name=args.env_name,
                robot=args.robot,
                generalization=args.generalization,
            ),
            obs_space=obs_space,
            action_space=action_space,
        ),
        env_name=f"{args.robot}_{args.env_name}_generalization_{args.generalization}_from_pixels_{args.from_pixels}",
        horizon=500,
        zero_shot=True,
    )

    group_name = f"{args.run_name}_{args.robot}_{args.env_name}_generalization_{args.generalization}_l_{args.max_seq_len}_from_pixels_{args.from_pixels}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=1002,
            batch_size=16,
            val_checks_per_epoch=0,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_env, timesteps=4_000, render=False)
        wandb.finish()
