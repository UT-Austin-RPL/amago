from argparse import ArgumentParser
from functools import partial

import wandb
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

import amago
from amago.envs.builtin.crafter_envs import CrafterEnv
from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument(
        "--default_rew",
        action="store_true",
        help="Use Crafter's default reward function ('undirected' multi-task learning)",
    )
    parser.add_argument(
        "--use_tech_tree",
        action="store_true",
        help="Use Crafter's ground-truth tech-tree to generate tasks (see Appendix)",
    )
    parser.add_argument("--relabel", choices=["some", "none", "all"], default="some")
    parser.add_argument("--max_seq_len", type=int, default=2500)
    parser.add_argument(
        "--obs_kind", choices=["render", "crop", "textures"], default="textures"
    )
    parser.add_argument("--eval_mode", action="store_true")
    return parser


class CrafterTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    def __init__(
        self,
        obs_kind: str,
        obs_space,
        goal_space,
        rl2_space,
        img_features: int = 256,
        emb_dim: int = 256,
    ):
        super().__init__(
            obs_space=obs_space, goal_space=goal_space, rl2_space=rl2_space
        )

        self.obs_kind = obs_kind
        self.out_norm = amago.nets.ff.Normalization("layer", emb_dim)
        self.rl2_norm = amago.nets.utils.InputNorm(self.rl2_space.shape[-1])
        if obs_kind in ["crop", "render"]:
            img_shape = obs_space["image"].shape
            self.cnn = amago.nets.cnn.NatureishCNN(
                img_shape=img_shape, channels_first=False, activation="leaky_relu"
            )
            img_feature_dim = self.cnn(
                torch.zeros((1, 1) + img_shape, dtype=torch.uint8)
            ).shape[-1]
            self.img_features = nn.Linear(img_feature_dim, img_features)
            self.img_norm = amago.nets.ff.Normalization("layer", img_features)
            mlp_in_dim = (
                img_features
                + (0 if obs_kind == "render" else self.obs_space["inventory"].shape[-1])
                + self.rl2_space.shape[-1]
                + self.goal_emb_dim
            )
        else:
            self.texture_emb = nn.Embedding(64 + 1, 4)
            self.texture_ff1 = nn.Linear(4 * 9 * 7, 192)
            self.texture_ff2 = nn.Linear(192, 192)
            self.texture_norm = amago.nets.ff.Normalization("layer", 192)
            mlp_in_dim = (
                192
                + self.obs_space["info"].shape[-1]
                + self.rl2_space.shape[-1]
                + self.goal_emb_dim
            )
        self.ff = amago.nets.ff.MLP(
            d_inp=mlp_in_dim, d_hidden=mlp_in_dim * 2, n_layers=2, d_output=emb_dim
        )
        self._emb_dim = emb_dim

    @property
    def emb_dim(self):
        # TstepEncoders need an emb_dim for AMAGO to build the rest of the architecture from here
        return self._emb_dim

    def inner_forward(self, obs, goal_rep, rl2s, log_dict=None):
        rl2s = self.rl2_norm(rl2s)
        if self.training:
            self.rl2_norm.update_stats(rl2s)
        if self.obs_kind in ["crop", "render"]:
            img_features = self.cnn(obs["image"])
            img_features = self.img_norm(self.img_features(img_features))
            extra = (rl2s, goal_rep)
            if self.obs_kind == "crop":
                extra += (obs["inventory"],)
            mlp_input = torch.cat(extra + (img_features,), dim=-1)
        else:
            textures = rearrange(obs["textures"], "b l h w -> b l (h w)").long()
            texture_features = rearrange(
                self.texture_emb(textures), "b l f d -> b l (f d)"
            )
            texture_features = F.leaky_relu(self.texture_ff1(texture_features))
            texture_features = self.texture_norm(self.texture_ff2(texture_features))
            mlp_input = torch.cat(
                (rl2s, goal_rep, obs["info"], texture_features), dim=-1
            )
        out = self.out_norm(self.ff(mlp_input))
        return out


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # no need to risk numerical instability when returns are this bounded
        "amago.agent.Agent.reward_multiplier": 10.0,
        "amago.agent.Agent.tstep_encoder_Cls": partial(
            CrafterTstepEncoder, obs_kind=args.obs_kind
        ),
        # token-based goal embedding
        "amago.nets.tstep_encoders.TstepEncoder.goal_emb_Cls": amago.nets.goal_embedders.TokenGoalEmb,
        "amago.nets.goal_embedders.TokenGoalEmb.zero_embedding": False,
        "amago.nets.goal_embedders.TokenGoalEmb.goal_emb_dim": 64,
    }
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    make_env = lambda: CrafterEnv(
        directed=not args.default_rew,
        k=5,
        min_k=1,
        time_limit=2500,
        obs_kind=args.obs_kind,
        use_tech_tree=args.use_tech_tree,
    )

    group_name = f"{args.run_name}_{'undirected' if args.default_rew else 'directed'}_crafter_{args.obs_kind}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=2501,
            stagger_traj_file_lengths=False,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=5000,
            # Hindsight Relabeling!
            relabel=args.relabel,
            goal_importance_sampling=True,
            save_trajs_as="trajectory",
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)

        if not args.eval_mode:
            ###########
            ## Train ##
            ###########
            experiment.learn()

        else:
            ##########
            ## Test ##
            ##########
            # (including manual tasks used in Figures and Appendix tables)
            assert args.ckpt

            # full task distribution
            experiment.evaluate_test(make_env, timesteps=20_000, render=False)

            single_tasks = [
                "collect_sapling",
                "place_table",
                "collect_wood",
                "collect_stone",
                "collect_drink",
                "place_stone",
                "collect_coal",
                "defeat_zombie",
                "defeat_skeleton",
                "eat_cow",
                "collect_iron",
                "place_furnace",
                "collect_diamond",
                "make_wood_pickaxe",
                "make_wood_sword",
                "make_stone_pickaxe",
                "make_stone_sword",
                "make_iron_pickaxe",
                "make_iron_sword",
                "place_plant",
                "wake_up",
                "eat_plant",
            ]

            # fmt: off
            extra_tasks = [
                # add any task as a list of <= 5 subgoals here
                ["collect_sapling", "place_plant", "place_plant", "place_plant", "eat_cow"],
                ["travel_10m_10m", "place_stone", "travel_50m_50m", "place_stone"],
                ["make_stone_pickaxe", "collect_stone", "collect_stone", "collect_iron"],
                ["make_stone_pickaxe", "collect_iron"],
                ["make_stone_pickaxe", "collect_iron", "collect_iron"],
                ["collect_wood", "place_table", "make_wood_pickaxe", "collect_coal"],
                ["make_stone_pickaxe", "collect_coal", "collect_iron", "place_furnace", "make_iron_sword"],
                ["collect_drink", "eat_cow", "wake_up", "make_stone_sword", "defeat_zombie"],
                ["make_wood_sword", "defeat_zombie", "defeat_zombie"],
                ["eat_cow", "make_stone_pickaxe", "collect_coal", "make_wood_sword", "defeat_zombie"],
            ]
            # fmt: on

            TASKS = [[t] for t in single_tasks] + extra_tasks
            for i, task in enumerate(TASKS):

                def _make_env():
                    e = make_env()
                    # changing the name will create a new metric on wandb
                    e.set_env_name(f"crafter_eval_{'_'.join(task)}")
                    # manually set the task
                    e.set_fixed_task([t.split("_") for t in task])
                    return e

                experiment.evaluate_test(_make_env, timesteps=5 * 2501)

        experiment.delete_buffer_from_disk()
        wandb.finish()
