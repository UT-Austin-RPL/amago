from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.crafter_envs import CrafterEnv
from example_utils import *


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
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # no need to risk numerical instability when returns are this bounded
        "amago.agent.Agent.reward_multiplier": 10.0,
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
    switch_tstep_encoder(config, arch="cnn", channels_first=False)
    use_config(config, args.configs)

    make_env = lambda: CrafterEnv(
        directed=not args.default_rew,
        k=5,
        min_k=1,
        time_limit=5000,
        obs_kind="render",
        # obs_kind=args.obs_kind,
        use_tech_tree=args.use_tech_tree,
    )

    group_name = (
        f"{args.run_name}_{'undirected' if args.default_rew else 'directed'}_crafter"
    )
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=5000,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=5000,
            # Hindsight Relabeling!
            relabel=args.relabel,
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_env, timesteps=50_000, render=False)
        wandb.finish()
