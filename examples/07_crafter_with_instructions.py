from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.crafter_envs import CrafterEnv
from utils import *


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
    """
    NOTE: most of the paper results used raw texture IDs to simplify the observation space and 
    allow for more ablations (Crafter's world generation causes `reset`s to be slow and
    stretches training times to the point where this was a necessary speedup). Unfortunately,
    the `EmbeddingTstepEncoder` was never ported to the version that became this open-source release.
    Adding this back is TODO.

    parser.add_argument(
        "--obs_kind", choices=["gridworld", "crop", "render"], default="render"
    )
    """
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
        dset_name = group_name + f"_trial_{trial}"

        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 5,
            dset_max_size=args.dset_max_size,
            run_name=dset_name,
            gpu=args.gpu,
            dset_root=args.buffer_dir,
            dset_name=dset_name,
            log_to_wandb=not args.no_log,
            epochs=args.epochs,
            parallel_actors=args.parallel_actors,
            train_timesteps_per_epoch=args.timesteps_per_epoch,
            train_grad_updates_per_epoch=args.grads_per_epoch,
            val_interval=args.val_interval,
            val_timesteps_per_epoch=0,
            ckpt_interval=args.ckpt_interval,
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
