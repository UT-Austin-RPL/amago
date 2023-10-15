from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.ale_retro import RetroArcade, RetroAMAGOWrapper
from utils import *


"""
AMAGO can be used for more traditional multi-task RL that assumes
we do not have task labels. In this case we can use short
sequenes to identify the task. The RL^2 format (where past actions
are in the input) can be surprisingly helpful because video games
have actions that are more non-markov than frame-stacking usually gives 
them credit for. A good example of this is training on a group of 
Mario levels at once. Mario's actions are very dependent on how 
long buttons are held down for.

Performance on a given set of games/levels has a lot to do
with whether the scale of returns is similar across the set. 
Fixing this is future work.

This script requires extra installation: https://stable-retro.farama.org/getting_started/.
You'll also need to find/load the ROMs(s).
"""

MarioGameStarts = {
    "SuperMarioBros-Nes": [
        "Level1-1.state",
        "Level2-1-clouds.state",
        "Level2-1.state",
        "Level4-1.state",
        "Level6-1.state",
        "Level8-1.state",
        "Level1-4.state",
        "Level3-1.state",
        "Level5-1.state",
        "Level7-1.state",
    ],
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=128)
    add_common_cli(parser)
    args = parser.parse_args()

    config = {}
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    switch_tstep_encoder(config, arch="cnn", channels_first=True)
    use_config(config, args.configs)

    make_env = lambda: RetroAMAGOWrapper(
        RetroArcade(game_start_dict=MarioGameStarts, use_discrete_actions=True)
    )

    group_name = f"{args.run_name}_mario_retro_example"
    for trial in range(args.trials):
        dset_name = group_name + f"_trial_{trial}"

        experiment = amago.Experiment(
            make_train_env=make_env,
            make_val_env=make_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
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
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_env, timesteps=50_000, render=False)
        wandb.finish()
