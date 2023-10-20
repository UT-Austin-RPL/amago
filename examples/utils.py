import gin
from argparse import ArgumentParser

import amago


def add_common_cli(parser: ArgumentParser) -> ArgumentParser:
    # extra gin configs
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        help="Extra `.gin` configuration files. These settings are usually added last in the examples and would overwrite the script's defaults.",
    )
    # basics
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Turn off wandb logging (usually for debugging).",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Start training from an epoch checkpoint saved in a buffer with the same `--run_name`",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Give the run a name. Used for logging and the disk replay buffer. Experiments with the same run_name share the same replay buffer, but log separately.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--buffer_dir",
        type=str,
        required=True,
        help="Path to disk location where replay buffer (and checkpoints) will be stored. Should probably be somewhere with lots of space...",
    )
    # trajectory encoder
    parser.add_argument(
        "--traj_encoder", choices=["ff", "transformer", "rnn"], default="transformer"
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=256,
        help="Model/token dimension for a Transformer; hidden state size for an RNN.",
    )
    parser.add_argument(
        "--memory_layers",
        type=int,
        default=3,
        help="Number of layers in the sequence model.",
    )
    # main learning schedule
    parser.add_argument(
        "--grads_per_epoch",
        type=int,
        default=1000,
        help="Gradient updates per training epoch.",
    )
    parser.add_argument(
        "--timesteps_per_epoch",
        type=int,
        default=1000,
        help="Timesteps of environment interaction per epoch. The update:data ratio is defined by `grads_per_epoch / (timesteps_per_epoch * parallel_actors)`.",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=20,
        help="How often (in epochs) to evaluate the agent on validation envs.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=50,
        help="How often (in epochs) to save an agent checkpoint.",
    )
    parser.add_argument(
        "--parallel_actors",
        type=int,
        default=12,
        help="Number of parallel environments (applies to training, validation, and testing).",
    )
    parser.add_argument(
        "--dset_max_size",
        type=int,
        default=20_000,
        help="Maximum size of the replay buffer (measured in trajectories, not timesteps).",
    )
    return parser


"""
Gin convenience functions.

Switch between the most common configurations without needing `.gin` config files.
"""


def switch_tstep_encoder(config, arch: str, **kwargs):
    """
    Convenient way to switch between TstepEncoders without gin config files

    `kwargs` should be the names and new defaults of kwargs in the TstepEncoder we
    are using (either `FFTstepEncoder` if arch == "ff", or `CNNTstepEncoder` if arch == "cnn")
    """
    assert arch in ["ff", "cnn"]
    if arch == "ff":
        config[
            "amago.agent.Agent.tstep_encoder_Cls"
        ] = amago.nets.tstep_encoders.FFTstepEncoder
        ff_config = "amago.nets.tstep_encoders.FFTstepEncoder"
        config.update({f"{ff_config}.{key}": val for key, val in kwargs.items()})
    elif arch == "cnn":
        config[
            "amago.agent.Agent.tstep_encoder_Cls"
        ] = amago.nets.tstep_encoders.CNNTstepEncoder
        cnn_config = "amago.nets.tstep_encoders.CNNTstepEncoder"
        config.update({f"{cnn_config}.{key}": val for key, val in kwargs.items()})
    return config


def turn_off_goal_conditioning(config: dict):
    """
    Make the goal embedding network redundant when the environment is not goal-conditioned.
    This applies to most environments that do not use hindsight relabeling.
    """
    config.update(
        {
            "amago.nets.tstep_encoders.TstepEncoder.goal_emb_Cls": amago.nets.goal_embedders.FFGoalEmb,
            "amago.nets.goal_embedders.FFGoalEmb.zero_embedding": True,
            "amago.nets.goal_embedders.FFGoalEmb.goal_emb_dim": 1,
        }
    )
    return config


def switch_traj_encoder(config: dict, arch: str, memory_size: int, layers: int):
    """
    Convenient way to switch between TrajEncoders of different sizes without gin config files.
    """
    assert arch in ["ff", "rnn", "transformer"]
    if arch == "transformer":
        tformer_config = "amago.nets.traj_encoders.TformerTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.TformerTrajEncoder,
                f"{tformer_config}.d_model": memory_size,
                f"{tformer_config}.d_ff": memory_size * 4,
                f"{tformer_config}.n_layers": layers,
            }
        )
    elif arch == "rnn":
        gru_config = "amago.nets.traj_encoders.GRUTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.GRUTrajEncoder,
                f"{gru_config}.n_layers": layers,
                f"{gru_config}.d_output": memory_size,
                f"{gru_config}.d_hidden": memory_size,
            }
        )

    elif arch == "ff":
        ff_config = "amago.nets.traj_encoders.FFTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.FFTrajEncoder,
                f"{ff_config}.d_model": memory_size,
                f"{ff_config}.n_layers": layers,
            }
        )
    return config


def naive(config : dict):
    config.update({
        "amago.nets.traj_encoders.TformerTrajEncoder.activation" : "gelu",
        "amago.nets.actor_critic.NCritics.activation" : "relu",
        "amago.nets.actor_critic.Actor.activation" : "relu",
        "amago.nets.tstep_encoders.FFTstepEncoder.activation" : "relu",
        "amago.nets.tstep_encoders.CNNTstepEncoder.activation" : "relu",
        "amago.nets.transformer.TransformerLayer.normformer_norms" : False,
        "amago.nets.transformer.TransformerLayer.sigma_reparam" : False,
        "amago.nets.transformer.AttentionLayer.sigma_reparam" : False,
        "amago.nets.transformer.AttentionLayer.head_scaling" : False,
        "amago.agent.Agent.num_critics" : 2,
        "amago.agent.Agent.gamma" : .99,
        "amago.agent.Agent.use_multigamma" : False,
    })
    return config



def use_config(custom_params: dict, gin_configs: list[str] | None = None):
    """
    Bind all the gin parameters from real .gin configs (which the examples avoid using)
    and regular dictionaries.

    Use before training begins.
    """
    for param, val in custom_params.items():
        gin.bind_parameter(param, val)
    # override defaults with custom gin config files
    if gin_configs is not None:
        for config in gin_configs:
            gin.parse_config_file(config)
    gin.finalize()
