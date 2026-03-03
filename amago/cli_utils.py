"""
Convenience functions that create a generic CLI for :class:`Experiment` and handle common gin configurations

These mostly exist to make the :file:`examples/` easier to maintain with less boilerplate,
and to break up configuration into several smaller steps.
"""

import os
from argparse import ArgumentParser
from typing import Optional

import gin
import wandb

import amago
from amago import TrajEncoder, TstepEncoder, Agent
from amago.agent import get_agent_cls, list_registered_agents
from amago.nets.traj_encoders import get_traj_encoder_cls, list_registered_traj_encoders
from amago.nets.tstep_encoders import (
    get_tstep_encoder_cls,
    list_registered_tstep_encoders,
)
from amago.loading import DiskTrajDataset, RLDataset
from amago.envs.exploration import (
    ExplorationWrapper,
    EpsilonGreedy,
    BilevelEpsilonGreedy,
    get_exploration_cls,
    list_registered_explorations,
)


class _LazyChoices:
    """A helper that defers evaluation of choices until argparse needs them."""

    def __init__(self, list_fn: callable):
        self._list_fn = list_fn

    def __contains__(self, item):
        return item in self._list_fn()

    def __iter__(self):
        return iter(self._list_fn())


def add_common_cli(parser: ArgumentParser) -> ArgumentParser:
    """Adds a common CLI for examples and basic training scripts.

    Args:
        parser: The argument parser containing problem-specific application-specific
            arguments.

    Returns:
        The argument parser with common CLI arguments added.
    """
    # extra gin configs
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        help="Extra `.gin` configuration files. These settings are usually added last in the examples and would overwrite the script's defaults.",
    )
    # basics
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--agent_type",
        type=str,
        default="agent",
        choices=_LazyChoices(list_registered_agents),
        help="Quick switch between registered Agent types. See `amago.agent.list_registered_agents()` for options. MultiTaskAgent is useful when training on mixed environments with multiple reward functions.",
    )
    parser.add_argument(
        "--env_mode",
        choices=["async", "sync", "already_vectorized"],
        default="async",
        help="`async` runs single-threaded environments in parallel, `sync` imitates the same parallel API but runs each step sequentially to save overhead, `already_vectorized` should be used for environments that are already parallelized at the lowest wrapper level (e.g. they're jax accelerated and have a batch dimension).",
    )
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
        help="Path to disk location where checkpoints (and, in most cases, replay buffers) will be stored. Should probably be somewhere with lots of space...",
    )
    # trajectory encoder
    parser.add_argument(
        "--traj_encoder",
        choices=_LazyChoices(list_registered_traj_encoders),
        default="transformer",
        help="Quick switch between registered TrajEncoders. See `amago.nets.traj_encoders.list_registered_traj_encoders()` for options. (ff == feedforward/memory-free)",
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
        "--batches_per_epoch",
        type=int,
        default=1000,
        help="Gradient updates per training epoch.",
    )
    parser.add_argument(
        "--timesteps_per_epoch",
        type=int,
        default=1000,
        help="Timesteps of environment interaction per epoch *per actor*. The update:data ratio is defined by `batches_per_epoch / (timesteps_per_epoch * parallel_actors)`.",
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
    parser.add_argument(
        "--start_learning_at_epoch",
        type=int,
        default=0,
        help="Skip learning updates for this many epochs at the beginning of training (if worried about overfitting to a small dataset)",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "bf16"],
        default="no",
        help="Train in bf16 mixed precision (requires a compatible GPU). Make sure to select this option during `accelerate config`.",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=8,
        help="Pytorch dataloader workers for loading trajectories from disk.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Training batch size (measured in trajectories, not timesteps).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["learn", "collect", "both"],
        help="Simple max-throughput async mode. Start the command with `--mode collect` 1+ times, and then start the same command with `--mode learner` in another terminal. Defaults to alternating collect/train steps.",
    )
    return parser


def switch_tstep_encoder(config: dict, arch: str, **kwargs) -> type[TstepEncoder]:
    """
    Set default kwargs for a TstepEncoder without gin syntax or config files.

    Args:
        config: A dictionary of gin parameters yet to be assigned.
        arch: A shortcut name for a registered TstepEncoder (e.g., "ff", "cnn").
            See `amago.nets.tstep_encoders.list_registered_tstep_encoders()` for options.
        **kwargs: Assign any of the chosen TstepEncoder's default kwargs.

    Returns:
        A reference to the TstepEncoder type that can be passed into the
        Experiment.

    Example:
        .. code-block:: python

            config = {}
            # Make the input MLP smaller
            tstep_encoder_type = switch_tstep_encoder(
                config, "ff", n_layers=1, d_hidden=128, d_output=128
            )
            cli_utils.use_config(config) # set new default parameters
            experiment = Experiment(
                ...,  # rest of args
                tstep_encoder_type=tstep_encoder_type,
            )
    """
    tstep_encoder_type = get_tstep_encoder_cls(arch)
    encoder_config = f"{tstep_encoder_type.__module__}.{tstep_encoder_type.__name__}"
    config.update({f"{encoder_config}.{key}": val for key, val in kwargs.items()})
    return tstep_encoder_type


def switch_agent(config: dict, agent: str, **kwargs) -> type[Agent]:
    """
    Set default kwargs for a built-in Agent without gin syntax or config files.

    Args:
        config: A dictionary of gin parameters yet to be assigned.
        agent: A shortcut name for a registered Agent (e.g., "agent", "multitask").
            See `amago.agent.list_registered_agents()` for available options.
        **kwargs: Assign any of the chosen Agent's default kwargs.

    Returns:
        A reference to the Agent type that can be passed into the Experiment.
    """
    agent_type = get_agent_cls(agent)
    # Build the gin config path from the class's module and name
    agent_config = f"{agent_type.__module__}.{agent_type.__name__}"
    config.update({f"{agent_config}.{key}": val for key, val in kwargs.items()})
    return agent_type


def switch_exploration(
    config: dict, strategy: str, **kwargs
) -> type[ExplorationWrapper]:
    """
    Set default kwargs for a built-in ExplorationWrapper without gin syntax or
    config files.

    Args:
        config: A dictionary of gin parameters yet to be assigned.
        strategy: A shortcut name for a registered ExplorationWrapper (e.g., "egreedy", "bilevel").
            See `amago.envs.exploration.list_registered_explorations()` for options.
        **kwargs: Assign any of the chosen ExplorationWrapper's default kwargs.

    Returns:
        A reference to the ExplorationWrapper type that can be passed into the
        Experiment.
    """
    strategy_type = get_exploration_cls(strategy)
    strategy_config = f"{strategy_type.__module__}.{strategy_type.__name__}"
    config.update({f"{strategy_config}.{key}": val for key, val in kwargs.items()})
    return strategy_type


def switch_traj_encoder(
    config: dict, arch: str, memory_size: int, layers: int, **kwargs
) -> type[TrajEncoder]:
    """
    Set default kwargs for a built-in TrajEncoder without gin syntax or config files.

    Args:
        config: A dictionary of gin parameters yet to be assigned.
        arch: A shortcut name for a registered TrajEncoder (e.g., "ff", "rnn", "transformer", "mamba").
            See `amago.nets.traj_encoders.list_registered_traj_encoders()` for options.
        memory_size: Sets the same conceptual state space dimension across the
            various architectures. For example, the size of the hidden state in an
            RNN or d_model in a Transformer.
        layers: Sets the number of layers in the TrajEncoder.
        **kwargs: Assign any of the chosen TrajEncoder's default kwargs.

    Returns:
        A reference to the TrajEncoder type that can be passed into the Experiment.
    """
    traj_encoder_type = get_traj_encoder_cls(arch)
    model_config = f"{traj_encoder_type.__module__}.{traj_encoder_type.__name__}"

    # Map memory_size and layers to architecture-specific param names
    # (Built-in encoders have different conventions for these params)
    if arch == "transformer":
        config.update(
            {
                f"{model_config}.d_model": memory_size,
                f"{model_config}.d_ff": memory_size * 4,
                f"{model_config}.n_layers": layers,
            }
        )
    elif arch == "rnn":
        config.update(
            {
                f"{model_config}.n_layers": layers,
                f"{model_config}.d_output": memory_size,
                f"{model_config}.d_hidden": memory_size,
            }
        )
    elif arch in ("ff", "mamba"):
        config.update(
            {
                f"{model_config}.d_model": memory_size,
                f"{model_config}.n_layers": layers,
            }
        )
    # For custom registered encoders, pass memory_size and layers as-is
    # (they can be overridden via **kwargs if needed)
    else:
        config.update(
            {
                f"{model_config}.memory_size": memory_size,
                f"{model_config}.n_layers": layers,
            }
        )

    config.update({f"{model_config}.{key}": val for key, val in kwargs.items()})
    return traj_encoder_type


def use_config(
    custom_params: dict, gin_configs: list[str] | None = None, finalize: bool = True
) -> None:
    """
    Bind gin parameters to edit kwarg defaults across the codebase before training
    begins.

    Args:
        custom_params: A dictionary of gin parameters to bind ({param:
            new_default_value}). This was probably created within the training script
            or from a few command line args.
        gin_configs: An optional list of .gin configuration files to use. Gin files
            are the correct way to handle configs for real projects... unlike the
            example scripts.
        finalize: If True, finalize/freeze the gin config to prevent later changes.
            Defaults to True.
    """
    for param, val in custom_params.items():
        gin.bind_parameter(param, val)
    # override defaults with custom gin config files
    if gin_configs is not None:
        for config in gin_configs:
            gin.parse_config_file(config)
    if finalize:
        gin.finalize()


def create_experiment_from_cli(
    command_line_args,
    make_train_env: callable,
    make_val_env: callable,
    max_seq_len: int,
    group_name: str,
    run_name: str,
    agent_type: type[Agent],
    tstep_encoder_type: type[TstepEncoder],
    traj_encoder_type: type[TrajEncoder],
    traj_save_len: Optional[int] = None,
    exploration_wrapper_type: type[ExplorationWrapper] = EpsilonGreedy,
    experiment_type: type[amago.Experiment] = amago.Experiment,
    dataset: Optional[RLDataset] = None,
    **extra_experiment_kwargs,
) -> amago.Experiment:
    """
    A convenience function that assigns Experiment kwargs from
    :py:func:`~amago.cli_utils.add_common_cli()` options.

    Args:
        command_line_args: The parsed command line arguments created by
            `cli_utils.add_common_cli()`.
        make_train_env: A callable that makes the training environment.
        make_val_env: A callable that makes the validation environment.
        max_seq_len: The maximum sequence length of the policy during training.
        group_name: The name of the wandb group to use for logging.
        run_name: The name of the run for logging & checkpoints.
        agent_type: The type of agent to use. Can be the output of
            `cli_utils.switch_agent()`.
        tstep_encoder_type: The type of tstep encoder to use. Can be the output of
            `cli_utils.switch_tstep_encoder()`.
        traj_encoder_type: The type of traj encoder to use. Can be the output of
            `cli_utils.switch_traj_encoder()`.
        traj_save_len: The length of the trajectory to save. Defaults to a very large
            number (which saves entire trajectories on terminated or truncated).
        exploration_wrapper_type: The type of exploration wrapper to use. Can be the
            output of `cli_utils.switch_exploration()`, but defaults to
            `EpsilonGreedy`.
        experiment_type: The type of experiment to use. Defaults to `amago.Experiment`.
        dataset: An optional dataset to use. If not provided, we create a
            `DiskTrajDataset` (an online RL replay buffer on disk) in the same
            directory where the CLI tells us it will save checkpoints
            ({args.buffer_dir}/{args.run_name}).
        **extra_experiment_kwargs: Additional keyword arguments to pass to the
            Experiment constructor.

    Returns:
        An Experiment instance.
    """

    cli = command_line_args

    traj_save_len = traj_save_len or 1e10

    if dataset is None:
        # create a new-style dataset in the place
        # where all the existing examples assume the dataset will be
        dataset = DiskTrajDataset(
            dset_root=cli.buffer_dir,
            dset_name=run_name,
            dset_max_size=cli.dset_max_size,
        )

    experiment = experiment_type(
        agent_type=agent_type,
        tstep_encoder_type=tstep_encoder_type,
        traj_encoder_type=traj_encoder_type,
        dataset=dataset,
        ckpt_base_dir=cli.buffer_dir,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        max_seq_len=max_seq_len,
        traj_save_len=traj_save_len,
        exploration_wrapper_type=exploration_wrapper_type,
        run_name=run_name,
        dloader_workers=cli.dloader_workers,
        log_to_wandb=not cli.no_log,
        wandb_group_name=group_name,
        batch_size=cli.batch_size,
        epochs=cli.epochs,
        parallel_actors=cli.parallel_actors,
        train_timesteps_per_epoch=cli.timesteps_per_epoch,
        train_batches_per_epoch=cli.batches_per_epoch,
        start_learning_at_epoch=cli.start_learning_at_epoch,
        val_interval=cli.val_interval,
        ckpt_interval=cli.ckpt_interval,
        mixed_precision=cli.mixed_precision,
        env_mode=cli.env_mode,
        **extra_experiment_kwargs,
    )

    return experiment


def make_experiment_learn_only(experiment: amago.Experiment) -> amago.Experiment:
    """
    Modify the experiment to run in learn-only mode.

    Args:
        experiment: The experiment to modify.

    Returns:
        The modified experiment.
    """
    experiment.start_collecting_at_epoch = float("inf")
    experiment.train_timesteps_per_epoch = 0
    experiment.val_interval = 10
    experiment.val_timesteps_per_epoch = 0
    # might throw warnings but cuts overhead of building envs we'll never use
    experiment.parallel_actors = 1
    experiment.always_save_latest = True
    experiment.always_load_latest = False
    return experiment


def make_experiment_collect_only(experiment: amago.Experiment) -> amago.Experiment:
    """
    Modify the experiment to run in collect-only mode.

    Args:
        experiment: The experiment to modify.

    Returns:
        The modified experiment.
    """
    experiment.start_collecting_at_epoch = 0
    experiment.start_learning_at_epoch = float("inf")
    experiment.train_batches_per_epoch = 0
    experiment.ckpt_interval = None
    experiment.always_save_latest = False
    experiment.always_load_latest = True
    # run "forever"; terminate manually (when learning process is done)
    experiment.epochs = max(experiment.epochs, 1_000_000)
    # do not delete anything from the collection process
    experiment.has_dset_edit_rights = False
    experiment.init_dsets()
    return experiment


def switch_async_mode(experiment: amago.Experiment, mode: str) -> amago.Experiment:
    """
    Switch the experiment mode between collect, learn, or both.

    Args:
        experiment: The experiment to modify.
        mode: The mode to switch to. Options are "collect", "learn", or "both".

    Returns:
        The modified experiment.
    """
    assert mode in ["collect", "learn", "both"]
    if mode == "collect":
        experiment = make_experiment_collect_only(experiment)
    elif mode == "learn":
        experiment = make_experiment_learn_only(experiment)
    return experiment
