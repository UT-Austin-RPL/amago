from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("amago")
except PackageNotFoundError:
    __version__ = "unknown"

from .experiment import Experiment
from .agent import Agent, register_agent, get_agent_cls, list_registered_agents
from .nets import (
    TstepEncoder,
    TrajEncoder,
    register_tstep_encoder,
    get_tstep_encoder_cls,
    list_registered_tstep_encoders,
    register_traj_encoder,
    get_traj_encoder_cls,
    list_registered_traj_encoders,
)
from .envs import (
    ExplorationWrapper,
    register_exploration,
    get_exploration_cls,
    list_registered_explorations,
)
from . import envs
from . import nets
from . import cli_utils
