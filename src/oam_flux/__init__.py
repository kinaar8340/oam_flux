"""OAM–flux coupling: photon helical packets on gauged Hopf lattice flywheels."""

from .constants import LatticeConstants, load_config
from .lattice import TwistLattice, helical_seed
from .photon import OAMPacket
from .coupling import CouplingState, run_coupling_step

__all__ = [
    "LatticeConstants",
    "load_config",
    "TwistLattice",
    "helical_seed",
    "OAMPacket",
    "CouplingState",
    "run_coupling_step",
]

__version__ = "0.1.0"