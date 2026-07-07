"""OAM–flux coupling: photon helical packets on gauged Hopf lattice flywheels."""

from .constants import LatticeConstants, load_config
from .coupling import CouplingState, run_coupling_step
from .flux_deposit import build_flux_kick, deposit_on_flywheels, hopf_fiber_coords
from .lattice import TwistLattice, helical_seed
from .photon import OAMPacket
from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from .emergence import (
    EmergenceAnalogs,
    ell_sweep,
    emergence_report,
    kappa_sweep,
    run_vqc_emergence_trial,
)
from .eddington import run_eddington_probe
from .helix_viz import build_helix_geometry, render_helix_frame
from .vqc_photonics import PhotonicsConfig, PropagationResult, propagate_multi_ell_vectorized

__all__ = [
    "LatticeConstants",
    "load_config",
    "TwistLattice",
    "helical_seed",
    "OAMPacket",
    "CouplingState",
    "run_coupling_step",
    "PhotonicsConfig",
    "PropagationResult",
    "propagate_multi_ell_vectorized",
    "hopf_fiber_coords",
    "build_flux_kick",
    "deposit_on_flywheels",
    "VQCCouplingState",
    "run_vqc_coupling_step",
    "EmergenceAnalogs",
    "run_vqc_emergence_trial",
    "kappa_sweep",
    "ell_sweep",
    "emergence_report",
    "build_helix_geometry",
    "render_helix_frame",
    "run_eddington_probe",
]

__version__ = "0.4.0"