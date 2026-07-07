import numpy as np
import pytest

from oam_flux.back_reaction import photon_pump_active
from oam_flux.lattice import TwistLattice
from oam_flux.vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from oam_flux.vqc_photonics import PhotonicsConfig


def test_photon_pump_active_threshold():
    assert photon_pump_active(0.05, 1.0, fraction=0.01)
    assert not photon_pump_active(0.005, 1.0, fraction=0.01)


def test_lattice_recovery_relaxes_twist():
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_rate=0.15)
    lattice.theta = lattice.theta + 1.5
    load_before = lattice.twist_load_vs_initial()
    for _ in range(30):
        lattice.relax_step(pump_active=False)
    assert lattice.twist_load_vs_initial() < load_before


def test_vqc_recovery_after_pump_depleted():
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_rate=0.12)
    photonics = PhotonicsConfig(l_max=3, n_z=40, nr=48, lambda_nm=1550.0)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.25, "flywheel_sites": 2, "energy_scale": 1.0},
    )
    eta_min = 1.0
    for step in range(40):
        run_vqc_coupling_step(state, step)
        eta_min = min(eta_min, state.history[-1].get("back_reaction_coupling", 1.0))
    last = state.history[-1]
    assert last.get("recovery_steps", 0) > 0
    assert last.get("back_reaction_coupling", 0) >= eta_min