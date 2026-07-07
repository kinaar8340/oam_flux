import numpy as np
import pytest

from oam_flux.back_reaction import apply_phase_slip, lattice_back_reaction
from oam_flux.coupling import CouplingState, run_coupling_step
from oam_flux.lattice import TwistLattice
from oam_flux.photon import OAMPacket
from oam_flux.vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from oam_flux.vqc_photonics import PhotonicsConfig


def test_back_reaction_weak_at_low_twist():
    lattice = TwistLattice(nx=12, kappa=0.85)
    br = lattice_back_reaction(lattice, ell=3)
    assert br["coupling_factor"] == pytest.approx(1.0, abs=0.15)
    assert br["phase_slip_fraction"] >= 0.0


def test_back_reaction_suppresses_coupling_at_high_twist():
    lattice = TwistLattice(nx=12, kappa=0.85)
    lattice.theta = np.clip(lattice.theta + 4.0, 0.01, 2 * np.pi - 0.01)
    br = lattice_back_reaction(lattice, ell=2)
    assert br["coupling_factor"] < 0.9
    assert br["phase_slip_fraction"] > 0.0


def test_vqc_step_records_back_reaction():
    lattice = TwistLattice(nx=12, kappa=0.85)
    photonics = PhotonicsConfig(l_max=4, n_z=8, nr=64, lambda_nm=1550.0)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.1, "flywheel_sites": 2, "energy_scale": 1.0},
    )
    for step in range(5):
        run_vqc_coupling_step(state, step)
    last = state.history[-1]
    assert "back_reaction_coupling" in last
    assert last["back_reaction_coupling"] <= 1.0


def test_analytic_back_reaction_conserves_momentum():
    photon = OAMPacket(ell=3, lambda_nm=1550.0, energy_scale=1.0)
    lattice = TwistLattice(nx=12, kappa=0.85)
    state = CouplingState(lattice=lattice, photon=photon, kick_strength=0.12)
    p0 = state.initial_total_momentum
    for step in range(25):
        run_coupling_step(state, step)
    last = state.history[-1]
    assert abs(last["conservation_residual"]) < 0.08 * p0
    assert last.get("cumulative_phase_slip", 0.0) >= 0.0