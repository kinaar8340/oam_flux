import pytest

from oam_flux.coupling import CouplingState, run_coupling_step
from oam_flux.lattice import TwistLattice
from oam_flux.momentum import conservation_check, oam_kinetic_momentum
from oam_flux.photon import OAMPacket


def test_oam_momentum_scales_with_ell():
    p3 = oam_kinetic_momentum(energy_scale=1.0, ell=3, lambda_nm=1550.0)
    p1 = oam_kinetic_momentum(energy_scale=1.0, ell=1, lambda_nm=1550.0)
    assert p3 > p1


def test_conservation_closed_transfer():
    photon = OAMPacket(ell=3, lambda_nm=1550.0, energy_scale=1.0)
    lattice = TwistLattice(nx=12, kappa=0.85)
    state = CouplingState(lattice=lattice, photon=photon, kick_strength=0.1)
    p0 = state.initial_total_momentum
    for step in range(20):
        run_coupling_step(state, step)
    last = state.history[-1]
    assert abs(last["conservation_residual"]) < 0.05
    assert last["total_momentum"] == pytest.approx(p0, rel=0.05)