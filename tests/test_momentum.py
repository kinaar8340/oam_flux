import pytest

from oam_flux.coupling import CouplingState, run_coupling_step
from oam_flux.lattice import TwistLattice
from oam_flux.momentum import (
    DEFAULT_CONSERVATION_TOLERANCE,
    conservation_check,
    energy_scale_from_ev,
    is_momentum_conserved,
    lambda_nm_from_ev,
    oam_kinetic_momentum,
    photon_energy_ev,
    photon_state,
)
from oam_flux.photon import OAMPacket


def test_oam_momentum_scales_with_ell():
    p3 = oam_kinetic_momentum(energy_scale=1.0, ell=3, lambda_nm=1550.0)
    p1 = oam_kinetic_momentum(energy_scale=1.0, ell=1, lambda_nm=1550.0)
    assert p3 > p1
    assert p3 == pytest.approx(3.0 * p1)


def test_oam_momentum_hbar_scaling_at_1550nm():
    p = oam_kinetic_momentum(energy_scale=1.0, ell=1, lambda_nm=1550.0)
    assert p == pytest.approx(0.4275, rel=0.01)


def test_conservation_tolerance_band():
    p0 = 1.0
    assert is_momentum_conserved(0.004 * p0, p0)
    assert not is_momentum_conserved(0.006 * p0, p0)
    check = conservation_check(photon_p=0.5, ledger=-0.496, initial_total=p0)
    assert check["conserved"] is True
    assert DEFAULT_CONSERVATION_TOLERANCE == 0.005


def test_photon_energy_at_1550nm():
    e = photon_energy_ev(lambda_nm=1550.0, energy_scale=1.0)
    assert e == pytest.approx(0.7999, rel=0.01)


def test_energy_lambda_coupling():
    e = photon_energy_ev(lambda_nm=800.0, energy_scale=1.0)
    lam = lambda_nm_from_ev(energy_ev=e, energy_scale=1.0)
    assert lam == pytest.approx(800.0, rel=0.01)
    scale = energy_scale_from_ev(energy_ev=2.0 * e, lambda_nm=800.0)
    assert scale == pytest.approx(2.0, rel=1e-6)


def test_photon_state_natural_units():
    st = photon_state(ell=3, lambda_nm=1550.0, energy_ev=photon_energy_ev(lambda_nm=1550.0))
    assert st["momentum_natural"] == pytest.approx(3.0, rel=1e-6)
    assert st["momentum"] == pytest.approx(3.0 * 0.4275, rel=0.02)


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