import numpy as np

from oam_flux.flux_deposit import build_flux_kick, hopf_fiber_coords
from oam_flux.lattice import TwistLattice
from oam_flux.vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from oam_flux.vqc_photonics import PhotonicsConfig, propagate_multi_ell_vectorized


def test_propagation_shape():
    cfg = PhotonicsConfig(l_max=4, n_z=50, nr=128)
    result = propagate_multi_ell_vectorized(cfg)
    assert result.intensity.shape == (50, 9)
    assert result.ells.shape == (9,)
    assert result.radial_weights.shape == (9, 128)


def test_ell_mode_lookup():
    result = propagate_multi_ell_vectorized(PhotonicsConfig(l_max=3, n_z=10, nr=64))
    assert result.ell_index(2) == 5  # [-3,-2,-1,0,1,2,...]
    assert result.mode_intensity(0, 0) >= 0.0


def test_hopf_coords_cover_torus():
    rho, phi, eta = hopf_fiber_coords(16)
    assert rho.shape == (16, 16, 16)
    assert np.all(rho >= 0)
    assert np.all((eta >= 0) & (eta <= 2 * np.pi))


def test_vqc_coupling_advances_z():
    lattice = TwistLattice(nx=12, dt=0.001)
    photonics = PhotonicsConfig(l_max=3, n_z=20, nr=64)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.05, "flywheel_sites": 2, "conserve_momentum": True},
    )
    z0 = state.z_index
    run_vqc_coupling_step(state, 0)
    assert state.z_index == z0 + 1
    assert len(state.history) == 1


def test_flux_kick_nonzero_for_active_mode():
    lattice = TwistLattice(nx=12)
    prop = propagate_multi_ell_vectorized(PhotonicsConfig(l_max=3, n_z=10, nr=64))
    kick, momentum = build_flux_kick(lattice, prop, ell=1, z_index=0, kick_strength=0.1)
    assert np.any(kick != 0)
    assert momentum > 0