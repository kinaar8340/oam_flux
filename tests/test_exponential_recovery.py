import numpy as np
import pytest

from oam_flux.lattice import (
    DEFAULT_RECOVERY_TAU,
    TwistLattice,
    recovery_step_alpha,
    tau_from_step_rate,
)
from oam_flux.pulse_train import PulseTrainConfig, run_vqc_pulse_train
from oam_flux.vqc_coupling import VQCCouplingState
from oam_flux.vqc_photonics import PhotonicsConfig


def test_tau_from_step_rate_matches_legacy_default():
    assert tau_from_step_rate(0.04) == pytest.approx(DEFAULT_RECOVERY_TAU, rel=0.05)


def test_recovery_step_alpha_one_efolding():
    alpha = recovery_step_alpha(tau=10.0)
    assert alpha == pytest.approx(1.0 - np.exp(-0.1), rel=1e-6)


def test_exponential_recovery_one_tau():
    tau = 12.0
    lattice = TwistLattice(nx=6, kappa=0.85, recovery_tau=tau)
    lattice.theta = np.clip(lattice.theta_initial + 1.8, 0.01, 2 * np.pi - 0.01)
    load0 = lattice.twist_load_vs_initial()
    for _ in range(int(tau)):
        lattice.relax_step(pump_active=False)
    load1 = lattice.twist_load_vs_initial()
    assert load1 == pytest.approx(load0 / np.e, rel=0.12)


def test_larger_tau_slower_recovery():
    tau_fast, tau_slow = 8.0, 50.0
    loads: list[float] = []
    for tau in (tau_fast, tau_slow):
        lattice = TwistLattice(nx=6, kappa=0.85, recovery_tau=tau)
        lattice.theta = np.clip(lattice.theta_initial + 1.5, 0.01, 2 * np.pi - 0.01)
        for _ in range(20):
            lattice.relax_step(pump_active=False)
        loads.append(lattice.twist_load_vs_initial())
    assert loads[0] < loads[1]


def test_recovery_tau_wired_through_pulse_train():
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_tau=15.0)
    photonics = PhotonicsConfig(l_max=3, n_z=60, nr=48, lambda_nm=1550.0)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.12, "flywheel_sites": 2, "energy_scale": 1.0},
    )
    cfg = PulseTrainConfig(
        n_pulses=2, pump_steps=10, gap_steps=12, recovery_tau=33.0,
    )
    run_vqc_pulse_train(state, cfg)
    assert state.recovery_tau == pytest.approx(33.0)
    assert state.lattice.recovery_tau == pytest.approx(33.0)
    assert state.history[-1].get("recovery_tau") == pytest.approx(33.0)