import pytest

from oam_flux.lattice import TwistLattice
from oam_flux.pulse_train import PulseTrainConfig, run_vqc_pulse_train
from oam_flux.vqc_coupling import VQCCouplingState
from oam_flux.vqc_photonics import PhotonicsConfig


def test_pulse_train_config_total_steps():
    cfg = PulseTrainConfig(n_pulses=3, pump_steps=20, gap_steps=10)
    assert cfg.total_steps == 90


def test_pulse_train_multiple_injections():
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_rate=0.1)
    photonics = PhotonicsConfig(l_max=3, n_z=100, nr=48, lambda_nm=1550.0)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.15, "flywheel_sites": 2, "energy_scale": 1.0},
    )
    cfg = PulseTrainConfig(n_pulses=3, pump_steps=15, gap_steps=10)
    n = run_vqc_pulse_train(state, cfg)
    assert n == 75
    assert state.pulses_fired == 3
    assert state.total_injected == pytest.approx(3 * state.initial_total_momentum)
    assert any(h.get("recovery_active", 0) > 0.5 for h in state.history)
    pulse_ids = {int(h.get("pulse_index", -1)) for h in state.history}
    assert pulse_ids == {0, 1, 2}