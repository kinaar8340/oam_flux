import pytest

from oam_flux.lattice import TwistLattice
from oam_flux.pulse_envelope import build_pump_envelope, normalize_pulse_shape
from oam_flux.pulse_train import PulseTrainConfig, run_vqc_pulse_train
from oam_flux.vqc_coupling import VQCCouplingState
from oam_flux.vqc_photonics import PhotonicsConfig


def test_square_envelope_flat():
    env = build_pump_envelope("square", 20)
    assert len(env) == 20
    assert env == pytest.approx([1.0] * 20)


def test_gaussian_envelope_peaks_at_center():
    env = build_pump_envelope("gaussian", 31)
    assert env[len(env) // 2] == pytest.approx(1.0)
    assert env[0] < env[len(env) // 2]
    assert env[-1] < env[len(env) // 2]


def test_invalid_pulse_shape_raises():
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        normalize_pulse_shape("triangular")


def test_gaussian_records_envelope_in_history():
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_tau=12.0)
    photonics = PhotonicsConfig(l_max=3, n_z=80, nr=48, lambda_nm=1550.0)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.15, "flywheel_sites": 2, "energy_scale": 1.0},
    )
    cfg = PulseTrainConfig(
        n_pulses=2, pump_steps=21, gap_steps=8, pulse_shape="gaussian",
    )
    run_vqc_pulse_train(state, cfg)
    pump_rows = [
        h for h in state.history
        if h.get("pulse_phase_pump", 0) > 0.5 and h.get("pump_envelope", 0) > 0
    ]
    assert pump_rows
    envelopes = [h["pump_envelope"] for h in pump_rows]
    assert max(envelopes) == pytest.approx(1.0)
    assert state.pulse_shape == "gaussian"


def test_gaussian_deposits_less_than_square_same_window():
    cfg_base = {
        "kick_strength": 0.18,
        "flywheel_sites": 2,
        "energy_scale": 1.0,
    }
    photonics = PhotonicsConfig(l_max=3, n_z=60, nr=48, lambda_nm=1550.0)
    deposits: list[float] = []
    for shape in ("square", "gaussian"):
        lattice = TwistLattice(nx=10, kappa=0.85, recovery_tau=15.0)
        state = VQCCouplingState.from_config(
            lattice, photonics, ell=2, coupling_cfg=cfg_base,
        )
        cfg = PulseTrainConfig(
            n_pulses=1, pump_steps=25, gap_steps=5, pulse_shape=shape,
        )
        run_vqc_pulse_train(state, cfg)
        deposits.append(state.history[-1].get("lattice_received", 0.0))
    assert deposits[1] < deposits[0]