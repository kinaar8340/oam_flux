import pytest

from oam_flux.dose_equivalence import (
    DoseEquivalenceConfig,
    compare_delivery_metrics,
    extract_delivery_metrics,
    run_continuous_dose_matched,
    run_pulsed_dose_matched,
)
from oam_flux.lattice import TwistLattice
from oam_flux.pulse_train import PulseTrainConfig
from oam_flux.vqc_coupling import VQCCouplingState
from oam_flux.vqc_photonics import PhotonicsConfig


def _make_state(**kwargs) -> VQCCouplingState:
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_tau=12.0)
    photonics = PhotonicsConfig(l_max=3, n_z=120, nr=48, lambda_nm=1550.0)
    cfg = {"kick_strength": 0.16, "flywheel_sites": 2, "energy_scale": 1.0, **kwargs}
    return VQCCouplingState.from_config(lattice, photonics, ell=2, coupling_cfg=cfg)


def test_continuous_matches_pulsed_total_injected():
    pcfg = PulseTrainConfig(n_pulses=3, pump_steps=18, gap_steps=12)
    state_cont = _make_state()
    state_pulse = _make_state()
    p0 = state_cont.initial_total_momentum
    horizon = pcfg.total_steps
    dcfg = DoseEquivalenceConfig(n_pulses=3, p0=p0, max_steps=horizon)

    run_continuous_dose_matched(state_cont, dcfg, pcfg)
    run_pulsed_dose_matched(state_pulse, pcfg)

    assert state_cont.total_injected == pytest.approx(state_pulse.total_injected)
    assert state_cont.total_injected == pytest.approx(3 * p0)
    assert state_pulse.total_injected == pytest.approx(3 * p0)


def test_continuous_shares_recovery_gap_schedule():
    pcfg = PulseTrainConfig(n_pulses=2, pump_steps=20, gap_steps=15, recovery_memory=0.6)
    state_cont = _make_state()
    state_pulse = _make_state()
    p0 = state_cont.initial_total_momentum
    dcfg = DoseEquivalenceConfig(n_pulses=2, p0=p0, max_steps=pcfg.total_steps)

    run_continuous_dose_matched(state_cont, dcfg, pcfg)
    run_pulsed_dose_matched(state_pulse, pcfg)

    cont_gaps = sum(
        1
        for h in state_cont.history
        if h.get("recovery_active", 0) > 0.5 and h.get("pulse_phase_pump", 0) < 0.5
    )
    pulse_gaps = sum(
        1
        for h in state_pulse.history
        if h.get("recovery_active", 0) > 0.5 and h.get("pulse_phase_pump", 0) < 0.5
    )
    expected_gaps = int(pcfg.gap_steps) * int(pcfg.n_pulses)
    assert cont_gaps == pytest.approx(expected_gaps)
    assert pulse_gaps == pytest.approx(expected_gaps)
    assert state_cont.recovery_memory == pytest.approx(0.6)


def test_continuous_twist_load_tracks_pulsed_with_memory():
    """With shared gaps and memory, lattice loading should stay close."""
    pcfg = PulseTrainConfig(
        n_pulses=3, pump_steps=18, gap_steps=12, recovery_memory=1.0, recovery_tau=8.0,
    )
    state_cont = _make_state()
    state_pulse = _make_state()
    p0 = state_cont.initial_total_momentum
    dcfg = DoseEquivalenceConfig(n_pulses=3, p0=p0, max_steps=pcfg.total_steps)

    run_continuous_dose_matched(state_cont, dcfg, pcfg)
    run_pulsed_dose_matched(state_pulse, pcfg)

    cont_load = extract_delivery_metrics(state_cont)["twist_load"]
    pulse_load = extract_delivery_metrics(state_pulse)["twist_load"]
    assert cont_load == pytest.approx(pulse_load, rel=0.05, abs=0.02)


def test_pulsed_has_structured_recovery_gaps():
    pcfg = PulseTrainConfig(n_pulses=2, pump_steps=20, gap_steps=15)
    state_pulse = _make_state()
    run_pulsed_dose_matched(state_pulse, pcfg)
    gap_steps_count = sum(
        1
        for h in state_pulse.history
        if h.get("recovery_active", 0) > 0.5 and h.get("pulse_phase_pump", 0) < 0.5
    )
    assert gap_steps_count >= int(pcfg.gap_steps) * int(pcfg.n_pulses) * 0.9


def test_compare_delivery_metrics_delta():
    cont = {
        "lattice_received": 0.4,
        "cumulative_phase_slip": 0.05,
        "eta_final": 0.8,
        "mean_twist_final": 1.2,
        "twist_load": 0.3,
        "recovery_steps": 10,
    }
    pulse = {
        "lattice_received": 0.35,
        "cumulative_phase_slip": 0.04,
        "eta_final": 0.85,
        "mean_twist_final": 1.1,
        "twist_load": 0.25,
        "recovery_steps": 30,
    }
    delta = compare_delivery_metrics(cont, pulse)
    assert delta["lattice_received"] == pytest.approx(-0.05)
    assert delta["recovery_steps"] == pytest.approx(20)