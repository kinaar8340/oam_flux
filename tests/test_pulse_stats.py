import pytest

from oam_flux.lattice import TwistLattice
from oam_flux.pulse_stats import compute_pulse_statistics
from oam_flux.pulse_train import PulseTrainConfig, run_vqc_pulse_train
from oam_flux.vqc_coupling import VQCCouplingState
from oam_flux.vqc_photonics import PhotonicsConfig


def test_pulse_statistics_per_pulse_breakdown():
    lattice = TwistLattice(nx=10, kappa=0.85, recovery_rate=0.1)
    photonics = PhotonicsConfig(l_max=3, n_z=100, nr=48, lambda_nm=1550.0)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=2,
        coupling_cfg={"kick_strength": 0.15, "flywheel_sites": 2, "energy_scale": 1.0},
    )
    cfg = PulseTrainConfig(n_pulses=3, pump_steps=15, gap_steps=10)
    run_vqc_pulse_train(state, cfg)

    stats = compute_pulse_statistics(state.history)
    assert len(stats["pulses"]) == 3
    assert stats["cumulative"]["total_injected"] == pytest.approx(state.total_injected)
    assert stats["cumulative"]["total_phase_slip"] >= 0.0
    assert stats["cumulative"]["total_lattice_momentum"] >= 0.0
    for p in stats["pulses"]:
        assert 0.0 < p["eta_min"] <= 1.0
        assert p["lattice_deposited"] >= 0.0


def test_recovery_memory_carries_twist_load():
    lattice_full = TwistLattice(nx=10, kappa=0.85, recovery_rate=0.12)
    lattice_mem = TwistLattice(nx=10, kappa=0.85, recovery_rate=0.12)
    photonics = PhotonicsConfig(l_max=3, n_z=80, nr=48, lambda_nm=1550.0)
    cfg = {"kick_strength": 0.2, "flywheel_sites": 2, "energy_scale": 1.0}

    state_full = VQCCouplingState.from_config(
        lattice_full, photonics, ell=2, coupling_cfg=cfg,
    )
    state_mem = VQCCouplingState.from_config(
        lattice_mem, photonics, ell=2, coupling_cfg=cfg,
    )
    pcfg_full = PulseTrainConfig(n_pulses=2, pump_steps=20, gap_steps=15, recovery_memory=0.0)
    pcfg_mem = PulseTrainConfig(n_pulses=2, pump_steps=20, gap_steps=15, recovery_memory=1.0)
    run_vqc_pulse_train(state_full, pcfg_full)
    run_vqc_pulse_train(state_mem, pcfg_mem)

    load_full = state_full.lattice.twist_load_vs_initial()
    load_mem = state_mem.lattice.twist_load_vs_initial()
    assert load_mem > load_full