"""Match continuous vs pulsed delivery at equal total injected momentum."""

from __future__ import annotations

from dataclasses import dataclass

from .back_reaction import photon_pump_active
from .pulse_train import PulseTrainConfig, run_vqc_pulse_train
from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step


@dataclass
class DoseEquivalenceConfig:
    """Target dose and horizon for continuous ↔ pulsed comparison."""

    n_pulses: int
    p0: float
    max_steps: int

    @property
    def target_dose(self) -> float:
        return float(self.n_pulses) * float(self.p0)


def _maybe_refill_continuous(state: VQCCouplingState, target_dose: float) -> None:
    """Refill when depleted if the target dose has not yet been delivered."""
    if (
        state.total_injected < target_dose
        and not photon_pump_active(state.photon_reservoir, state.initial_total_momentum)
    ):
        state.refill_reservoir()


def run_continuous_dose_matched(
    state: VQCCouplingState,
    cfg: DoseEquivalenceConfig,
    pcfg: PulseTrainConfig,
) -> int:
    """
    Deliver ``n_pulses × p₀`` on the same pump/gap schedule as the pulsed path.

    During each pump window the reservoir refills immediately when depleted
    (back-to-back packets). Gap windows use the same ``recovery_memory`` and
    ``recovery_tau`` as the pulsed train so both paths see equivalent lattice
    recovery physics.
    """
    step = 0
    state.pulses_fired = 0
    state.total_injected = 0.0
    state.pulse_train_mode = True
    state.equivalence_mode = True
    state.delivery_mode = "continuous"
    state.target_total_dose = cfg.target_dose
    state.recovery_memory = float(pcfg.recovery_memory)
    state.recovery_tau = float(pcfg.recovery_tau)
    state.lattice.recovery_tau = float(pcfg.recovery_tau)
    state.pulse_shape = "square"
    state.pump_envelope_factor = 1.0

    if state.total_injected < cfg.target_dose:
        state.refill_reservoir()

    for pulse in range(int(pcfg.n_pulses)):
        state.current_pulse = pulse
        state.pulse_phase = "pump"
        for _ in range(int(pcfg.pump_steps)):
            state.pump_envelope_factor = 1.0
            run_vqc_coupling_step(state, step)
            _maybe_refill_continuous(state, cfg.target_dose)
            step += 1

        state.photon_reservoir = 0.0
        state.pump_envelope_factor = 0.0
        state.pulse_phase = "recovery"
        for _ in range(int(pcfg.gap_steps)):
            run_vqc_coupling_step(state, step)
            step += 1

    while step < int(cfg.max_steps):
        state.photon_reservoir = 0.0
        state.pump_envelope_factor = 0.0
        state.pulse_phase = "recovery"
        run_vqc_coupling_step(state, step)
        step += 1

    state.pulse_phase = "done"
    state.pump_envelope_factor = 0.0
    return step


def run_pulsed_dose_matched(
    state: VQCCouplingState,
    pcfg: PulseTrainConfig,
) -> int:
    """Run a pulse train and tag state for equivalence comparison."""
    state.equivalence_mode = True
    state.delivery_mode = "pulsed"
    state.target_total_dose = float(pcfg.n_pulses) * state.initial_total_momentum
    return run_vqc_pulse_train(state, pcfg)


def extract_delivery_metrics(state: VQCCouplingState) -> dict[str, float]:
    """Final observables for equivalence comparison."""
    last = state.history[-1] if state.history else {}
    return {
        "delivery_mode": state.delivery_mode,
        "total_injected": float(state.total_injected),
        "target_dose": float(getattr(state, "target_total_dose", state.total_injected)),
        "pulses_fired": float(state.pulses_fired),
        "lattice_received": float(last.get("lattice_received", 0.0)),
        "cumulative_phase_slip": float(last.get("cumulative_phase_slip", 0.0)),
        "eta_final": float(last.get("back_reaction_coupling", 1.0)),
        "mean_twist_final": float(state.lattice.mean_twist),
        "twist_load": float(state.lattice.twist_load_vs_initial()),
        "recovery_steps": float(last.get("recovery_steps", 0.0)),
        "sim_steps": float(len(state.history)),
        "photon_final": float(last.get("photon_momentum", 0.0)),
    }


def compare_delivery_metrics(
    continuous: dict[str, float],
    pulsed: dict[str, float],
) -> dict[str, float]:
    """Δ(pulsed − continuous) for key observables."""
    keys = (
        "lattice_received",
        "cumulative_phase_slip",
        "eta_final",
        "mean_twist_final",
        "twist_load",
        "recovery_steps",
    )
    return {k: float(pulsed.get(k, 0.0)) - float(continuous.get(k, 0.0)) for k in keys}