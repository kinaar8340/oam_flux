"""Multi-pulse photon train: pump → saturate → recover → repeat."""

from __future__ import annotations

from dataclasses import dataclass

from .lattice import DEFAULT_RECOVERY_TAU
from .pulse_envelope import DEFAULT_PULSE_SHAPE, build_pump_envelope, normalize_pulse_shape
from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step


@dataclass
class PulseTrainConfig:
    n_pulses: int = 3
    pump_steps: int = 30
    gap_steps: int = 20
    recovery_memory: float = 0.0
    recovery_tau: float = DEFAULT_RECOVERY_TAU
    pulse_shape: str = DEFAULT_PULSE_SHAPE

    def __post_init__(self) -> None:
        self.pulse_shape = normalize_pulse_shape(self.pulse_shape)

    @property
    def total_steps(self) -> int:
        return int(self.n_pulses) * (int(self.pump_steps) + int(self.gap_steps))


def run_vqc_pulse_train(state: VQCCouplingState, cfg: PulseTrainConfig) -> int:
    """
    Run a pulse train on an initialized VQCCouplingState.

    Each pulse refills the photon reservoir, pumps for ``pump_steps``, then
    forces recovery for ``gap_steps`` before the next pulse.
    """
    step = 0
    state.pulses_fired = 0
    state.total_injected = 0.0
    state.pulse_train_mode = True
    state.recovery_memory = float(cfg.recovery_memory)
    state.recovery_tau = float(cfg.recovery_tau)
    state.lattice.recovery_tau = float(cfg.recovery_tau)
    state.pulse_shape = cfg.pulse_shape
    envelope = build_pump_envelope(cfg.pulse_shape, int(cfg.pump_steps))

    for pulse in range(int(cfg.n_pulses)):
        state.refill_reservoir()
        state.current_pulse = pulse
        state.pulse_phase = "pump"
        for i in range(int(cfg.pump_steps)):
            state.pump_envelope_factor = envelope[i] if i < len(envelope) else 0.0
            run_vqc_coupling_step(state, step)
            step += 1

        state.photon_reservoir = 0.0
        state.pump_envelope_factor = 0.0
        state.pulse_phase = "recovery"
        for _ in range(int(cfg.gap_steps)):
            run_vqc_coupling_step(state, step)
            step += 1

    state.pulse_phase = "done"
    state.pump_envelope_factor = 0.0
    return step