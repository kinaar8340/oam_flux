"""Multi-pulse photon train: pump → saturate → recover → repeat."""

from __future__ import annotations

from dataclasses import dataclass

from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step


@dataclass
class PulseTrainConfig:
    n_pulses: int = 3
    pump_steps: int = 30
    gap_steps: int = 20

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

    for pulse in range(int(cfg.n_pulses)):
        state.refill_reservoir()
        state.current_pulse = pulse
        state.pulse_phase = "pump"
        for _ in range(int(cfg.pump_steps)):
            run_vqc_coupling_step(state, step)
            step += 1

        state.photon_reservoir = 0.0
        state.pulse_phase = "recovery"
        for _ in range(int(cfg.gap_steps)):
            run_vqc_coupling_step(state, step)
            step += 1

    state.pulse_phase = "done"
    return step