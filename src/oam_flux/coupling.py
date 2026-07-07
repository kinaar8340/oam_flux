"""OAM photon ↔ flux-flywheel coupling with momentum ledger."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import TwistLattice
from .photon import OAMPacket


@dataclass
class CouplingState:
    lattice: TwistLattice
    photon: OAMPacket
    kick_strength: float = 0.08
    flywheel_sites: int = 4
    conserve_momentum: bool = True
    initial_total_momentum: float = 0.0
    cumulative_phase_slip: float = 0.0
    recovery_steps: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_total_momentum == 0.0:
            self.initial_total_momentum = self.photon.momentum

    def encounter_mask(self) -> np.ndarray:
        """Gaussian envelope centered on flywheel sites."""
        mask = np.zeros_like(self.lattice.theta)
        for idx in self.lattice.flywheel_indices(self.flywheel_sites):
            local = np.zeros_like(mask)
            local[idx] = 1.0
            # diffuse kick over 3³ neighborhood
            for ax in range(3):
                local = 0.25 * (
                    np.roll(local, 1, ax) + np.roll(local, -1, ax) + 2 * local
                )
            mask += local
        return mask / max(mask.max(), 1e-12)

    def photon_kick(self) -> np.ndarray:
        """Map OAM flux to discrete twist impulse on resonator sites."""
        mask = self.encounter_mask()
        flux = self.photon.momentum * self.photon.helical_phase_gradient
        return self.kick_strength * flux * mask

    def record(self, step: int) -> None:
        from .momentum import conservation_check

        p0 = self.initial_total_momentum
        check = conservation_check(
            photon_p=self.photon.momentum,
            ledger=self.lattice.momentum_ledger,
            initial_total=p0,
        )
        self.history.append(
            {
                "step": float(step),
                "initial_total": p0,
                "mean_twist": self.lattice.mean_twist,
                "twist_variance": self.lattice.twist_variance,
                "photon_z": self.photon.z,
                "photon_momentum": check["photon_p"],
                "lattice_received": check["lattice_received"],
                "total_momentum": check["total_now"],
                "conservation_residual": check["conservation_residual"],
                "momentum_ledger": self.lattice.momentum_ledger,
            }
        )


def run_coupling_step(state: CouplingState, step: int, *, dz: float = 0.025) -> None:
    """One coupled timestep: photon propagates, kicks flywheels, lattice relaxes."""
    from .back_reaction import apply_phase_slip, lattice_back_reaction, photon_pump_active

    pump_active = photon_pump_active(
        state.photon.momentum, state.initial_total_momentum,
    )
    slip = 0.0

    if pump_active:
        br_dep = lattice_back_reaction(state.lattice, ell=state.photon.ell)
        k_coupled = state.kick_strength * br_dep["coupling_factor"]
        kick = state.photon_kick() * br_dep["coupling_factor"]
        p_before = state.photon.momentum
        if state.conserve_momentum:
            delta_p = state.photon.transfer_momentum(p_before * k_coupled)
            state.lattice.apply_kick(kick, photon_momentum=delta_p)
            _, slip = apply_phase_slip(
                state.photon.momentum, phase_slip_fraction=br_dep["phase_slip_fraction"],
            )
            if slip > 0:
                actual_slip = state.photon.transfer_momentum(slip)
                state.lattice.momentum_ledger -= actual_slip
                state.cumulative_phase_slip += actual_slip
        else:
            state.lattice.theta = np.clip(state.lattice.theta + kick, 0.01, 2 * np.pi - 0.01)
    else:
        state.recovery_steps += 1

    state.photon.propagate(dz)
    state.lattice.relax_step(pump_active=pump_active)
    br = lattice_back_reaction(state.lattice, ell=state.photon.ell)
    state.record(step)
    state.history[-1].update(
        {
            "pump_active": 1.0 if pump_active else 0.0,
            "recovery_active": 0.0 if pump_active else 1.0,
            "twist_load_initial": state.lattice.twist_load_vs_initial(),
            "recovery_steps": float(state.recovery_steps),
            "back_reaction_coupling": br["coupling_factor"],
            "back_reaction_slip": slip if pump_active else 0.0,
            "back_reaction_ell_shift": br["effective_ell_shift"],
            "cumulative_phase_slip": state.cumulative_phase_slip,
        }
    )