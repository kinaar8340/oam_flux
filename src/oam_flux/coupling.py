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
    history: list[dict[str, float]] = field(default_factory=list)

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
        self.history.append(
            {
                "step": float(step),
                "mean_twist": self.lattice.mean_twist,
                "twist_variance": self.lattice.twist_variance,
                "photon_z": self.photon.z,
                "photon_momentum": self.photon.momentum,
                "momentum_ledger": self.lattice.momentum_ledger,
            }
        )


def run_coupling_step(state: CouplingState, step: int, *, dz: float = 0.025) -> None:
    """One coupled timestep: photon propagates, kicks flywheels, lattice relaxes."""
    kick = state.photon_kick()
    p_before = state.photon.momentum

    if state.conserve_momentum:
        state.lattice.apply_kick(kick, photon_momentum=p_before)
        # photon transfers momentum to lattice; packet depletes
        state.photon.energy_scale = max(0.0, state.photon.energy_scale - p_before * state.kick_strength)
    else:
        state.lattice.theta = np.clip(state.lattice.theta + kick, 0.01, 2 * np.pi - 0.01)

    state.photon.propagate(dz)
    state.lattice.relax_step()
    state.record(step)