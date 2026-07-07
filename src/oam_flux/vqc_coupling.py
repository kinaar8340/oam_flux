"""VQC photonics ↔ lattice coupling with z-resolved flux deposition."""

from __future__ import annotations

from dataclasses import dataclass, field

from .flux_deposit import deposit_on_flywheels
from .lattice import TwistLattice
from .vqc_photonics import PhotonicsConfig, PropagationResult, propagate_multi_ell_vectorized


@dataclass
class VQCCouplingState:
    lattice: TwistLattice
    propagation: PropagationResult
    ell: int
    kick_strength: float = 0.08
    flywheel_sites: int = 4
    conserve_momentum: bool = True
    z_index: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        lattice: TwistLattice,
        photonics_cfg: PhotonicsConfig,
        *,
        ell: int,
        coupling_cfg: dict,
    ) -> VQCCouplingState:
        propagation = propagate_multi_ell_vectorized(photonics_cfg)
        return cls(
            lattice=lattice,
            propagation=propagation,
            ell=ell,
            kick_strength=float(coupling_cfg.get("kick_strength", 0.08)),
            flywheel_sites=int(coupling_cfg.get("flywheel_sites", 4)),
            conserve_momentum=bool(coupling_cfg.get("conserve_momentum", True)),
        )

    def record(self, step: int) -> None:
        z_idx = min(self.z_index, self.propagation.n_z - 1)
        self.history.append(
            {
                "step": float(step),
                "z_index": float(z_idx),
                "z_km": float(self.propagation.z_steps[z_idx]),
                "mode_intensity": self.propagation.mode_intensity(self.ell, z_idx),
                "mean_twist": self.lattice.mean_twist,
                "twist_variance": self.lattice.twist_variance,
                "momentum_ledger": self.lattice.momentum_ledger,
            }
        )


def run_vqc_coupling_step(state: VQCCouplingState, step: int) -> None:
    """Advance one step: deposit VQC flux at current z, relax lattice, step z."""
    z_idx = min(state.z_index, state.propagation.n_z - 1)
    kick, deposited = deposit_on_flywheels(
        state.lattice,
        state.propagation,
        ell=state.ell,
        z_index=z_idx,
        kick_strength=state.kick_strength,
        flywheel_sites=state.flywheel_sites,
    )

    if state.conserve_momentum:
        state.lattice.apply_kick(kick, photon_momentum=deposited)
    else:
        import numpy as np
        state.lattice.theta = np.clip(state.lattice.theta + kick, 0.01, 2 * np.pi - 0.01)

    state.lattice.relax_step()
    state.record(step)

    if state.z_index < state.propagation.n_z - 1:
        state.z_index += 1