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
    energy_scale: float = 1.0
    flywheel_sites: int = 4
    conserve_momentum: bool = True
    z_index: int = 0
    lambda_nm: float = 1550.0
    photon_reservoir: float = 0.0
    initial_total_momentum: float = 0.0
    cumulative_phase_slip: float = 0.0
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
        from .momentum import oam_kinetic_momentum

        lam = float(getattr(photonics_cfg, "lambda_nm", 1550.0))
        e_scale = float(coupling_cfg.get("energy_scale", 1.0))
        p0 = oam_kinetic_momentum(energy_scale=e_scale, ell=ell, lambda_nm=lam)
        return cls(
            lattice=lattice,
            propagation=propagation,
            ell=ell,
            kick_strength=float(coupling_cfg.get("kick_strength", 0.08)),
            energy_scale=e_scale,
            flywheel_sites=int(coupling_cfg.get("flywheel_sites", 4)),
            conserve_momentum=bool(coupling_cfg.get("conserve_momentum", True)),
            lambda_nm=lam,
            photon_reservoir=p0,
            initial_total_momentum=p0,
        )

    def record(self, step: int) -> None:
        from .momentum import conservation_check

        z_idx = min(self.z_index, self.propagation.n_z - 1)
        check = conservation_check(
            photon_p=self.photon_reservoir,
            ledger=self.lattice.momentum_ledger,
            initial_total=self.initial_total_momentum,
        )
        self.history.append(
            {
                "step": float(step),
                "z_index": float(z_idx),
                "z_km": float(self.propagation.z_steps[z_idx]),
                "mode_intensity": self.propagation.mode_intensity(self.ell, z_idx),
                "mean_twist": self.lattice.mean_twist,
                "twist_variance": self.lattice.twist_variance,
                "photon_momentum": check["photon_p"],
                "lattice_received": check["lattice_received"],
                "total_momentum": check["total_now"],
                "conservation_residual": check["conservation_residual"],
                "initial_total": self.initial_total_momentum,
                "momentum_ledger": self.lattice.momentum_ledger,
            }
        )


def run_vqc_coupling_step(state: VQCCouplingState, step: int) -> None:
    """Advance one step: deposit VQC flux at current z, relax lattice, step z."""
    from .back_reaction import apply_phase_slip, lattice_back_reaction
    from .momentum import effective_kick_strength

    z_idx = min(state.z_index, state.propagation.n_z - 1)
    br = lattice_back_reaction(state.lattice, ell=state.ell)
    k_eff = (
        effective_kick_strength(state.kick_strength, state.energy_scale)
        * br["coupling_factor"]
    )
    kick, deposited = deposit_on_flywheels(
        state.lattice,
        state.propagation,
        ell=state.ell,
        z_index=z_idx,
        kick_strength=k_eff,
        flywheel_sites=state.flywheel_sites,
    )

    slip = 0.0
    if state.conserve_momentum:
        delta_p = min(deposited, state.photon_reservoir)
        state.photon_reservoir = max(0.0, state.photon_reservoir - delta_p)
        state.lattice.apply_kick(kick, photon_momentum=delta_p)
        state.photon_reservoir, slip = apply_phase_slip(
            state.photon_reservoir,
            phase_slip_fraction=br["phase_slip_fraction"],
        )
        if slip > 0:
            state.lattice.momentum_ledger -= slip
            state.cumulative_phase_slip += slip
    else:
        import numpy as np
        state.lattice.theta = np.clip(state.lattice.theta + kick, 0.01, 2 * np.pi - 0.01)

    state.lattice.relax_step()
    state.record(step)
    state.history[-1].update(
        {
            "back_reaction_coupling": br["coupling_factor"],
            "back_reaction_slip": slip if state.conserve_momentum else 0.0,
            "back_reaction_ell_shift": br["effective_ell_shift"],
            "cumulative_phase_slip": state.cumulative_phase_slip,
        }
    )

    if state.z_index < state.propagation.n_z - 1:
        state.z_index += 1