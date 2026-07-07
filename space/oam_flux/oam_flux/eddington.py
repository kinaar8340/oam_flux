"""Mini-Eddington limit: flywheel cluster binding vs accumulated lattice momentum."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import TwistLattice
from .momentum import oam_kinetic_momentum
from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from .vqc_photonics import PhotonicsConfig


@dataclass
class FlywheelSite:
    index: tuple[int, int, int]
    momentum_received: float = 0.0
    outward_flux: float = 0.0
    local_twist: float = 0.0
    binding: float = 0.0
    unstable: bool = False


@dataclass
class EddingtonResult:
    kappa: float
    ell: int
    n_sites: int
    sites: list[FlywheelSite] = field(default_factory=list)
    history: list[dict[str, float]] = field(default_factory=list)
    limit_exceeded: bool = False
    total_outward_flux: float = 0.0

    def to_dict(self) -> dict:
        unstable = sum(1 for s in self.sites if s.unstable)
        return {
            "kappa": self.kappa,
            "ell": float(self.ell),
            "n_sites": float(self.n_sites),
            "limit_exceeded": self.limit_exceeded,
            "unstable_sites": float(unstable),
            "total_outward_flux": self.total_outward_flux,
            "max_binding": max((s.binding for s in self.sites), default=0.0),
            "max_momentum": max((s.momentum_received for s in self.sites), default=0.0),
        }


def binding_at_site(local_twist: float, kappa: float, *, theta_crit: float = 5.8) -> float:
    """
    Flywheel binding capacity ∝ κ × local twist (gauge-restoring torque scale).

    Eddington analog: outward flux when deposited momentum exceeds binding.
    """
    return float(kappa) * max(abs(local_twist), 0.01) * (theta_crit / 5.8)


def run_eddington_probe(
    *,
    kappa: float = 0.85,
    ell: int = 3,
    lambda_nm: float = 1550.0,
    n_flywheels: int = 6,
    n_steps: int = 80,
    kick_strength: float = 0.08,
    l_max: int = 6,
    nx: int = 20,
) -> EddingtonResult:
    """Pump VQC flux into flywheel cluster; detect Eddington-style outward flux."""
    lattice = TwistLattice(nx=nx, kappa=kappa)
    photonics = PhotonicsConfig(l_max=l_max, n_z=n_steps, nr=256, lambda_nm=lambda_nm)
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=ell,
        coupling_cfg={
            "kick_strength": kick_strength,
            "flywheel_sites": n_flywheels,
            "conserve_momentum": True,
        },
    )

    indices = lattice.flywheel_indices(n_flywheels)
    sites = [FlywheelSite(index=idx) for idx in indices]
    result = EddingtonResult(kappa=kappa, ell=ell, n_sites=n_flywheels, sites=sites)

    p0 = oam_kinetic_momentum(energy_scale=1.0, ell=ell, lambda_nm=lambda_nm)
    steps = min(n_steps, state.propagation.n_z)

    for step in range(steps):
        z_idx = min(state.z_index, state.propagation.n_z - 1)
        from .flux_deposit import deposit_on_flywheels

        kick, deposited = deposit_on_flywheels(
            lattice,
            state.propagation,
            ell=ell,
            z_index=z_idx,
            kick_strength=kick_strength,
            flywheel_sites=n_flywheels,
        )

        delta_p = min(deposited, state.photon_reservoir)
        state.photon_reservoir = max(0.0, state.photon_reservoir - delta_p)
        lattice.apply_kick(kick, photon_momentum=delta_p)
        lattice.relax_step()

        step_outward = 0.0
        for site in sites:
            i, j, k = site.index
            site.local_twist = float(lattice.theta[i, j, k])
            site.binding = binding_at_site(site.local_twist, kappa)
            # Attribute deposited momentum share per site from local kick magnitude
            local_kick = abs(float(kick[i, j, k]))
            kick_sum = max(float(np.abs(kick).sum()), 1e-12)
            site.momentum_received += delta_p * (local_kick / kick_sum)
            excess = max(0.0, site.momentum_received - site.binding)
            new_outward = excess - site.outward_flux
            if new_outward > 0:
                site.outward_flux = excess
                site.unstable = True
                step_outward += new_outward
                # Shed twist back to lattice (outward radiation)
                lattice.theta[i, j, k] = max(
                    0.01,
                    site.local_twist - 0.15 * new_outward / max(site.binding, 0.01),
                )

        result.total_outward_flux += step_outward
        result.history.append(
            {
                "step": float(step),
                "photon_p": state.photon_reservoir,
                "lattice_received": -lattice.momentum_ledger,
                "outward_flux": step_outward,
                "cumulative_outward": result.total_outward_flux,
                "unstable_count": float(sum(1 for s in sites if s.unstable)),
            }
        )

        if state.z_index < state.propagation.n_z - 1:
            state.z_index += 1

    result.limit_exceeded = any(s.unstable for s in sites)
    return result