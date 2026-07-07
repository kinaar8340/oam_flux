"""Mini-Eddington limit: flywheel cluster binding vs accumulated lattice momentum."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import TwistLattice
from .momentum import effective_kick_strength, energy_scale_from_ev, oam_kinetic_momentum
from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from .vqc_photonics import PhotonicsConfig


@dataclass
class FlywheelSite:
    index: tuple[int, int, int]
    momentum_received: float = 0.0
    outward_flux: float = 0.0
    outward_flux_vec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_twist: float = 0.0
    binding: float = 0.0
    unstable: bool = False


@dataclass
class EddingtonResult:
    kappa: float
    ell: int
    n_sites: int
    energy_scale: float = 1.0
    kick_strength: float = 0.08
    effective_kick: float = 0.08
    sites: list[FlywheelSite] = field(default_factory=list)
    history: list[dict[str, float]] = field(default_factory=list)
    limit_exceeded: bool = False
    total_outward_flux: float = 0.0
    cumulative_phase_slip: float = 0.0
    wind_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_dict(self) -> dict:
        unstable = sum(1 for s in self.sites if s.unstable)
        w = self.wind_vector
        wind_mag = float(np.linalg.norm(w))
        return {
            "kappa": self.kappa,
            "ell": float(self.ell),
            "n_sites": float(self.n_sites),
            "limit_exceeded": self.limit_exceeded,
            "unstable_sites": float(unstable),
            "total_outward_flux": self.total_outward_flux,
            "energy_scale": self.energy_scale,
            "effective_kick": self.effective_kick,
            "wind_x": float(w[0]),
            "wind_y": float(w[1]),
            "wind_z": float(w[2]),
            "wind_magnitude": wind_mag,
            "max_binding": max((s.binding for s in self.sites), default=0.0),
            "max_momentum": max((s.momentum_received for s in self.sites), default=0.0),
        }


def _hopf_wind_torque(
    lattice: TwistLattice,
    index: tuple[int, int, int],
    tangent: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Spread excess momentum as twist perturbation along the local Hopf fiber."""
    i, j, k = index
    nx = lattice.nx
    torque = np.zeros_like(lattice.theta)
    di, dj, dk = tangent
    for hop in range(1, 4):
        ii = int(np.clip(i + hop * di * 2.0, 0, nx - 1))
        jj = int(np.clip(j + hop * dj * 2.0, 0, nx - 1))
        kk = int(np.clip(k + hop * dk * 2.0, 0, nx - 1))
        torque[ii, jj, kk] -= 0.04 * strength / hop
    torque[i, j, k] += 0.08 * strength * float(dk)
    return torque


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
    energy_scale: float = 1.0,
    energy_ev: float | None = None,
    l_max: int = 6,
    nx: int = 20,
) -> EddingtonResult:
    """Pump VQC flux into flywheel cluster; detect Eddington-style outward flux."""
    from .helix_viz import hopf_tangent_at_lattice_site

    if energy_ev is not None:
        energy_scale = energy_scale_from_ev(energy_ev=float(energy_ev), lambda_nm=lambda_nm)
    k_eff = effective_kick_strength(kick_strength, energy_scale)

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
            "energy_scale": energy_scale,
        },
    )

    indices = lattice.flywheel_indices(n_flywheels)
    sites = [FlywheelSite(index=idx) for idx in indices]
    result = EddingtonResult(
        kappa=kappa,
        ell=ell,
        n_sites=n_flywheels,
        energy_scale=energy_scale,
        kick_strength=kick_strength,
        effective_kick=k_eff,
        sites=sites,
    )

    steps = min(n_steps, state.propagation.n_z)

    from .back_reaction import apply_phase_slip, lattice_back_reaction

    for step in range(steps):
        z_idx = min(state.z_index, state.propagation.n_z - 1)
        from .flux_deposit import deposit_on_flywheels

        br = lattice_back_reaction(lattice, ell=ell)
        k_step = k_eff * br["coupling_factor"]
        kick, deposited = deposit_on_flywheels(
            lattice,
            state.propagation,
            ell=ell,
            z_index=z_idx,
            kick_strength=k_step,
            flywheel_sites=n_flywheels,
        )

        delta_p = min(deposited, state.photon_reservoir)
        state.photon_reservoir = max(0.0, state.photon_reservoir - delta_p)
        lattice.apply_kick(kick, photon_momentum=delta_p)
        state.photon_reservoir, slip = apply_phase_slip(
            state.photon_reservoir,
            phase_slip_fraction=br["phase_slip_fraction"],
        )
        if slip > 0:
            lattice.momentum_ledger -= slip
            result.cumulative_phase_slip += slip
        lattice.relax_step()

        step_outward = 0.0
        step_wind = np.zeros(3)
        wind_torque = np.zeros_like(lattice.theta)
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
                tangent = hopf_tangent_at_lattice_site(site.index, nx=nx)
                site.outward_flux = excess
                site.outward_flux_vec += tangent * new_outward
                step_wind += tangent * new_outward
                site.unstable = True
                step_outward += new_outward
                wind_torque += _hopf_wind_torque(lattice, site.index, tangent, new_outward)
                # Shed twist locally; directional wind spreads along Hopf fiber
                lattice.theta[i, j, k] = max(
                    0.01,
                    site.local_twist - 0.15 * new_outward / max(site.binding, 0.01),
                )

        if float(np.abs(wind_torque).max()) > 0:
            lattice.theta = np.clip(lattice.theta + 0.12 * wind_torque, 0.01, 2 * np.pi - 0.01)

        result.total_outward_flux += step_outward
        result.wind_vector += step_wind
        w = result.wind_vector
        result.history.append(
            {
                "step": float(step),
                "photon_p": state.photon_reservoir,
                "lattice_received": -lattice.momentum_ledger,
                "outward_flux": step_outward,
                "cumulative_outward": result.total_outward_flux,
                "wind_x": float(w[0]),
                "wind_y": float(w[1]),
                "wind_z": float(w[2]),
                "unstable_count": float(sum(1 for s in sites if s.unstable)),
                "back_reaction_coupling": br["coupling_factor"],
                "back_reaction_slip": slip,
            }
        )

        if state.z_index < state.propagation.n_z - 1:
            state.z_index += 1

    result.limit_exceeded = any(s.unstable for s in sites)
    return result