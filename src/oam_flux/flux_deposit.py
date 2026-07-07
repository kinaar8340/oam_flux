"""Deposit VQC OAM flux onto gauged Hopf lattice fibers."""

from __future__ import annotations

import numpy as np

from .lattice import TwistLattice
from .vqc_photonics import PropagationResult


def hopf_fiber_coords(nx: int, *, rho_scale: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map 3-torus voxel centers to Hopf-style fiber coordinates.

    Returns (rho, phi, eta):
      - rho: transverse distance from beam axis (for LG radial lookup)
      - phi: azimuthal angle (OAM phase winding)
      - eta: fiber parameter along propagation direction
    """
    coords = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    rho = np.sqrt((x - np.pi) ** 2 + (y - np.pi) ** 2)
    rho_max = float(rho.max()) if rho.max() > 0 else 1.0
    scale = rho_scale if rho_scale is not None else rho_max
    rho = rho / rho_max * scale
    phi = np.arctan2(y - np.pi, x - np.pi)
    # Hopf fiber combination: phase mixes all three torus angles
    eta = np.mod(x + y + z, 2 * np.pi)
    return rho, phi, eta


def interpolate_radial(rho_grid: np.ndarray, radial: np.ndarray, rho_voxels: np.ndarray) -> np.ndarray:
    """Interpolate LG radial profile onto lattice rho coordinates."""
    return np.interp(rho_voxels.ravel(), rho_grid, radial, left=0.0, right=0.0).reshape(rho_voxels.shape)


def build_flux_kick(
    lattice: TwistLattice,
    propagation: PropagationResult,
    *,
    ell: int,
    z_index: int,
    kick_strength: float,
    helical_weight: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Build twist kick from VQC intensity at z_index deposited on lattice fibers.

    Returns (kick field, deposited momentum).
    """
    rho_vox, phi, eta = hopf_fiber_coords(lattice.nx, rho_scale=float(propagation.rho.max()))
    radial = propagation.radial_profile(ell)
    radial_on_lattice = interpolate_radial(propagation.rho, radial, rho_vox)

    intensity_z = propagation.mode_intensity(ell, z_index)
    # Helical OAM winding: ℓ·φ along fiber η
    helical = np.cos(ell * phi + 0.5 * eta)

    kick = kick_strength * intensity_z * radial_on_lattice * helical * helical_weight
    deposited_momentum = float(intensity_z * kick_strength * np.abs(kick).sum())
    return kick, deposited_momentum


def deposit_on_flywheels(
    lattice: TwistLattice,
    propagation: PropagationResult,
    *,
    ell: int,
    z_index: int,
    kick_strength: float,
    flywheel_sites: int,
) -> tuple[np.ndarray, float]:
    """Restrict VQC flux deposition to flywheel resonator neighborhoods."""
    kick, momentum = build_flux_kick(
        lattice,
        propagation,
        ell=ell,
        z_index=z_index,
        kick_strength=kick_strength,
    )
    mask = np.zeros_like(lattice.theta)
    for idx in lattice.flywheel_indices(flywheel_sites):
        local = np.zeros_like(mask)
        local[idx] = 1.0
        for ax in range(3):
            local = 0.25 * (
                np.roll(local, 1, ax) + np.roll(local, -1, ax) + 2 * local
            )
        mask += local
    mask /= max(mask.max(), 1e-12)
    return kick * mask, momentum