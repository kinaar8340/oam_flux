"""Helical OAM photon packets — bridge to vqc_sims_public photonics layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import genlaguerre


@dataclass
class OAMPacket:
    """Propagating ℓ-mode twist packet with kinetic flux bookkeeping."""

    ell: int = 3
    lambda_nm: float = 1550.0
    w0: float = 1.0
    energy_scale: float = 1.0
    z: float = 0.0

    @property
    def momentum(self) -> float:
        """Normalized p ∝ E/c; energy_scale is the sole knob in this reduced model."""
        return self.energy_scale

    @property
    def helical_phase_gradient(self) -> float:
        """Azimuthal phase gradient ∝ ℓ — kinetic flux carrier."""
        return float(self.ell)

    def lg_radial(self, rho: np.ndarray) -> np.ndarray:
        """p=0 Laguerre–Gaussian radial profile (matches vqc photonics.py convention)."""
        L = abs(self.ell)
        norm = np.sqrt(2 / (np.pi * math.factorial(L))) / self.w0
        rw = rho / self.w0
        x = 2 * rw**2
        lag = genlaguerre(0, L)(x)
        radial = norm * (np.sqrt(2) ** L) * (rw**L) * np.exp(-rw**2) * lag
        integral = np.sum(radial**2 * rho * (rho[1] - rho[0] if len(rho) > 1 else 1.0))
        norm_factor = np.sqrt(integral) if integral > 1e-12 else 1.0
        return radial / norm_factor

    def propagate(self, dz: float) -> None:
        self.z += dz

    def phase_at(self, phi: np.ndarray, z: float | None = None) -> np.ndarray:
        """Helical twist phase: ℓ·φ + kz surrogate."""
        z_val = self.z if z is None else z
        k = 2 * np.pi / (self.lambda_nm * 1e-9)
        return self.ell * phi + 0.001 * k * z_val

    def flux_density(self, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Intensity × phase-gradient proxy for Poynting-like kinetic flux."""
        radial = self.lg_radial(rho)
        phase = self.phase_at(phi)
        return (radial[:, None] ** 2) * np.abs(np.gradient(phase, axis=-1))