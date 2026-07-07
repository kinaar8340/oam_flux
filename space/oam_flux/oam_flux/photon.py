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
    _momentum_p: float | None = None

    @property
    def momentum(self) -> float:
        """p = (h/λ) · |ℓ| · E_scale — OAM kinetic momentum (×10⁻²⁷ kg·m/s)."""
        if self._momentum_p is not None:
            return self._momentum_p
        from .momentum import oam_kinetic_momentum
        return oam_kinetic_momentum(
            energy_scale=self.energy_scale,
            ell=self.ell,
            lambda_nm=self.lambda_nm,
        )

    @property
    def energy_ev(self) -> float:
        """E = hc/λ in eV (scaled by packet ``energy_scale``)."""
        from .momentum import photon_energy_ev
        return photon_energy_ev(lambda_nm=self.lambda_nm, energy_scale=self.energy_scale)

    def transfer_momentum(self, requested: float) -> float:
        """Transfer up to `requested` momentum to lattice; return actual Δp."""
        actual = min(max(requested, 0.0), self.momentum)
        current = self.momentum
        if current > 1e-15:
            # Scale energy with momentum fraction for consistency
            self.energy_scale *= (current - actual) / current
        self._momentum_p = current - actual
        return actual

    @property
    def wavelength_um(self) -> float:
        return self.lambda_nm * 1e-3

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