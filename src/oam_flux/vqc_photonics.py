"""VQC vectorized OAM propagation — adapted from vqc_sims_public/src/photonics.py."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import factorial, genlaguerre


@dataclass
class PhotonicsConfig:
    l_max: int = 8
    w0: float = 1.0
    nr: int = 512
    z_start: float = 0.0
    z_end: float = 5.0
    n_z: int = 200
    turbulence: float = 0.0
    chirp: float = 0.0
    qec_suppression: int = 1  # exponent; 1 = no QEC attenuation in coupling demos


@dataclass
class PropagationResult:
    z_steps: np.ndarray
    ells: np.ndarray
    intensity: np.ndarray  # shape (n_z, n_modes)
    rho: np.ndarray
    radial_weights: np.ndarray  # shape (n_modes, nr)
    config: PhotonicsConfig = field(repr=False)

    @property
    def n_z(self) -> int:
        return int(self.z_steps.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.ells.shape[0])

    def ell_index(self, ell: int) -> int:
        matches = np.where(self.ells == ell)[0]
        if matches.size == 0:
            raise ValueError(f"ℓ={ell} not in propagated ladder {self.ells.tolist()}")
        return int(matches[0])

    def mode_intensity(self, ell: int, z_index: int) -> float:
        return float(self.intensity[z_index, self.ell_index(ell)])

    def radial_profile(self, ell: int) -> np.ndarray:
        return self.radial_weights[self.ell_index(ell)]


def kolmogorov_radial_phase_profile(nr: int = 2048, r0: float = 0.15) -> np.ndarray:
    """Kolmogorov phase screen (VQC photonics convention)."""
    r = np.linspace(0, 10, nr)
    phase_var = (r / r0) ** (5 / 3)
    phase = np.cumsum(np.random.normal(0, np.sqrt(np.gradient(phase_var)), nr))
    return phase - phase[0]


def lg_radial_weights(
    *,
    nr: int = 512,
    w0: float = 1.0,
    l_max: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """LG p=0 radial weights for ℓ ∈ [-l_max, l_max] (SciPy path)."""
    rho_max = float(max(8.0, 3.0 * np.sqrt(2 * l_max + 1)))
    rho = np.linspace(0, rho_max, nr)
    dr = rho[1] - rho[0]
    weights = np.zeros((2 * l_max + 1, nr), dtype=np.float64)

    for idx, ell in enumerate(range(-l_max, l_max + 1)):
        L = abs(ell)
        norm = np.sqrt(2 / (np.pi * factorial(L))) / w0
        rw = rho / w0
        x = 2 * rw**2
        lag_poly = genlaguerre(0, L)(x)
        radial = norm * (np.sqrt(2) ** L) * (rw**L) * np.exp(-rw**2) * lag_poly
        radial = np.nan_to_num(radial, nan=0.0, posinf=0.0, neginf=0.0)
        integral = np.sum(radial**2 * rho * dr)
        norm_factor = np.sqrt(integral) if integral > 1e-100 else 1.0
        weights[idx] = radial / norm_factor

    return weights, rho


def propagate_multi_ell_vectorized(config: PhotonicsConfig | None = None) -> PropagationResult:
    """
    Vectorized multi-ℓ propagation (VQC photonics core).

    Returns intensity cube (z, ℓ) plus radial weights for lattice flux deposition.
    """
    cfg = config or PhotonicsConfig()
    z_steps = np.linspace(cfg.z_start, cfg.z_end, cfg.n_z)
    weights, rho = lg_radial_weights(nr=cfg.nr, w0=cfg.w0, l_max=cfg.l_max)
    dr = rho[1] - rho[0]

    phase_z = np.exp(1j * cfg.chirp * z_steps**2)[:, None, None]

    if cfg.turbulence > 0:
        screen = kolmogorov_radial_phase_profile(nr=len(rho))
        screen = np.interp(rho, np.linspace(0, 10, len(screen)), screen, left=0, right=0)
        turb_phase = np.exp(1j * cfg.turbulence * screen)[None, None, :]
    else:
        turb_phase = np.ones((1, 1, len(rho)), dtype=complex)

    field = weights[None, :, :] * (phase_z * turb_phase).conj()
    intensity = np.sum(np.abs(field) ** 2 * rho[None, None, :] * dr, axis=-1)
    intensity = np.clip(intensity, 0.0, 1.0)
    if cfg.qec_suppression != 1:
        intensity **= cfg.qec_suppression

    ells = np.arange(-cfg.l_max, cfg.l_max + 1)
    return PropagationResult(
        z_steps=z_steps,
        ells=ells,
        intensity=intensity,
        rho=rho,
        radial_weights=weights,
        config=cfg,
    )