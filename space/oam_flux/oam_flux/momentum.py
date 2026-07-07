"""Momentum and energy accounting: photon kinetic flux ↔ lattice ledger."""

from __future__ import annotations

import math

# SI constants; momentum display maps kg·m/s → ×10⁻²⁷ kg·m/s for optical λ.
H_PLANCK_J_S = 6.62607015e-34
HBAR_J_S = H_PLANCK_J_S / (2.0 * math.pi)
C_M_S = 299792458.0
HC_EV_NM = 1239.841984332893  # hc in eV·nm (E[eV] = HC_EV_NM / λ[nm])
MOMENTUM_DISPLAY_SCALE = 1e27
DEFAULT_CONSERVATION_TOLERANCE = 0.005  # 0.5 % of p₀
LAMBDA_NM_MIN = 400.0
LAMBDA_NM_MAX = 2000.0


def oam_kinetic_momentum(
    *,
    energy_scale: float,
    ell: int,
    lambda_nm: float,
    intensity: float = 1.0,
) -> float:
    """
    OAM photon momentum: p = (h/λ) · |ℓ| · I · E_scale.

    Equivalent to p = (2π ℏ |ℓ|)/λ for azimuthal mode index |ℓ| on the
    propagation wavevector.  λ is in nm; the return value is in display
    units of ×10⁻²⁷ kg·m/s (SI momentum × ``MOMENTUM_DISPLAY_SCALE``).
    """
    lambda_m = max(float(lambda_nm), 1.0) * 1e-9
    p_si = (
        H_PLANCK_J_S
        * abs(int(ell))
        * float(intensity)
        * float(energy_scale)
        / lambda_m
    )
    return p_si * MOMENTUM_DISPLAY_SCALE


def photon_energy_ev(*, lambda_nm: float, energy_scale: float = 1.0) -> float:
    """Photon energy E = hc/λ in eV (scaled by ``energy_scale``)."""
    lam = max(float(lambda_nm), 1.0)
    return HC_EV_NM / lam * float(energy_scale)


def photon_frequency_thz(*, lambda_nm: float) -> float:
    """Optical frequency f = c/λ in THz."""
    lambda_m = max(float(lambda_nm), 1.0) * 1e-9
    return C_M_S / lambda_m / 1e12


def energy_scale_from_ev(*, energy_ev: float, lambda_nm: float) -> float:
    """Packet energy scale relative to single-photon E = hc/λ at ``lambda_nm``."""
    ref = photon_energy_ev(lambda_nm=lambda_nm, energy_scale=1.0)
    return max(float(energy_ev), 1e-12) / ref


def lambda_nm_from_ev(*, energy_ev: float, energy_scale: float = 1.0) -> float:
    """Invert E = hc/λ for wavelength (nm) at fixed ``energy_scale``."""
    e = max(float(energy_ev), 1e-12) / max(float(energy_scale), 1e-12)
    return HC_EV_NM / e


def clip_lambda_nm(lambda_nm: float) -> float:
    return float(max(LAMBDA_NM_MIN, min(LAMBDA_NM_MAX, float(lambda_nm))))


def momentum_natural_units(
    *,
    energy_scale: float,
    ell: int,
    lambda_nm: float = 1550.0,
) -> float:
    """Dimensionless p / (ℏ k) = |ℓ| · energy_scale for this normalization."""
    _ = lambda_nm  # retained for API symmetry with oam_kinetic_momentum
    return abs(int(ell)) * float(energy_scale)


def effective_kick_strength(kick_strength: float, energy_scale: float) -> float:
    """User kick knob scaled by packet energy: κ_eff = κ · energy_scale."""
    return float(kick_strength) * max(float(energy_scale), 0.0)


def photon_state(
    *,
    ell: int,
    lambda_nm: float,
    energy_ev: float,
) -> dict[str, float]:
    """Derive coupled photon scalars from (ℓ, λ, E)."""
    scale = energy_scale_from_ev(energy_ev=energy_ev, lambda_nm=lambda_nm)
    p = oam_kinetic_momentum(energy_scale=scale, ell=ell, lambda_nm=lambda_nm)
    return {
        "ell": float(int(ell)),
        "lambda_nm": float(lambda_nm),
        "energy_ev": float(energy_ev),
        "energy_scale": scale,
        "momentum": p,
        "momentum_natural": momentum_natural_units(
            energy_scale=scale, ell=ell, lambda_nm=lambda_nm
        ),
        "frequency_thz": photon_frequency_thz(lambda_nm=lambda_nm),
    }


def is_momentum_conserved(
    residual: float,
    initial_total: float,
    *,
    tolerance_fraction: float = DEFAULT_CONSERVATION_TOLERANCE,
) -> bool:
    """True when |residual| is within ``tolerance_fraction`` of |p₀|."""
    ref = max(abs(initial_total), 1e-12)
    return abs(residual) <= tolerance_fraction * ref


def conservation_check(
    *,
    photon_p: float,
    ledger: float,
    initial_total: float,
    tolerance_fraction: float = DEFAULT_CONSERVATION_TOLERANCE,
) -> dict[str, float]:
    """
    Ledger convention: lattice_received = -ledger (ledger decreases when lattice gains).

    Closed system: photon_p + lattice_received ≈ initial_total.
    """
    lattice_received = -ledger
    total_now = photon_p + lattice_received
    residual = total_now - initial_total
    ref = max(abs(initial_total), 1e-12)
    return {
        "photon_p": photon_p,
        "lattice_received": lattice_received,
        "total_now": total_now,
        "initial_total": initial_total,
        "conservation_residual": residual,
        "conserved_pct": 100.0 * (1.0 - abs(residual) / ref),
        "conserved": is_momentum_conserved(
            residual, initial_total, tolerance_fraction=tolerance_fraction
        ),
    }