"""Momentum accounting: photon kinetic flux ↔ lattice ledger."""

from __future__ import annotations

# SI Planck constant (J·s); display scale maps kg·m/s → ×10⁻²⁷ kg·m/s for optical λ.
H_PLANCK_J_S = 6.62607015e-34
MOMENTUM_DISPLAY_SCALE = 1e27
DEFAULT_CONSERVATION_TOLERANCE = 0.005  # 0.5 % of p₀


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