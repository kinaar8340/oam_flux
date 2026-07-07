"""Momentum accounting: photon kinetic flux ↔ lattice ledger."""

from __future__ import annotations


def oam_kinetic_momentum(
    *,
    energy_scale: float,
    ell: int,
    lambda_nm: float,
    intensity: float = 1.0,
) -> float:
    """
    Normalized photon momentum p ∝ E/c with OAM flux factor |ℓ|/λ.

    Uses λ in nm for display-friendly magnitudes (1550 nm → p ~ ℓ·E/1550).
    """
    lam = max(float(lambda_nm), 1.0)
    return float(energy_scale) * float(intensity) * abs(int(ell)) / lam


def conservation_check(
    *,
    photon_p: float,
    ledger: float,
    initial_total: float,
) -> dict[str, float]:
    """
    Ledger convention: lattice_received = -ledger (ledger decreases when lattice gains).

    Closed system: photon_p + lattice_received ≈ initial_total.
    """
    lattice_received = -ledger
    total_now = photon_p + lattice_received
    residual = total_now - initial_total
    return {
        "photon_p": photon_p,
        "lattice_received": lattice_received,
        "total_now": total_now,
        "initial_total": initial_total,
        "conservation_residual": residual,
        "conserved_pct": 100.0 * (1.0 - abs(residual) / max(abs(initial_total), 1e-12)),
    }