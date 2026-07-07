"""Lattice → photon back-reaction from accumulated twist."""

from __future__ import annotations

import numpy as np

from .lattice import TwistLattice

DEFAULT_BACK_REACTION_STRENGTH = 0.35
MIN_COUPLING_FACTOR = 0.15


def lattice_back_reaction(
    lattice: TwistLattice,
    *,
    ell: int,
    strength: float = DEFAULT_BACK_REACTION_STRENGTH,
) -> dict[str, float]:
    """
    Back-reaction of gauged lattice twist on the propagating photon.

    High ⟨θ⟩ and twist variance reduce coupling efficiency (κ_eff multiplier)
    and bleed momentum via phase slip into the lattice ledger.
    """
    ref = max(float(lattice.theta_crit), 0.01)
    theta_norm = float(lattice.mean_twist) / ref
    var_norm = float(np.sqrt(lattice.twist_variance)) / ref
    load = 0.6 * min(theta_norm, 2.0) + 0.4 * min(var_norm, 2.0)
    coupling_factor = max(MIN_COUPLING_FACTOR, 1.0 - float(strength) * load)
    phase_slip_fraction = float(strength) * 0.12 * load
    ell_shift = (
        -float(np.sign(ell)) * float(strength) * 0.05 * load if int(ell) != 0 else 0.0
    )
    return {
        "coupling_factor": coupling_factor,
        "phase_slip_fraction": phase_slip_fraction,
        "effective_ell_shift": ell_shift,
        "twist_load": load,
    }


def apply_phase_slip(
    reservoir: float,
    *,
    phase_slip_fraction: float,
) -> tuple[float, float]:
    """Return (remaining reservoir, momentum slipped to lattice)."""
    slip = max(0.0, float(reservoir) * max(float(phase_slip_fraction), 0.0))
    return max(0.0, float(reservoir) - slip), slip