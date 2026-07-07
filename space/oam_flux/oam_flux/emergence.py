"""Mystery-style emergence probes under VQC OAM–flux coupling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import PHI, RESIDUAL_R, LatticeConstants, load_config
from .lattice import TwistLattice
from .vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from .vqc_photonics import PhotonicsConfig


E = math.e
PI = math.pi
E_INV2 = math.exp(-2.0)
GOLDEN_ANGLE_DEG = 360.0 * (1.0 - 1.0 / PHI)
GOLDEN_FRACTION = GOLDEN_ANGLE_DEG / 1000.0


@dataclass(frozen=True)
class EmergenceAnalogs:
    """Mystery-compatible dynamical analog targets."""

    residual_r: float = RESIDUAL_R
    e_inv2: float = E_INV2
    golden_fraction: float = GOLDEN_FRACTION
    kappa_doc: float = 0.85
    kappa_sim: float = 0.89

    @property
    def kappa_star(self) -> float:
        return E / PI - RESIDUAL_R / PI**2

    def as_dict(self) -> dict[str, float]:
        return {
            "R": self.residual_r,
            "e_inv2": self.e_inv2,
            "golden_angle": self.golden_fraction,
            "kappa_star": self.kappa_star,
            "kappa_doc": self.kappa_doc,
            "kappa_sim": self.kappa_sim,
        }


@dataclass
class TrialResult:
    kappa: float
    ell: int
    n_steps: int
    lambda_t: float
    initial_mean_twist: float
    final_mean_twist: float
    initial_variance: float
    final_variance: float
    mean_survival: float
    fluctuation_survival: float
    bound_b_kappa: float
    holonomy_gap: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kappa": self.kappa,
            "ell": float(self.ell),
            "n_steps": float(self.n_steps),
            "lambda_t": self.lambda_t,
            "initial_mean_twist": self.initial_mean_twist,
            "final_mean_twist": self.final_mean_twist,
            "mean_survival": self.mean_survival,
            "fluctuation_survival": self.fluctuation_survival,
            "bound_B_kappa": self.bound_b_kappa,
            "holonomy_gap": self.holonomy_gap,
        }


def bound_b(kappa: float) -> float:
    """Skyrme holonomy-gap scaling B(κ) = π²(e/π − κ)."""
    return PI**2 * (E / PI - kappa)


def lambda_t_steps(kappa: float, dt: float, lambda_t: float = 2.0) -> int:
    """Steps for λt = lambda_t given mean-field rate λ ≈ κ."""
    return max(1, int(round(lambda_t / (kappa * dt))))


def delta_pct(measured: float, target: float) -> float:
    if abs(target) < 1e-15:
        return float("inf") if abs(measured) > 1e-15 else 0.0
    return 100.0 * abs(measured - target) / abs(target)


def best_analog_match(
    measured: float,
    analogs: EmergenceAnalogs | None = None,
    *,
    keys: tuple[str, ...] = ("R", "e_inv2", "golden_angle"),
) -> dict[str, float | str]:
    """Find closest mystery analog to a measured scalar."""
    a = analogs or EmergenceAnalogs()
    targets = a.as_dict()
    best_name = keys[0]
    best_val = targets[best_name]
    best_delta = delta_pct(measured, best_val)
    for name in keys:
        val = targets[name]
        d = delta_pct(measured, val)
        if d < best_delta:
            best_name, best_val, best_delta = name, val, d
    return {
        "best_analog": best_name,
        "best_match": best_val,
        "measured": measured,
        "delta_pct": best_delta,
    }


def golden_quantized_ells(l_max: int) -> list[int]:
    """
    ℓ values whose azimuthal phase increment best aligns with golden-angle packing.

    Phase ladder: Δφ_ℓ = |ℓ| · (2π / (2·l_max + 1)).
    Score by distance to multiples of golden_angle_rad.
    """
    golden_rad = math.radians(GOLDEN_ANGLE_DEG)
    n_modes = 2 * l_max + 1
    scored: list[tuple[float, int]] = []
    for ell in range(-l_max, l_max + 1):
        if ell == 0:
            continue
        phase_inc = abs(ell) * (2 * PI / n_modes)
        # nearest multiple of golden angle
        k = round(phase_inc / golden_rad)
        k = max(k, 1)
        dist = abs(phase_inc - k * golden_rad)
        scored.append((dist, ell))
    scored.sort()
    # top quarter of modes by golden alignment
    n_pick = max(1, len(scored) // 4)
    return sorted({ell for _, ell in scored[:n_pick]})


def run_vqc_emergence_trial(
    *,
    kappa: float,
    ell: int,
    lambda_t: float = 2.0,
    dt: float = 0.001,
    nx: int = 20,
    photonics: PhotonicsConfig | None = None,
    coupling_cfg: dict | None = None,
    pump_fraction: float = 0.5,
) -> TrialResult:
    """
    Two-phase trial aligned with mystery λt=2 convention:

    1. VQC pump phase — deposit OAM flux on flywheels
    2. Pure relaxation — PDE only, no photon injection (survival measured here)
    """
    ph = photonics or PhotonicsConfig(l_max=6, n_z=100, nr=256)
    cpl = coupling_cfg or {"kick_strength": 0.06, "flywheel_sites": 4, "conserve_momentum": True}

    lattice = TwistLattice(nx=nx, dt=dt, kappa=kappa)
    initial_mean = lattice.mean_twist
    initial_var = lattice.twist_variance

    state = VQCCouplingState.from_config(lattice, ph, ell=ell, coupling_cfg=cpl)
    total_steps = lambda_t_steps(kappa, dt, lambda_t)
    pump_steps = min(
        max(1, int(round(total_steps * pump_fraction))),
        state.propagation.n_z,
    )
    relax_steps = max(0, total_steps - pump_steps)

    for step in range(pump_steps):
        run_vqc_coupling_step(state, step)

    post_pump_mean = state.lattice.mean_twist
    post_pump_var = state.lattice.twist_variance

    for _ in range(relax_steps):
        state.lattice.relax_step()

    final_mean = state.lattice.mean_twist
    final_var = state.lattice.twist_variance

    # Survival = retained fraction after relaxation (mystery convention)
    ref_mean = post_pump_mean if post_pump_mean > 1e-12 else initial_mean
    ref_var = post_pump_var if post_pump_var > 1e-12 else initial_var
    mean_survival = final_mean / ref_mean if ref_mean > 1e-12 else final_mean
    fluctuation_survival = final_var / ref_var if ref_var > 1e-12 else final_var

    return TrialResult(
        kappa=kappa,
        ell=ell,
        n_steps=total_steps,
        lambda_t=lambda_t,
        initial_mean_twist=initial_mean,
        final_mean_twist=final_mean,
        initial_variance=initial_var,
        final_variance=final_var,
        mean_survival=mean_survival,
        fluctuation_survival=fluctuation_survival,
        bound_b_kappa=bound_b(kappa),
        holonomy_gap=E / PI - kappa,
    )


@dataclass
class KappaSweepResult:
    rows: list[TrialResult] = field(default_factory=list)
    analogs: EmergenceAnalogs = field(default_factory=EmergenceAnalogs)

    def best_kappa_for_analog(self, metric: str = "mean_survival", analog: str = "R") -> dict:
        targets = self.analogs.as_dict()
        target = targets[analog]
        best = min(self.rows, key=lambda r: abs(getattr(r, metric) - target))
        return {
            "kappa": best.kappa,
            "metric": metric,
            "measured": getattr(best, metric),
            "analog": analog,
            "target": target,
            "delta_pct": delta_pct(getattr(best, metric), target),
        }


def kappa_sweep(
    *,
    kappa_min: float = 0.80,
    kappa_max: float = 0.90,
    n_points: int = 21,
    ell: int = 3,
    lambda_t: float = 2.0,
    pump_fraction: float = 0.5,
    photonics: PhotonicsConfig | None = None,
    coupling_cfg: dict | None = None,
) -> KappaSweepResult:
    """Sweep κ under VQC pumping at fixed λt=2."""
    kappas = np.linspace(kappa_min, kappa_max, n_points)
    result = KappaSweepResult()
    for kappa in kappas:
        result.rows.append(
            run_vqc_emergence_trial(
                kappa=float(kappa),
                ell=ell,
                lambda_t=lambda_t,
                pump_fraction=pump_fraction,
                photonics=photonics,
                coupling_cfg=coupling_cfg,
            )
        )
    return result


@dataclass
class EllSweepResult:
    rows: list[TrialResult] = field(default_factory=list)
    golden_ells: list[int] = field(default_factory=list)
    analogs: EmergenceAnalogs = field(default_factory=EmergenceAnalogs)

    def best_ell_for_analog(self, metric: str = "mean_survival", analog: str = "R") -> dict:
        targets = self.analogs.as_dict()
        target = targets[analog]
        best = min(self.rows, key=lambda r: abs(getattr(r, metric) - target))
        return {
            "ell": best.ell,
            "metric": metric,
            "measured": getattr(best, metric),
            "analog": analog,
            "target": target,
            "delta_pct": delta_pct(getattr(best, metric), target),
            "is_golden_quantized": best.ell in self.golden_ells,
        }


def ell_sweep(
    *,
    l_max: int = 6,
    kappa: float = 0.85,
    lambda_t: float = 2.0,
    pump_fraction: float = 0.5,
    photonics: PhotonicsConfig | None = None,
    coupling_cfg: dict | None = None,
) -> EllSweepResult:
    """Sweep active ℓ under VQC pumping; mark golden-quantized modes."""
    ph = photonics or PhotonicsConfig(l_max=l_max, n_z=100, nr=256)
    golden = golden_quantized_ells(l_max)
    result = EllSweepResult(golden_ells=golden)
    for ell in range(-l_max, l_max + 1):
        result.rows.append(
            run_vqc_emergence_trial(
                kappa=kappa,
                ell=ell,
                lambda_t=lambda_t,
                pump_fraction=pump_fraction,
                photonics=ph,
                coupling_cfg=coupling_cfg,
            )
        )
    return result


def emergence_report(
    *,
    kappa_result: KappaSweepResult,
    ell_result: EllSweepResult,
) -> dict:
    """Compile mystery-style analog comparison report."""
    analogs = EmergenceAnalogs()
    # Use κ_doc trial from ell sweep as primary measured point
    kappa_doc_trial = next((r for r in ell_result.rows if r.ell == 3), ell_result.rows[0])

    comparisons = []
    for label, measured in [
        ("mean_survival@κ_doc", kappa_doc_trial.mean_survival),
        ("fluctuation_survival@κ_doc", kappa_doc_trial.fluctuation_survival),
        ("bound_B@κ_doc", kappa_doc_trial.bound_b_kappa),
        ("holonomy_gap@κ_doc", kappa_doc_trial.holonomy_gap),
    ]:
        keys = ("R", "e_inv2", "golden_angle") if "survival" in label else ("R", "kappa_star", "kappa_doc")
        match = best_analog_match(measured, analogs, keys=keys)
        comparisons.append({"label": label, **match})

    return {
        "analogs": analogs.as_dict(),
        "comparisons": comparisons,
        "kappa_sweep_best_R": kappa_result.best_kappa_for_analog("mean_survival", "R"),
        "kappa_sweep_best_e_inv2": kappa_result.best_kappa_for_analog("mean_survival", "e_inv2"),
        "ell_sweep_best_R": ell_result.best_ell_for_analog("mean_survival", "R"),
        "ell_sweep_best_e_inv2": ell_result.best_ell_for_analog("mean_survival", "e_inv2"),
        "golden_quantized_ells": ell_result.golden_ells,
    }


def emergence_config_from_yaml(cfg: dict | None = None) -> dict:
    """Load emergence section merged with lattice/vqc/coupling defaults."""
    base = cfg or load_config()
    return {
        "lambda_t": float(base.get("emergence", {}).get("lambda_t", 2.0)),
        "kappa_min": float(base.get("emergence", {}).get("kappa_min", 0.80)),
        "kappa_max": float(base.get("emergence", {}).get("kappa_max", 0.90)),
        "kappa_sweep_points": int(base.get("emergence", {}).get("kappa_sweep_points", 21)),
        "ell_sweep_l_max": int(base.get("emergence", {}).get("ell_sweep_l_max", 6)),
        "probe_ell": int(base.get("photon", {}).get("ell", 3)),
        "probe_kappa": float(base.get("lattice", {}).get("kappa", 0.85)),
        "photonics": PhotonicsConfig(
            l_max=int(base.get("emergence", {}).get("ell_sweep_l_max", 6)),
            n_z=int(base.get("emergence", {}).get("n_z", 100)),
            nr=int(base.get("emergence", {}).get("nr", 256)),
            turbulence=float(base.get("vqc", {}).get("turbulence", 0.0)),
            chirp=float(base.get("vqc", {}).get("chirp", 0.0)),
        ),
        "coupling_cfg": base.get("coupling", {}),
        "pump_fraction": float(base.get("emergence", {}).get("pump_fraction", 0.5)),
    }