#!/usr/bin/env python3
"""v0.3: Mystery emergence probes under VQC OAM–flux coupling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oam_flux.constants import load_config
from oam_flux.emergence import (
    E_INV2,
    GOLDEN_FRACTION,
    RESIDUAL_R,
    EmergenceAnalogs,
    ell_sweep,
    emergence_config_from_yaml,
    emergence_report,
    kappa_sweep,
)


def _plot_kappa_sweep(result, analogs: EmergenceAnalogs, path: Path) -> None:
    kappas = [r.kappa for r in result.rows]
    survival = [r.mean_survival for r in result.rows]
    bound_vals = [r.bound_b_kappa for r in result.rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(kappas, survival, "o-", color="#2a6f97", lw=2, ms=4, label="mean_survival (VQC)")
    ax.axhline(RESIDUAL_R, color="#c9a227", ls="--", label=f"R = {RESIDUAL_R:.4f}")
    ax.axhline(E_INV2, color="#e76f51", ls="--", label=f"e⁻² = {E_INV2:.4f}")
    ax.axhline(GOLDEN_FRACTION, color="#6a4c93", ls="--", label=f"golden = {GOLDEN_FRACTION:.4f}")
    ax.axvline(analogs.kappa_doc, color="#e63946", ls=":", label=f"κ_doc = {analogs.kappa_doc}")
    ax.axvline(analogs.kappa_sim, color="#457b9d", ls=":", label=f"κ_sim ≈ {analogs.kappa_sim}")
    ax.axvline(analogs.kappa_star, color="#2a9d8f", ls="-.", label=f"κ* = {analogs.kappa_star:.4f}")
    ax.set_xlabel("κ")
    ax.set_ylabel("mean_survival @ λt=2")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    ax.set_title("κ sweep — post-pump relaxation survival vs mystery analogs")

    ax2 = axes[1]
    ax2.plot(kappas, bound_vals, color="#264653", lw=2)
    ax2.axhline(RESIDUAL_R, color="#c9a227", ls="--", label=f"R = {RESIDUAL_R:.4f}")
    ax2.axvline(analogs.kappa_doc, color="#e63946", ls=":")
    ax2.axvline(analogs.kappa_star, color="#2a9d8f", ls="-.", label="κ* (B null)")
    ax2.set_xlabel("κ")
    ax2.set_ylabel("B(κ) = π²(e/π − κ)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("Holonomy-gap scaling under κ sweep")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ell_sweep(result, analogs: EmergenceAnalogs, path: Path) -> None:
    ells = [r.ell for r in result.rows]
    survival = [r.mean_survival for r in result.rows]
    golden_set = set(result.golden_ells)

    colors = ["#e9c46a" if e in golden_set else "#2a6f97" for e in ells]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(ells, survival, color=colors, edgecolor="#333", linewidth=0.5)
    ax.axhline(RESIDUAL_R, color="#c9a227", ls="--", label=f"R = {RESIDUAL_R:.4f}")
    ax.axhline(E_INV2, color="#e76f51", ls="--", label=f"e⁻² = {E_INV2:.4f}")
    ax.axhline(GOLDEN_FRACTION, color="#6a4c93", ls="--", label=f"golden = {GOLDEN_FRACTION:.4f}")
    ax.set_xlabel("ℓ (OAM quantum number)")
    ax.set_ylabel("mean_survival @ λt=2")
    ax.set_title("ℓ sweep — post-pump survival; gold = golden-angle quantized modes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_table(report: dict) -> str:
    lines = [
        f"{'Metric':<30} {'Measured':>10} {'Best analog':>14} {'Δ%':>8}",
        "-" * 66,
    ]
    for row in report["comparisons"]:
        lines.append(
            f"{row['label']:<30} {row['measured']:>10.6f} "
            f"{row['best_match']:>14.6f} {row['delta_pct']:>7.2f}%"
        )
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    ecfg = emergence_config_from_yaml(cfg)
    analogs = EmergenceAnalogs()

    print("OAM–flux emergence probes (v0.3) | λt =", ecfg["lambda_t"])
    print(f"Analogs: R={RESIDUAL_R:.6f}  e⁻²={E_INV2:.6f}  golden={GOLDEN_FRACTION:.6f}")
    print(f"κ*: {analogs.kappa_star:.6f}  κ_doc={analogs.kappa_doc}  κ_sim={analogs.kappa_sim}\n")

    kappa_result = kappa_sweep(
        kappa_min=ecfg["kappa_min"],
        kappa_max=ecfg["kappa_max"],
        n_points=ecfg["kappa_sweep_points"],
        ell=ecfg["probe_ell"],
        lambda_t=ecfg["lambda_t"],
        pump_fraction=ecfg["pump_fraction"],
        photonics=ecfg["photonics"],
        coupling_cfg=ecfg["coupling_cfg"],
    )

    ell_result = ell_sweep(
        l_max=ecfg["ell_sweep_l_max"],
        kappa=ecfg["probe_kappa"],
        lambda_t=ecfg["lambda_t"],
        pump_fraction=ecfg["pump_fraction"],
        photonics=ecfg["photonics"],
        coupling_cfg=ecfg["coupling_cfg"],
    )

    report = emergence_report(kappa_result=kappa_result, ell_result=ell_result)

    out_dir = ROOT / "outputs" / "emergence_probes"
    out_dir.mkdir(parents=True, exist_ok=True)

    kappa_plot = out_dir / "kappa_survival_sweep.png"
    ell_plot = out_dir / "ell_survival_sweep.png"
    _plot_kappa_sweep(kappa_result, analogs, kappa_plot)
    _plot_ell_sweep(ell_result, analogs, ell_plot)

    json_path = out_dir / "report.json"
    serializable = {
        **report,
        "kappa_sweep": [r.to_dict() for r in kappa_result.rows],
        "ell_sweep": [r.to_dict() for r in ell_result.rows],
    }
    json_path.write_text(json.dumps(serializable, indent=2))

    print(_format_table(report))
    print()
    print("Best κ for R:", report["kappa_sweep_best_R"])
    print("Best ℓ for R:", report["ell_sweep_best_R"])
    print(f"Golden-quantized ℓ: {report['golden_quantized_ells']}")
    print(f"\nSaved → {kappa_plot}")
    print(f"Saved → {ell_plot}")
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()