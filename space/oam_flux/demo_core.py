"""HF Space demo core — wraps oam_flux simulations for Gradio."""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oam_flux.constants import RESIDUAL_R
from oam_flux.coupling import CouplingState, run_coupling_step
from oam_flux.emergence import (
    E_INV2,
    GOLDEN_FRACTION,
    EmergenceAnalogs,
    ell_sweep,
    emergence_report,
    kappa_sweep,
)
from oam_flux.lattice import TwistLattice
from oam_flux.photon import OAMPacket
from oam_flux.vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from oam_flux.vqc_photonics import PhotonicsConfig

GITHUB_URL = "https://github.com/kinaar8340/oam_flux"
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar111/oam_flux"
TOE_URL = "https://github.com/kinaar8340/toe"
VQC_URL = "https://github.com/kinaar8340/vqc_sims_public"
MYSTERY_URL = "https://github.com/kinaar8340/mystery"

ANALOGS = EmergenceAnalogs()


def is_hf_space() -> bool:
    return bool(
        __import__("os").environ.get("SPACE_ID")
        or __import__("os").environ.get("SYSTEM") == "spaces"
    )


def get_build_label() -> str:
    try:
        from build_info import BUILD_COMMIT, BUILD_UPDATED_UTC
        return f"build {BUILD_COMMIT} · {BUILD_UPDATED_UTC}"
    except ImportError:
        return "dev build"


def _fig_to_pil(fig: plt.Figure):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def run_vqc_coupling(
    ell: int,
    kappa: float,
    kick_strength: float,
    n_steps: int,
    l_max: int,
    turbulence: float,
) -> tuple:
    """VQC coupling demo → (timeseries_img, heatmap_img, kick_img, summary_md)."""
    lattice = TwistLattice(nx=20, kappa=kappa)
    photonics = PhotonicsConfig(
        l_max=l_max,
        n_z=min(n_steps, 150),
        nr=256,
        turbulence=turbulence,
    )
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=ell,
        coupling_cfg={
            "kick_strength": kick_strength,
            "flywheel_sites": 4,
            "conserve_momentum": True,
        },
    )
    steps = min(n_steps, state.propagation.n_z)
    for step in range(steps):
        run_vqc_coupling_step(state, step)

    # Timeseries
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    hist = state.history
    axes[0].plot([h["step"] for h in hist], [h["mean_twist"] for h in hist], color="green")
    axes[0].set_ylabel("⟨θ⟩")
    axes[0].set_title(f"VQC coupling ℓ={ell} κ={kappa:.3f}")
    axes[0].grid(alpha=0.3)
    axes[1].plot([h["step"] for h in hist], [h["momentum_ledger"] for h in hist], color="purple")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("momentum ledger")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    ts_img = _fig_to_pil(fig)

    # Heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.imshow(
        state.propagation.intensity.T,
        aspect="auto",
        origin="lower",
        extent=[
            state.propagation.z_steps[0],
            state.propagation.z_steps[-1],
            state.propagation.ells[0] - 0.5,
            state.propagation.ells[-1] + 0.5,
        ],
        cmap="magma",
    )
    ax2.set_xlabel("z")
    ax2.set_ylabel("ℓ")
    ax2.set_title("VQC multi-ℓ propagation")
    fig2.tight_layout()
    heat_img = _fig_to_pil(fig2)

    # Kick slice
    from oam_flux.flux_deposit import build_flux_kick
    mid = steps // 2
    kick, _ = build_flux_kick(lattice, state.propagation, ell=ell, z_index=mid, kick_strength=kick_strength)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.imshow(kick[kick.shape[0] // 2], origin="lower", cmap="RdBu_r")
    ax3.set_title(f"Flux kick slice z={mid}")
    fig3.tight_layout()
    kick_img = _fig_to_pil(fig3)

    md = (
        f"### VQC coupling summary\n"
        f"- **ℓ** = {ell} · **κ** = {kappa:.4f}\n"
        f"- Final ⟨θ⟩ = **{state.lattice.mean_twist:.4f}**\n"
        f"- Twist σ² = **{state.lattice.twist_variance:.4f}**\n"
        f"- Momentum ledger = **{state.lattice.momentum_ledger:.4f}**\n"
        f"- Residual R = **{RESIDUAL_R:.6f}** (mystery)\n"
    )
    return ts_img, heat_img, kick_img, md


def run_emergence(
    ell: int,
    kappa: float,
    kappa_points: int,
    l_max: int,
    lambda_t: float,
) -> tuple:
    """Emergence probes → (kappa_plot, ell_plot, report_md)."""
    photonics = PhotonicsConfig(l_max=l_max, n_z=80, nr=192)
    cpl = {"kick_strength": 0.06, "flywheel_sites": 4, "conserve_momentum": True}

    kap = kappa_sweep(
        kappa_min=0.80,
        kappa_max=0.90,
        n_points=int(kappa_points),
        ell=ell,
        lambda_t=lambda_t,
        photonics=photonics,
        coupling_cfg=cpl,
    )
    ell_res = ell_sweep(
        l_max=l_max,
        kappa=kappa,
        lambda_t=lambda_t,
        photonics=photonics,
        coupling_cfg=cpl,
    )
    report = emergence_report(kappa_result=kap, ell_result=ell_res)

    # κ plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    kappas = [r.kappa for r in kap.rows]
    surv = [r.mean_survival for r in kap.rows]
    axes[0].plot(kappas, surv, "o-", color="#2a6f97")
    axes[0].axhline(RESIDUAL_R, ls="--", color="#c9a227", label=f"R={RESIDUAL_R:.4f}")
    axes[0].axhline(E_INV2, ls="--", color="#e76f51", label=f"e⁻²={E_INV2:.4f}")
    axes[0].axhline(GOLDEN_FRACTION, ls="--", color="#6a4c93", label=f"golden={GOLDEN_FRACTION:.4f}")
    axes[0].axvline(ANALOGS.kappa_doc, ls=":", color="#e63946", label="κ_doc")
    axes[0].axvline(ANALOGS.kappa_star, ls="-.", color="#2a9d8f", label="κ*")
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("κ")
    axes[0].set_ylabel("mean_survival @ λt=2")
    axes[0].grid(alpha=0.3)
    axes[1].plot(kappas, [r.bound_b_kappa for r in kap.rows], color="#264653")
    axes[1].axhline(RESIDUAL_R, ls="--", color="#c9a227")
    axes[1].axvline(ANALOGS.kappa_star, ls="-.", color="#2a9d8f")
    axes[1].set_xlabel("κ")
    axes[1].set_ylabel("B(κ)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    kappa_img = _fig_to_pil(fig)

    # ℓ plot
    ells = [r.ell for r in ell_res.rows]
    esurv = [r.mean_survival for r in ell_res.rows]
    golden = set(ell_res.golden_ells)
    colors = ["#e9c46a" if e in golden else "#2a6f97" for e in ells]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(ells, esurv, color=colors, edgecolor="#333", linewidth=0.4)
    ax2.axhline(RESIDUAL_R, ls="--", color="#c9a227", label=f"R={RESIDUAL_R:.4f}")
    ax2.axhline(GOLDEN_FRACTION, ls="--", color="#6a4c93", label="golden")
    ax2.set_xlabel("ℓ")
    ax2.set_ylabel("mean_survival")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    ell_img = _fig_to_pil(fig2)

    lines = [
        "### Mystery emergence report",
        "",
        "| Metric | Measured | Best analog | Δ% |",
        "|--------|----------|-------------|-----|",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"| {row['label']} | {row['measured']:.4f} | {row['best_analog']} ({row['best_match']:.4f}) | {row['delta_pct']:.1f}% |"
        )
    lines += [
        "",
        f"**Best κ for R:** κ={report['kappa_sweep_best_R']['kappa']:.3f} "
        f"(Δ={report['kappa_sweep_best_R']['delta_pct']:.1f}%)",
        f"**Best ℓ for R:** ℓ={report['ell_sweep_best_R']['ell']} "
        f"(Δ={report['ell_sweep_best_R']['delta_pct']:.1f}%)",
        f"**Golden-quantized ℓ:** {report['golden_quantized_ells']}",
        "",
        f"κ* = {ANALOGS.kappa_star:.6f} · κ_doc = {ANALOGS.kappa_doc} · κ_sim = {ANALOGS.kappa_sim}",
    ]
    return kappa_img, ell_img, "\n".join(lines)


def run_analytic_coupling(ell: int, kappa: float, n_steps: int) -> tuple:
    """v0.1 analytic packet demo."""
    lattice = TwistLattice(nx=20, kappa=kappa)
    photon = OAMPacket(ell=ell, energy_scale=1.0)
    state = CouplingState(lattice=lattice, photon=photon, kick_strength=0.08)
    for step in range(n_steps):
        run_coupling_step(state, step)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot([h["step"] for h in state.history], [h["mean_twist"] for h in state.history], color="green")
    axes[0].set_ylabel("⟨θ⟩")
    axes[1].plot([h["step"] for h in state.history], [h["photon_momentum"] for h in state.history], label="photon")
    axes[1].plot([h["step"] for h in state.history], [h["momentum_ledger"] for h in state.history], label="ledger")
    axes[1].legend()
    axes[1].set_xlabel("step")
    fig.tight_layout()
    img = _fig_to_pil(fig)
    md = f"Final ⟨θ⟩ = **{state.lattice.mean_twist:.4f}** · ledger = **{state.lattice.momentum_ledger:.4f}**"
    return img, md