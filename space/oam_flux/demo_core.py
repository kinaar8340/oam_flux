"""HF Space demo core — wraps oam_flux simulations for Gradio."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from oam_flux.momentum import oam_kinetic_momentum
from oam_flux.photon import OAMPacket
from oam_flux.vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from oam_flux.vqc_photonics import PhotonicsConfig

GITHUB_URL = "https://github.com/kinaar8340/oam_flux"
TOE_URL = "https://github.com/kinaar8340/toe"
VQC_URL = "https://github.com/kinaar8340/vqc_sims_public"
MYSTERY_URL = "https://github.com/kinaar8340/mystery"

ANALOGS = EmergenceAnalogs()


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


def _conservation_badge(history: list[dict]) -> str:
    if not history:
        return "—"
    last = history[-1]
    res = abs(last.get("conservation_residual", 0.0))
    total = last.get("initial_total", 1.0)
    pct = 100.0 * (1.0 - res / max(abs(total), 1e-12))
    ok = "✅" if res < 0.02 * max(abs(total), 1e-9) else "⚠️"
    return f"{ok} **Momentum conserved** — residual `{res:.5f}` ({pct:.1f}% of initial p₀)"


def _plot_momentum_history(history: list[dict], *, title: str, p0_label: str = "p₀") -> plt.Figure:
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})

    axes[0].plot(steps, [h["mean_twist"] for h in history], color="#2a9d8f", lw=1.5)
    axes[0].set_ylabel("⟨θ⟩")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [h["photon_momentum"] for h in history], color="#457b9d", lw=1.8, label="p_photon ∝ ℓ/λ")
    ax.plot(steps, [h.get("lattice_received", -h["momentum_ledger"]) for h in history],
            color="#e76f51", lw=1.8, label="p_lattice (received)")
    if history:
        p0 = history[0].get("initial_total", history[0]["photon_momentum"])
        ax.axhline(p0, color="#c9a227", ls=":", lw=1.2, label=f"{p0_label}={p0:.4f}")
        ax.plot(steps, [h.get("total_momentum", 0) for h in history], color="#6a4c93",
                ls="--", lw=1.0, alpha=0.8, label="p_total")
    ax.set_xlabel("step")
    ax.set_ylabel("momentum (norm.)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run_vqc_coupling(
    ell: int,
    kappa: float,
    kick_strength: float,
    n_steps: int,
    l_max: int,
    turbulence: float,
    lambda_nm: float,
) -> tuple:
    """VQC coupling demo → (timeseries_img, heatmap_img, kick_img, summary_md)."""
    p0 = oam_kinetic_momentum(energy_scale=1.0, ell=ell, lambda_nm=lambda_nm)
    lattice = TwistLattice(nx=20, kappa=kappa)
    photonics = PhotonicsConfig(
        l_max=l_max,
        n_z=min(n_steps, 150),
        nr=256,
        turbulence=turbulence,
        lambda_nm=lambda_nm,
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

    ts_img = _fig_to_pil(_plot_momentum_history(
        state.history,
        title=f"VQC coupling  ℓ={ell}  κ={kappa:.3f}  λ={lambda_nm:.0f} nm  p₀={p0:.5f}",
    ))

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
    ax2.axhline(ell, color="cyan", ls="--", lw=1.0, alpha=0.9)
    ax2.set_xlabel("z")
    ax2.set_ylabel("ℓ")
    ax2.set_title(f"Multi-ℓ propagation (active ℓ={ell})")
    fig2.tight_layout()
    heat_img = _fig_to_pil(fig2)

    from oam_flux.flux_deposit import build_flux_kick
    mid = steps // 2
    kick, _ = build_flux_kick(lattice, state.propagation, ell=ell, z_index=mid, kick_strength=kick_strength)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.imshow(kick[kick.shape[0] // 2], origin="lower", cmap="RdBu_r")
    ax3.set_title(f"Flux kick slice z={mid}")
    fig3.tight_layout()
    kick_img = _fig_to_pil(fig3)

    last = state.history[-1] if state.history else {}
    md = (
        f"### VQC coupling — ℓ={ell} · κ={kappa:.4f} · λ={lambda_nm:.0f} nm\n"
        f"- **p₀** = {p0:.6f}  (p ∝ |ℓ|/λ)\n"
        f"- Final ⟨θ⟩ = **{state.lattice.mean_twist:.4f}**\n"
        f"- p_photon final = **{last.get('photon_momentum', 0):.4f}**\n"
        f"- p_lattice received = **{last.get('lattice_received', 0):.4f}**\n"
        f"- Residual R = **{RESIDUAL_R:.6f}** (mystery)\n\n"
        f"{_conservation_badge(state.history)}\n"
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
        kappa_min=0.80, kappa_max=0.90, n_points=int(kappa_points),
        ell=ell, lambda_t=lambda_t, photonics=photonics, coupling_cfg=cpl,
    )
    ell_res = ell_sweep(
        l_max=l_max, kappa=kappa, lambda_t=lambda_t, photonics=photonics, coupling_cfg=cpl,
    )
    report = emergence_report(kappa_result=kap, ell_result=ell_res)

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
        f"### Mystery emergence report (probe ℓ={ell}, κ={kappa:.3f})",
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
        f"**Best κ for R:** κ={report['kappa_sweep_best_R']['kappa']:.3f}",
        f"**Best ℓ for R:** ℓ={report['ell_sweep_best_R']['ell']}",
        f"**Golden-quantized ℓ:** {report['golden_quantized_ells']}",
    ]
    return kappa_img, ell_img, "\n".join(lines)


def run_analytic_coupling(ell: int, kappa: float, n_steps: int, lambda_nm: float) -> tuple:
    """v0.1 analytic packet demo with explicit p ∝ ℓ/λ."""
    photon = OAMPacket(ell=ell, lambda_nm=lambda_nm, energy_scale=1.0)
    p0 = photon.momentum
    lattice = TwistLattice(nx=20, kappa=kappa)
    state = CouplingState(lattice=lattice, photon=photon, kick_strength=0.08)
    for step in range(int(n_steps)):
        run_coupling_step(state, step)

    img = _fig_to_pil(_plot_momentum_history(
        state.history,
        title=f"Analytic coupling  ℓ={ell}  κ={kappa:.3f}  λ={lambda_nm:.0f} nm  p₀={p0:.5f}",
    ))
    last = state.history[-1] if state.history else {}
    md = (
        f"### Analytic momentum ledger\n"
        f"- **ℓ** = {ell} · **λ** = {lambda_nm:.0f} nm · **κ** = {kappa:.4f}\n"
        f"- **p₀** = {p0:.6f}  (|ℓ|/λ)\n"
        f"- Final ⟨θ⟩ = **{state.lattice.mean_twist:.4f}**\n"
        f"- Δp_lattice = **{last.get('lattice_received', 0):.4f}**\n\n"
        f"{_conservation_badge(state.history)}\n"
    )
    return img, md