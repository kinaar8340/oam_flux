"""HF Space demo core — wraps oam_flux simulations for Gradio."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tempfile
from pathlib import Path

from oam_flux.constants import RESIDUAL_R
from oam_flux.eddington import run_eddington_probe
from oam_flux.helix_viz import (
    build_helix_geometry,
    figures_to_pil,
    frames_to_gif,
    frames_to_mp4,
    helix_animation_frames,
    render_helix_frame,
)
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
from oam_flux.momentum import (
    DEFAULT_CONSERVATION_TOLERANCE,
    clip_lambda_nm,
    effective_kick_strength,
    energy_scale_from_ev,
    is_momentum_conserved,
    lambda_nm_from_ev,
    oam_kinetic_momentum,
    photon_energy_ev,
    photon_state,
)
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


def format_photon_readout(ell: int, lambda_nm: float, energy_ev: float) -> str:
    """Live coupled ℓ, λ, E readout with p and natural units."""
    st = photon_state(ell=int(ell), lambda_nm=float(lambda_nm), energy_ev=float(energy_ev))
    return (
        f"**Active:** ℓ={int(ell)} · λ={st['lambda_nm']:.0f} nm · "
        f"E={st['energy_ev']:.4f} eV · f={st['frequency_thz']:.2f} THz · "
        f"**p₀={st['momentum']:.6f}** ×10⁻²⁷ kg·m/s · "
        f"p/(ℏk)={st['momentum_natural']:.3f} · R_ref=0.137486"
    )


def couple_from_lambda(lambda_nm: float, ell: int, energy_ev: float, lock: str) -> tuple:
    """λ change: update E when λ drives E."""
    lam = clip_lambda_nm(lambda_nm)
    e = float(energy_ev)
    if lock == "λ drives E":
        e = photon_energy_ev(lambda_nm=lam, energy_scale=1.0)
    return e, format_photon_readout(int(ell), lam, e)


def couple_from_energy(energy_ev: float, ell: int, lambda_nm: float, lock: str) -> tuple:
    """E change: update λ when E drives λ."""
    e = max(float(energy_ev), 0.62)
    lam = clip_lambda_nm(lambda_nm)
    if lock == "E drives λ":
        lam = clip_lambda_nm(lambda_nm_from_ev(energy_ev=e, energy_scale=1.0))
    return lam, format_photon_readout(int(ell), lam, e)


def _kick_scale_line(kick_strength: float, energy_scale: float) -> str:
    k_eff = effective_kick_strength(kick_strength, energy_scale)
    return (
        f"- **energy_scale** = {energy_scale:.4f} · "
        f"**κ_eff** = {k_eff:.4f} (κ × energy_scale)\n"
    )


def couple_from_ell(ell: int, lambda_nm: float, energy_ev: float) -> str:
    """ℓ change: E and λ fixed; p updates via readout."""
    return format_photon_readout(int(ell), clip_lambda_nm(lambda_nm), float(energy_ev))


def _conservation_badge(history: list[dict]) -> str:
    if not history:
        return "—"
    last = history[-1]
    residual = last.get("conservation_residual", 0.0)
    total = last.get("initial_total", 1.0)
    res = abs(residual)
    pct = 100.0 * (1.0 - res / max(abs(total), 1e-12))
    tol_pct = 100.0 * DEFAULT_CONSERVATION_TOLERANCE
    ok = "✅" if is_momentum_conserved(residual, total) else "⚠️"
    return (
        f"{ok} **Momentum conserved** — residual `{res:.5f}` ({pct:.1f}% of p₀; "
        f"tolerance ±{tol_pct:.1f}%)"
    )


def _plot_momentum_history(history: list[dict], *, title: str, p0_label: str = "p₀") -> plt.Figure:
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})

    axes[0].plot(steps, [h["mean_twist"] for h in history], color="#2a9d8f", lw=1.5, label="⟨θ⟩")
    if history and "back_reaction_coupling" in history[0]:
        ax_br = axes[0].twinx()
        ax_br.plot(
            steps,
            [h.get("back_reaction_coupling", 1.0) for h in history],
            color="#6a4c93",
            ls="--",
            lw=1.2,
            label="back-react η",
        )
        ax_br.set_ylabel("η (coupling)", color="#6a4c93")
        ax_br.set_ylim(0.0, 1.05)
        ax_br.tick_params(axis="y", labelcolor="#6a4c93")
    axes[0].set_ylabel("⟨θ⟩")
    axes[0].set_title(title)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [h["photon_momentum"] for h in history], color="#457b9d", lw=1.8, label="p_photon ∝ h|ℓ|/λ")
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
    energy_ev: float,
) -> tuple:
    """VQC coupling demo → (timeseries_img, heatmap_img, kick_img, summary_md)."""
    st = photon_state(ell=int(ell), lambda_nm=float(lambda_nm), energy_ev=float(energy_ev))
    e_scale = st["energy_scale"]
    p0 = st["momentum"]
    e0 = st["energy_ev"]
    f0 = st["frequency_thz"]
    p_nat = st["momentum_natural"]
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
            "energy_scale": e_scale,
        },
    )
    steps = min(n_steps, state.propagation.n_z)
    for step in range(steps):
        run_vqc_coupling_step(state, step)

    ts_img = _fig_to_pil(_plot_momentum_history(
        state.history,
        title=(
            f"VQC coupling  ℓ={ell}  κ={kappa:.3f}  λ={lambda_nm:.0f} nm  "
            f"E={e0:.3f} eV  p₀={p0:.5f}"
        ),
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
    k_eff = effective_kick_strength(kick_strength, e_scale)
    kick, _ = build_flux_kick(
        lattice, state.propagation, ell=ell, z_index=mid, kick_strength=k_eff,
    )
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.imshow(kick[kick.shape[0] // 2], origin="lower", cmap="RdBu_r")
    ax3.set_title(f"Flux kick slice z={mid}  (κ_eff={k_eff:.3f})")
    fig3.tight_layout()
    kick_img = _fig_to_pil(fig3)

    last = state.history[-1] if state.history else {}
    md = (
        f"### VQC coupling — ℓ={ell} · κ={kappa:.4f} · λ={lambda_nm:.0f} nm\n"
        f"- **E₀** = {e0:.4f} eV  (E = hc/λ) · **f** = {f0:.2f} THz\n"
        f"- **p₀** = {p0:.6f} ×10⁻²⁷ kg·m/s  (p = h|ℓ|/λ) · **p/(ℏk)** = {p_nat:.3f}\n"
        f"{_kick_scale_line(kick_strength, e_scale)}"
        f"- **Back-reaction** η_final = **{last.get('back_reaction_coupling', 1.0):.3f}** · "
        f"phase slip = **{last.get('cumulative_phase_slip', 0.0):.4f}**\n"
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


def run_analytic_coupling(
    ell: int,
    kappa: float,
    n_steps: int,
    lambda_nm: float,
    energy_ev: float,
) -> tuple:
    """v0.1 analytic packet demo with explicit p ∝ ℓ/λ and E = hc/λ."""
    e_scale = energy_scale_from_ev(energy_ev=float(energy_ev), lambda_nm=float(lambda_nm))
    photon = OAMPacket(ell=ell, lambda_nm=lambda_nm, energy_scale=e_scale)
    p0 = photon.momentum
    e0 = photon.energy_ev
    p_nat = photon_state(ell=ell, lambda_nm=lambda_nm, energy_ev=energy_ev)["momentum_natural"]
    lattice = TwistLattice(nx=20, kappa=kappa)
    state = CouplingState(lattice=lattice, photon=photon, kick_strength=0.08)
    for step in range(int(n_steps)):
        run_coupling_step(state, step)

    img = _fig_to_pil(_plot_momentum_history(
        state.history,
        title=(
            f"Analytic coupling  ℓ={ell}  κ={kappa:.3f}  λ={lambda_nm:.0f} nm  "
            f"E={e0:.3f} eV  p₀={p0:.5f}"
        ),
    ))
    last = state.history[-1] if state.history else {}
    md = (
        f"### Analytic momentum ledger\n"
        f"- **ℓ** = {ell} · **λ** = {lambda_nm:.0f} nm · **κ** = {kappa:.4f}\n"
        f"- **E₀** = {e0:.4f} eV · **p₀** = {p0:.6f} ×10⁻²⁷ kg·m/s · **p/(ℏk)** = {p_nat:.3f}\n"
        f"{_kick_scale_line(0.08, e_scale)}"
        f"- *Analytic kicks scale via p(E); κ_eff shown for comparison.*\n"
        f"- **Back-reaction** η_final = **{last.get('back_reaction_coupling', 1.0):.3f}** · "
        f"phase slip = **{last.get('cumulative_phase_slip', 0.0):.4f}**\n"
        f"- Final ⟨θ⟩ = **{state.lattice.mean_twist:.4f}**\n"
        f"- Δp_lattice = **{last.get('lattice_received', 0):.4f}**\n\n"
        f"{_conservation_badge(state.history)}\n"
    )
    return img, md


def run_helix_3d(
    ell: int,
    l_inner: int,
    turns: int,
    animate: bool,
    knot_mod: bool,
) -> tuple:
    """Helix-within-helix + Hopf fiber — static frame + optional GIF."""
    inner = int(l_inner) if int(l_inner) != 0 else abs(int(ell))
    geom = build_helix_geometry(
        l_outer=max(1, abs(int(ell))),
        l_inner=inner,
        active_ell=int(ell),
        turns=int(turns),
        knot_mod=bool(knot_mod),
        num_points=600,
    )
    title = f"Helix⊂Helix + Hopf  ℓ={ell}  inner={inner}  turns={turns}"
    still = _fig_to_pil(render_helix_frame(geom, azim=52, elev=22, title=title))

    anim_path = None
    anim_note = ""
    if animate:
        fig_frames = helix_animation_frames(geom, n_frames=24)
        pil_frames = figures_to_pil(fig_frames)
        tmp_dir = Path(tempfile.gettempdir())
        mp4_tmp = tmp_dir / f"oam_helix_{ell}_{inner}.mp4"
        anim_path = frames_to_mp4(pil_frames, str(mp4_tmp), fps=11.0)
        if anim_path is None:
            gif_tmp = tmp_dir / f"oam_helix_{ell}_{inner}.gif"
            frames_to_gif(pil_frames, str(gif_tmp), duration_ms=90)
            anim_path = str(gif_tmp)
            anim_note = "\n- *MP4 unavailable — GIF fallback (no native pause)*"
        else:
            anim_note = "\n- Use the **video player controls** to pause or scrub the rotation."

    md = (
        f"### Helix-within-helix 3D\n"
        f"- **Active ℓ** = {ell} · **inner** = {inner} · **turns** = {turns}\n"
        f"- **Outer** counter-propagating OAM helix (blue)\n"
        f"- **Inner** nested helix with 8₃ knot modulation (orange)\n"
        f"- **Hopf fiber** backbone on gauged lattice (teal dashed)\n"
        f"{anim_note}"
    )
    return still, anim_path, md


def run_eddington(
    ell: int,
    kappa: float,
    lambda_nm: float,
    n_flywheels: int,
    n_steps: int,
    kick_strength: float,
    energy_ev: float,
) -> tuple:
    """Mini-Eddington flywheel cluster probe."""
    st = photon_state(ell=int(ell), lambda_nm=float(lambda_nm), energy_ev=float(energy_ev))
    result = run_eddington_probe(
        kappa=float(kappa),
        ell=int(ell),
        lambda_nm=float(lambda_nm),
        n_flywheels=int(n_flywheels),
        n_steps=int(n_steps),
        kick_strength=float(kick_strength),
        energy_ev=float(energy_ev),
    )

    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=False)
    hist = result.history
    steps = [h["step"] for h in hist]
    axes[0].plot(steps, [h["lattice_received"] for h in hist], color="#e76f51", label="p_lattice")
    axes[0].plot(steps, [h["cumulative_outward"] for h in hist], color="#c9a227", label="outward flux")
    if hist and "wind_z" in hist[0]:
        axes[0].plot(
            steps,
            [h.get("wind_z", 0.0) for h in hist],
            color="#2a9d8f",
            ls="--",
            lw=1.4,
            label="Hopf wind (z)",
        )
    axes[0].set_ylabel("momentum")
    axes[0].set_xlabel("step")
    axes[0].set_title(
        f"Mini-Eddington  ℓ={ell}  κ={kappa:.3f}  κ_eff={result.effective_kick:.3f}  "
        f"flywheels={n_flywheels}"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    sites_x = np.arange(result.n_sites)
    axes[1].bar(
        sites_x,
        [s.momentum_received for s in result.sites],
        color="#457b9d",
        alpha=0.7,
        label="received",
    )
    axes[1].bar(
        sites_x,
        [s.binding for s in result.sites],
        color="#2a9d8f",
        alpha=0.5,
        label="binding κ·θ",
    )
    for idx, site in enumerate(result.sites):
        vec = site.outward_flux_vec
        mag = float(np.linalg.norm(vec))
        if mag > 1e-9:
            axes[1].quiver(
                idx,
                site.binding,
                vec[0] * 0.35,
                vec[2] * 0.35,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#c9a227",
                width=0.006,
            )
    axes[1].set_xlabel("flywheel site")
    axes[1].set_ylabel("per-site")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_img = _fig_to_pil(fig)

    unstable = sum(1 for s in result.sites if s.unstable)
    status = "🔴 **Eddington limit exceeded**" if result.limit_exceeded else "🟢 **Within binding**"
    w = result.wind_vector
    wind_mag = float(np.linalg.norm(w))
    wind_note = (
        f"- **Hopf wind** = ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}) · |wind| = **{wind_mag:.4f}**\n"
        if wind_mag > 1e-9
        else "- **Hopf wind** = none (within binding)\n"
    )
    md = (
        f"### Mini-Eddington probe\n"
        f"- **ℓ** = {ell} · **κ** = {kappa:.4f} · **λ** = {lambda_nm:.0f} nm · "
        f"**E** = {st['energy_ev']:.4f} eV\n"
        f"{_kick_scale_line(kick_strength, result.energy_scale)}"
        f"- **Back-reaction** phase slip = **{result.cumulative_phase_slip:.4f}**\n"
        f"- Flywheels = **{n_flywheels}** · unstable sites = **{unstable}**\n"
        f"- Total outward flux = **{result.total_outward_flux:.4f}**\n"
        f"{wind_note}"
        f"- {status}\n\n"
        f"When **p_received > κ·θ_binding**, excess momentum radiates **along the local "
        f"Hopf fiber** (preferred wind axis; gold arrows in per-site chart).\n"
    )
    return plot_img, md