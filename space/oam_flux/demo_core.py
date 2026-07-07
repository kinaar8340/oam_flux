"""HF Space demo core — wraps oam_flux simulations for Gradio."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _annotate_pulse_train(axes, history: list[dict]) -> None:
    """Mark pump onsets and recovery gaps on the timeseries."""
    if not history or "pulse_index" not in history[0]:
        return
    seen: set[int] = set()
    for h in history:
        pid = int(h.get("pulse_index", 0))
        if pid not in seen and h.get("pulse_phase_pump", 0) > 0.5:
            axes[0].axvline(h["step"], color="#457b9d", ls=":", lw=0.9, alpha=0.55)
            seen.add(pid)
    rec_start: float | None = None
    for h in history:
        if h.get("recovery_active", 0) > 0.5:
            if rec_start is None:
                rec_start = h["step"]
        elif rec_start is not None:
            axes[0].axvspan(rec_start, h["step"], color="#c9a227", alpha=0.08)
            rec_start = None
    if rec_start is not None:
        axes[0].axvspan(rec_start, history[-1]["step"], color="#c9a227", alpha=0.08)


def _pulse_train_summary_line(state) -> str:
    if not getattr(state, "pulse_train_mode", False) or state.pulses_fired <= 1:
        return ""
    mem = float(getattr(state, "recovery_memory", 0.0))
    mem_note = (
        f" · recovery memory = **{mem:.2f}** (0 = full θ reset, 1 = carry twist)"
        if mem > 0.0
        else ""
    )
    shape = getattr(state, "pulse_shape", "square")
    return (
        f"- **Pulse train** = {state.pulses_fired} pulses · "
        f"shape = **{shape}** · "
        f"total injected = **{state.total_injected:.4f}** (p₀ each){mem_note}\n"
    )


def _pulse_cumulative_block(history: list[dict]) -> str:
    from oam_flux.pulse_stats import compute_pulse_statistics

    stats = compute_pulse_statistics(history)
    cum = stats["cumulative"]
    if not stats["pulses"]:
        return (
            f"- **Cumulative** phase slip = **{cum['total_phase_slip']:.4f}** · "
            f"p_lattice received = **{cum['total_lattice_momentum']:.4f}**\n"
        )
    lines = [
        f"- **Cumulative** phase slip = **{cum['total_phase_slip']:.4f}** · "
        f"p_lattice received = **{cum['total_lattice_momentum']:.4f}** · "
        f"⟨η_min⟩/pulse = **{cum['mean_eta_pump_min']:.3f}**\n",
        "- **Per pulse** (η_min → η_gap · slip · Δp_lattice):\n",
    ]
    for p in stats["pulses"]:
        lines.append(
            f"  - #{p['pulse']}: **{p['eta_min']:.3f}** → **{p['eta_after_gap']:.3f}** · "
            f"slip **{p['slip']:.4f}** · lattice **{p['lattice_deposited']:.4f}**\n"
        )
    return "".join(lines)


def _recovery_summary_line(history: list[dict], *, recovery_tau: float | None = None) -> str:
    if not history or "recovery_steps" not in history[-1]:
        return ""
    last = history[-1]
    rec = int(last.get("recovery_steps", 0))
    if rec <= 0:
        return "- **Recovery** = none (pump active throughout)\n"
    etas = [h.get("back_reaction_coupling", 1.0) for h in history]
    eta_min = min(etas) if etas else 1.0
    eta_final = history[-1].get("back_reaction_coupling", 1.0)
    tau = recovery_tau if recovery_tau is not None else last.get("recovery_tau")
    tau_note = f" · τ = **{float(tau):.1f}** steps (exp)" if tau is not None else ""
    return (
        f"- **Recovery** = {rec} steps after pump off{tau_note} · "
        f"η: {eta_min:.3f} → **{eta_final:.3f}**\n"
    )


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
    if history and "pulse_index" in history[0]:
        _annotate_pulse_train(axes, history)
    elif history and "recovery_active" in history[0]:
        rec_steps = [h["step"] for h in history if h.get("recovery_active", 0) > 0.5]
        if rec_steps:
            axes[0].axvspan(
                rec_steps[0], steps[-1], color="#c9a227", alpha=0.08, label="recovery",
            )
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
        if any(h.get("cumulative_phase_slip", 0) > 0 for h in history):
            ax.plot(
                steps,
                [h.get("cumulative_phase_slip", 0) for h in history],
                color="#9b2226",
                ls="-.",
                lw=1.1,
                alpha=0.85,
                label="cum. phase slip",
            )
        if any(h.get("pump_envelope", 0) > 0 for h in history):
            env_curve = [h.get("pump_envelope", 0.0) * p0 for h in history]
            ax.fill_between(
                steps, 0, env_curve, color="#2a9d8f", alpha=0.12, label="pump envelope",
            )
            ax.plot(steps, env_curve, color="#2a9d8f", ls=":", lw=0.9, alpha=0.55)
    ax.set_xlabel("step")
    ax.set_ylabel("momentum (norm.)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _build_vqc_state(
    *,
    ell: int,
    kappa: float,
    kick_strength: float,
    sim_steps: int,
    l_max: int,
    turbulence: float,
    lambda_nm: float,
    e_scale: float,
    recovery_tau: float,
) -> tuple[VQCCouplingState, TwistLattice]:
    lattice = TwistLattice(nx=20, kappa=kappa, recovery_tau=float(recovery_tau))
    photonics = PhotonicsConfig(
        l_max=l_max,
        n_z=min(int(sim_steps), 200),
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
    state.recovery_tau = float(recovery_tau)
    return state, lattice


def _plot_equivalence_comparison(
    hist_cont: list[dict],
    hist_pulse: list[dict],
    *,
    title: str,
    p0: float,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col")

    def _plot_row(ax_twist, ax_mom, history: list[dict], label: str, *, annotate: bool) -> None:
        if not history:
            return
        steps = [h["step"] for h in history]
        ax_twist.plot(steps, [h["mean_twist"] for h in history], color="#2a9d8f", lw=1.4, label="⟨θ⟩")
        if annotate:
            _annotate_pulse_train([ax_twist], history)
        ax_eta = ax_twist.twinx()
        ax_eta.plot(
            steps,
            [h.get("back_reaction_coupling", 1.0) for h in history],
            color="#6a4c93",
            ls="--",
            lw=1.0,
            label="η",
        )
        ax_eta.set_ylim(0.0, 1.05)
        ax_twist.set_ylabel(f"{label} ⟨θ⟩")
        ax_twist.grid(alpha=0.3)

        ax_mom.plot(steps, [h["photon_momentum"] for h in history], color="#457b9d", lw=1.5, label="p_photon")
        ax_mom.plot(
            steps,
            [h.get("lattice_received", 0.0) for h in history],
            color="#e76f51",
            lw=1.5,
            label="p_lattice",
        )
        ax_mom.axhline(p0, color="#c9a227", ls=":", lw=1.0, alpha=0.8)
        ax_mom.set_ylabel(f"{label} momentum")
        ax_mom.grid(alpha=0.3)

    _plot_row(axes[0, 0], axes[1, 0], hist_cont, "Continuous", annotate=False)
    _plot_row(axes[0, 1], axes[1, 1], hist_pulse, "Pulsed", annotate=True)
    axes[1, 0].set_xlabel("step")
    axes[1, 1].set_xlabel("step")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def _equivalence_summary_md(
    cont: dict,
    pulse: dict,
    delta: dict,
    *,
    n_pulses: int,
    p0: float,
    pump_steps: int,
    gap_steps: int,
    recovery_tau: float,
    pulse_shape: str,
) -> str:
    target = float(n_pulses) * float(p0)
    return (
        f"### Dose equivalence — **{int(n_pulses)} × p₀** = **{target:.4f}** total injected\n"
        f"Same horizon: **{int(cont['sim_steps'])}** steps "
        f"(pulse window {int(pump_steps)} + gap {int(gap_steps)} × {int(n_pulses)})\n\n"
        "| Metric | Continuous | Pulsed | Δ (pulsed − cont) |\n"
        "|--------|------------|--------|-------------------|\n"
        f"| Total injected | {cont['total_injected']:.4f} | {pulse['total_injected']:.4f} | "
        f"{pulse['total_injected'] - cont['total_injected']:+.4f} |\n"
        f"| p_lattice received | {cont['lattice_received']:.4f} | {pulse['lattice_received']:.4f} | "
        f"{delta['lattice_received']:+.4f} |\n"
        f"| Phase slip | {cont['cumulative_phase_slip']:.4f} | {pulse['cumulative_phase_slip']:.4f} | "
        f"{delta['cumulative_phase_slip']:+.4f} |\n"
        f"| η_final | {cont['eta_final']:.3f} | {pulse['eta_final']:.3f} | "
        f"{delta['eta_final']:+.3f} |\n"
        f"| ⟨θ⟩_final | {cont['mean_twist_final']:.4f} | {pulse['mean_twist_final']:.4f} | "
        f"{delta['mean_twist_final']:+.4f} |\n"
        f"| Twist load | {cont['twist_load']:.4f} | {pulse['twist_load']:.4f} | "
        f"{delta['twist_load']:+.4f} |\n"
        f"| Recovery steps | {int(cont['recovery_steps'])} | {int(pulse['recovery_steps'])} | "
        f"{delta['recovery_steps']:+.0f} |\n\n"
        f"- **Pulsed** shape = **{pulse_shape}** · τ = **{recovery_tau:.0f}** steps\n"
        f"- Continuous = same pump/gap schedule · immediate refill in pump windows · "
        f"shared recovery memory\n"
    )


def _compute_dose_equivalence_matrix(
    *,
    ell: int,
    kappa: float,
    kick_strength: float,
    l_max: int,
    turbulence: float,
    lambda_nm: float,
    e_scale: float,
    recovery_tau: float,
    n_pulses: int,
    pump_steps: int,
    gap_steps: int,
    p0: float,
    memories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Δ load and Δ slip grids [shape, memory] for square and gaussian."""
    from oam_flux.dose_equivalence import (
        DoseEquivalenceConfig,
        compare_delivery_metrics,
        extract_delivery_metrics,
        run_continuous_dose_matched,
        run_pulsed_dose_matched,
    )
    from oam_flux.pulse_train import PulseTrainConfig

    horizon = int(n_pulses) * (int(pump_steps) + int(gap_steps))
    dcfg = DoseEquivalenceConfig(n_pulses=int(n_pulses), p0=p0, max_steps=horizon)
    shapes = ("square", "gaussian")
    delta_load = np.zeros((len(shapes), len(memories)))
    delta_slip = np.zeros((len(shapes), len(memories)))

    for row, shape in enumerate(shapes):
        for col, mem in enumerate(memories):
            pcfg = PulseTrainConfig(
                n_pulses=int(n_pulses),
                pump_steps=int(pump_steps),
                gap_steps=int(gap_steps),
                recovery_memory=float(mem),
                recovery_tau=float(recovery_tau),
                pulse_shape=shape,
            )
            state_cont, _ = _build_vqc_state(
                ell=ell, kappa=kappa, kick_strength=kick_strength, sim_steps=horizon,
                l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
                e_scale=e_scale, recovery_tau=recovery_tau,
            )
            state_pulse, _ = _build_vqc_state(
                ell=ell, kappa=kappa, kick_strength=kick_strength, sim_steps=horizon,
                l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
                e_scale=e_scale, recovery_tau=recovery_tau,
            )
            run_continuous_dose_matched(state_cont, dcfg, pcfg)
            run_pulsed_dose_matched(state_pulse, pcfg)
            delta = compare_delivery_metrics(
                extract_delivery_metrics(state_cont),
                extract_delivery_metrics(state_pulse),
            )
            delta_load[row, col] = delta["twist_load"]
            delta_slip[row, col] = delta["cumulative_phase_slip"]

    return delta_load, delta_slip


def _plot_dose_equivalence_matrix_heatmap(
    memories: np.ndarray,
    delta_load: np.ndarray,
    delta_slip: np.ndarray,
    *,
    current_memory: float,
    n_pulses: int,
    pump_steps: int,
    gap_steps: int,
    recovery_tau: float,
) -> plt.Figure:
    """Heatmap of Δ(pulsed − continuous) over shape × recovery memory."""
    shapes = ["square", "gaussian"]
    mem_labels = [f"{m:.1f}" for m in memories]
    cur_col = int(np.argmin(np.abs(memories - float(current_memory))))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))

    panels = (
        (axes[0], delta_load * 1e3, "Δ twist load ×10³", "YlOrRd"),
        (axes[1], delta_slip, "Δ phase slip", "Purples"),
    )
    for ax, data, label, cmap in panels:
        vmax = max(float(np.max(np.abs(data))), 1e-6)
        if label.startswith("Δ twist"):
            vmax = max(vmax, 0.05)
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            extent=[-0.5, len(memories) - 0.5, -0.5, len(shapes) - 0.5],
        )
        ax.set_xticks(range(len(memories)))
        ax.set_xticklabels(mem_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(shapes)))
        ax.set_yticklabels(shapes)
        ax.set_xlabel("Recovery memory")
        ax.set_title(label)
        # highlight active memory column
        ax.add_patch(plt.Rectangle(
            (cur_col - 0.5, -0.5),
            1.0,
            len(shapes),
            fill=False,
            edgecolor="#2a9d8f",
            lw=2.5,
            ls="--",
        ))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Dose equiv matrix  {int(n_pulses)}×p₀  "
        f"{int(pump_steps)}+{int(gap_steps)}  fair continuous  τ={recovery_tau:.0f}",
        fontsize=11,
    )
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
    pulse_train: bool = False,
    n_pulses: int = 3,
    pump_steps: int = 30,
    gap_steps: int = 20,
    recovery_memory: float = 0.0,
    recovery_tau: float = 25.0,
    pulse_shape: str = "square",
    dose_equivalence: bool = False,
) -> tuple:
    """VQC coupling demo → (timeseries, propagation, kick, matrix_heatmap, summary_md)."""
    from oam_flux.dose_equivalence import (
        DoseEquivalenceConfig,
        compare_delivery_metrics,
        extract_delivery_metrics,
        run_continuous_dose_matched,
        run_pulsed_dose_matched,
    )
    from oam_flux.pulse_train import PulseTrainConfig, run_vqc_pulse_train

    st = photon_state(ell=int(ell), lambda_nm=float(lambda_nm), energy_ev=float(energy_ev))
    e_scale = st["energy_scale"]
    p0 = st["momentum"]
    e0 = st["energy_ev"]
    f0 = st["frequency_thz"]
    p_nat = st["momentum_natural"]

    pcfg = PulseTrainConfig(
        n_pulses=int(n_pulses),
        pump_steps=int(pump_steps),
        gap_steps=int(gap_steps),
        recovery_memory=float(recovery_memory),
        recovery_tau=float(recovery_tau),
        pulse_shape=str(pulse_shape),
    )

    if dose_equivalence:
        sim_steps = pcfg.total_steps
        state_cont, lattice_cont = _build_vqc_state(
            ell=ell, kappa=kappa, kick_strength=kick_strength, sim_steps=sim_steps,
            l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
            e_scale=e_scale, recovery_tau=recovery_tau,
        )
        state_pulse, lattice = _build_vqc_state(
            ell=ell, kappa=kappa, kick_strength=kick_strength, sim_steps=sim_steps,
            l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
            e_scale=e_scale, recovery_tau=recovery_tau,
        )
        dcfg = DoseEquivalenceConfig(
            n_pulses=int(n_pulses), p0=p0, max_steps=sim_steps,
        )
        run_continuous_dose_matched(state_cont, dcfg, pcfg)
        run_pulsed_dose_matched(state_pulse, pcfg)
        cont_m = extract_delivery_metrics(state_cont)
        pulse_m = extract_delivery_metrics(state_pulse)
        delta = compare_delivery_metrics(cont_m, pulse_m)
        state = state_pulse
        steps = int(pulse_m["sim_steps"])
        ts_img = _fig_to_pil(_plot_equivalence_comparison(
            state_cont.history,
            state_pulse.history,
            title=(
                f"Dose equiv  ℓ={ell}  κ={kappa:.3f}  "
                f"{int(n_pulses)}×p₀  λ={lambda_nm:.0f} nm"
            ),
            p0=p0,
        ))
        md_equiv = _equivalence_summary_md(
            cont_m, pulse_m, delta,
            n_pulses=int(n_pulses), p0=p0,
            pump_steps=int(pump_steps), gap_steps=int(gap_steps),
            recovery_tau=float(recovery_tau), pulse_shape=str(pulse_shape),
        )
        mem_grid = np.linspace(0.0, 1.0, 11)
        delta_load, delta_slip = _compute_dose_equivalence_matrix(
            ell=ell, kappa=kappa, kick_strength=kick_strength,
            l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
            e_scale=e_scale, recovery_tau=float(recovery_tau),
            n_pulses=int(n_pulses), pump_steps=int(pump_steps), gap_steps=int(gap_steps),
            p0=p0, memories=mem_grid,
        )
        matrix_img = _fig_to_pil(_plot_dose_equivalence_matrix_heatmap(
            mem_grid, delta_load, delta_slip,
            current_memory=float(recovery_memory),
            n_pulses=int(n_pulses), pump_steps=int(pump_steps), gap_steps=int(gap_steps),
            recovery_tau=float(recovery_tau),
        ))
    elif pulse_train:
        sim_steps = pcfg.total_steps
        state, lattice = _build_vqc_state(
            ell=ell, kappa=kappa, kick_strength=kick_strength, sim_steps=sim_steps,
            l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
            e_scale=e_scale, recovery_tau=recovery_tau,
        )
        steps = run_vqc_pulse_train(state, pcfg)
        mode = f"pulse×{int(n_pulses)} ({int(pump_steps)}+{int(gap_steps)}, {pulse_shape})"
        ts_img = _fig_to_pil(_plot_momentum_history(
            state.history,
            title=(
                f"VQC {mode}  ℓ={ell}  κ={kappa:.3f}  λ={lambda_nm:.0f} nm  "
                f"E={e0:.3f} eV  p₀={p0:.5f}"
            ),
        ))
        md_equiv = None
        matrix_img = None
    else:
        sim_steps = int(n_steps)
        state, lattice = _build_vqc_state(
            ell=ell, kappa=kappa, kick_strength=kick_strength, sim_steps=sim_steps,
            l_max=l_max, turbulence=turbulence, lambda_nm=lambda_nm,
            e_scale=e_scale, recovery_tau=recovery_tau,
        )
        steps = min(sim_steps, state.propagation.n_z)
        for step in range(steps):
            run_vqc_coupling_step(state, step)
        ts_img = _fig_to_pil(_plot_momentum_history(
            state.history,
            title=(
                f"VQC continuous  ℓ={ell}  κ={kappa:.3f}  λ={lambda_nm:.0f} nm  "
                f"E={e0:.3f} eV  p₀={p0:.5f}"
            ),
        ))
        md_equiv = None
        matrix_img = None

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
    if dose_equivalence and md_equiv is not None:
        md = (
            md_equiv
            + f"\n**Photon readout:** p₀ = {p0:.6f} · E₀ = {e0:.4f} eV · "
            f"p/(ℏk) = {p_nat:.3f}\n"
            f"{_kick_scale_line(kick_strength, e_scale)}"
            f"- **Matrix heatmap:** shape × memory (0→1) · dashed column = active memory\n"
        )
    else:
        md = (
            f"### VQC coupling — ℓ={ell} · κ={kappa:.4f} · λ={lambda_nm:.0f} nm\n"
            f"- **E₀** = {e0:.4f} eV  (E = hc/λ) · **f** = {f0:.2f} THz\n"
            f"- **p₀** = {p0:.6f} ×10⁻²⁷ kg·m/s  (p = h|ℓ|/λ) · **p/(ℏk)** = {p_nat:.3f}\n"
            f"{_kick_scale_line(kick_strength, e_scale)}"
            f"{_pulse_train_summary_line(state)}"
            f"{_pulse_cumulative_block(state.history) if pulse_train else ''}"
            f"- **Back-reaction** η_final = **{last.get('back_reaction_coupling', 1.0):.3f}** · "
            f"phase slip = **{last.get('cumulative_phase_slip', 0.0):.4f}**\n"
            f"{_recovery_summary_line(state.history, recovery_tau=float(recovery_tau))}"
            f"- Final ⟨θ⟩ = **{state.lattice.mean_twist:.4f}**\n"
            f"- p_photon final = **{last.get('photon_momentum', 0):.4f}**\n"
            f"- p_lattice received = **{last.get('lattice_received', 0):.4f}**\n"
            f"- Residual R = **{RESIDUAL_R:.6f}** (mystery)\n\n"
            f"{_conservation_badge(state.history)}\n"
        )
    return ts_img, heat_img, kick_img, matrix_img, md


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
        f"{_recovery_summary_line(state.history)}"
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
        f"{_recovery_summary_line(result.history)}"
        f"- Flywheels = **{n_flywheels}** · unstable sites = **{unstable}**\n"
        f"- Total outward flux = **{result.total_outward_flux:.4f}**\n"
        f"{wind_note}"
        f"- {status}\n\n"
        f"When **p_received > κ·θ_binding**, excess momentum radiates **along the local "
        f"Hopf fiber** (preferred wind axis; gold arrows in per-site chart).\n"
    )
    return plot_img, md