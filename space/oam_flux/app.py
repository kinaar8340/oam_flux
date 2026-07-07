#!/usr/bin/env python3
"""OAM–Flux Hugging Face Space — Gradio demo."""

from __future__ import annotations

import logging

import gradio as gr

from demo_core import (
    GITHUB_URL,
    MYSTERY_URL,
    TOE_URL,
    VQC_URL,
    get_build_label,
    run_analytic_coupling,
    run_eddington,
    run_emergence,
    run_helix_3d,
    run_vqc_coupling,
)

logger = logging.getLogger(__name__)


def _patch_gradio_client_bool_schema() -> None:
    try:
        from gradio_client import utils as client_utils
        if getattr(client_utils, "_oam_bool_patch", False):
            return
        orig = client_utils.get_type

        def get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            return orig(schema)

        client_utils.get_type = get_type
        client_utils._oam_bool_patch = True
    except Exception:
        logger.warning("gradio_client patch skipped", exc_info=True)


_patch_gradio_client_bool_schema()

ABOUT_MD = f"""
# OAM–Flux

Synthesis of [toe]({TOE_URL}) · [vqc_sims_public]({VQC_URL}) · [mystery]({MYSTERY_URL}).

| Tab | Feature |
|-----|---------|
| **VQC Coupling** | Multi-ℓ propagation + flux kicks + momentum ledger |
| **Emergence** | κ/ℓ sweeps (synced from VQC ℓ, κ) |
| **Helix 3D** | Helix-within-helix + Hopf fiber animation |
| **Eddington** | Flywheel cluster binding vs outward flux |
| **Analytic** | Fast OAM packet baseline |

GitHub: [{GITHUB_URL}]({GITHUB_URL}) · {get_build_label()}
"""

with gr.Blocks(title="OAM–Flux", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(
        f"# 🌀 OAM–Flux\n"
        f"Helical photon flux on gauged Hopf lattice flywheels · "
        f"[GitHub]({GITHUB_URL}) · {get_build_label()}"
    )

    sync_note = gr.Markdown("**Cross-tab sync:** Emergence inherits ℓ, κ, L_max from VQC Coupling.")

    with gr.Tabs():
        with gr.Tab("VQC Coupling"):
            gr.Markdown(
                "**VQC** — Multi-ℓ propagation → Hopf fiber flux kicks. "
                "Momentum **p ∝ |ℓ|/λ**."
            )
            vqc_status = gr.Markdown("*Adjust ℓ and κ — Emergence tab follows automatically.*")
            with gr.Row():
                vqc_ell = gr.Slider(-8, 8, value=3, step=1, label="ℓ (OAM quantum number)")
                vqc_kappa = gr.Slider(0.75, 0.95, value=0.85, step=0.01, label="κ (gauge damping)")
                vqc_lambda = gr.Slider(400, 2000, value=1550, step=10, label="λ (nm)")
            with gr.Row():
                vqc_kick = gr.Slider(0.01, 0.2, value=0.08, step=0.01, label="kick strength")
                vqc_steps = gr.Slider(20, 200, value=80, step=10, label="steps")
                vqc_lmax = gr.Slider(3, 12, value=6, step=1, label="L_max")
                vqc_turb = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="turbulence")
            vqc_btn = gr.Button("Run VQC coupling", variant="primary")
            with gr.Row():
                vqc_ts = gr.Image(label="Timeseries")
                vqc_heat = gr.Image(label="Propagation ℓ×z")
            with gr.Row():
                vqc_kick_img = gr.Image(label="Flux kick slice")
                vqc_md = gr.Markdown()

        with gr.Tab("Emergence"):
            gr.Markdown("**Mystery probes** @ λt=2 — ℓ, κ, L_max synced from VQC tab.")
            em_sync_badge = gr.Markdown("*Synced from VQC*")
            with gr.Row():
                em_ell = gr.Slider(-8, 8, value=3, step=1, label="probe ℓ (synced)")
                em_kappa = gr.Slider(0.75, 0.95, value=0.85, step=0.01, label="κ (synced)")
                em_lmax = gr.Slider(3, 12, value=6, step=1, label="ℓ sweep L_max (synced)")
            with gr.Row():
                em_kpoints = gr.Slider(5, 21, value=11, step=2, label="κ sweep points")
                em_lambda_t = gr.Slider(1.0, 3.0, value=2.0, step=0.5, label="λt")
            em_btn = gr.Button("Run emergence probes", variant="primary")
            with gr.Row():
                em_kappa_plot = gr.Image(label="κ survival sweep")
                em_ell_plot = gr.Image(label="ℓ survival sweep")
            em_md = gr.Markdown()

        with gr.Tab("Helix 3D"):
            gr.Markdown("**Helix-within-helix** (VQC) + **Hopf fiber** (toe lattice) — kinetic flux geometry.")
            with gr.Row():
                hx_ell = gr.Slider(-8, 8, value=3, step=1, label="ℓ (active OAM)")
                hx_inner = gr.Slider(1, 20, value=3, step=1, label="inner |ℓ|")
                hx_turns = gr.Slider(3, 12, value=6, step=1, label="turns")
            with gr.Row():
                hx_animate = gr.Checkbox(value=True, label="Generate rotation GIF")
                hx_knot = gr.Checkbox(value=True, label="8₃ knot modulation")
            hx_btn = gr.Button("Render helix 3D", variant="primary")
            with gr.Row():
                hx_still = gr.Image(label="3D snapshot")
                hx_gif = gr.Image(label="Rotation animation")
            hx_md = gr.Markdown()

        with gr.Tab("Eddington"):
            gr.Markdown(
                "**Mini-Eddington** — flywheel cluster: outward flux when "
                "**p_received > κ·θ_binding**."
            )
            with gr.Row():
                ed_ell = gr.Slider(-8, 8, value=3, step=1, label="ℓ")
                ed_kappa = gr.Slider(0.75, 0.95, value=0.85, step=0.01, label="κ")
                ed_lambda = gr.Slider(400, 2000, value=1550, step=10, label="λ (nm)")
            with gr.Row():
                ed_flywheels = gr.Slider(2, 10, value=6, step=1, label="flywheel sites")
                ed_steps = gr.Slider(20, 150, value=80, step=10, label="steps")
                ed_kick = gr.Slider(0.02, 0.2, value=0.08, step=0.01, label="kick strength")
            ed_btn = gr.Button("Run Eddington probe", variant="primary")
            ed_plot = gr.Image(label="Momentum + per-site binding")
            ed_md = gr.Markdown()

        with gr.Tab("Analytic"):
            gr.Markdown("v0.1 — analytic OAM packet; photon vs lattice momentum.")
            with gr.Row():
                an_ell = gr.Slider(-8, 8, value=3, step=1, label="ℓ")
                an_kappa = gr.Slider(0.75, 0.95, value=0.85, step=0.01, label="κ")
                an_lambda = gr.Slider(400, 2000, value=1550, step=10, label="λ (nm)")
                an_steps = gr.Slider(50, 500, value=150, step=50, label="steps")
            an_btn = gr.Button("Run analytic coupling", variant="secondary")
            an_img = gr.Image(label="Momentum + twist timeseries")
            an_md = gr.Markdown()

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    # --- Event handlers ---
    def _vqc_preview(ell, kappa, lam):
        from oam_flux.momentum import oam_kinetic_momentum
        p = oam_kinetic_momentum(energy_scale=1.0, ell=int(ell), lambda_nm=lam)
        return (
            f"**Active:** ℓ={int(ell)} · κ={kappa:.3f} · λ={lam:.0f} nm · "
            f"**p₀ ≈ {p:.6f}** · R_ref=0.137486"
        )

    def _sync_from_vqc(ell, kappa, lmax):
        badge = f"**Synced:** ℓ={int(ell)} · κ={kappa:.3f} · L_max={int(lmax)} (from VQC)"
        return int(ell), float(kappa), int(lmax), badge

    def _sync_vqc_to_helix_ed(ell, lam):
        return int(ell), int(ell), float(lam)

    for ctrl in (vqc_ell, vqc_kappa, vqc_lambda):
        ctrl.change(_vqc_preview, [vqc_ell, vqc_kappa, vqc_lambda], [vqc_status])

    for ctrl in (vqc_ell, vqc_kappa, vqc_lmax):
        ctrl.change(
            _sync_from_vqc,
            [vqc_ell, vqc_kappa, vqc_lmax],
            [em_ell, em_kappa, em_lmax, em_sync_badge],
        )

    vqc_ell.change(
        _sync_vqc_to_helix_ed,
        [vqc_ell, vqc_lambda],
        [hx_ell, ed_ell, ed_lambda],
    )
    vqc_lambda.change(
        lambda lam: lam,
        [vqc_lambda],
        [ed_lambda, an_lambda],
    )
    vqc_kappa.change(
        lambda k: k,
        [vqc_kappa],
        [ed_kappa, an_kappa],
    )

    vqc_btn.click(
        run_vqc_coupling,
        [vqc_ell, vqc_kappa, vqc_kick, vqc_steps, vqc_lmax, vqc_turb, vqc_lambda],
        [vqc_ts, vqc_heat, vqc_kick_img, vqc_md],
    )
    em_btn.click(
        run_emergence,
        [em_ell, em_kappa, em_kpoints, em_lmax, em_lambda_t],
        [em_kappa_plot, em_ell_plot, em_md],
    )
    hx_btn.click(
        run_helix_3d,
        [hx_ell, hx_inner, hx_turns, hx_animate, hx_knot],
        [hx_still, hx_gif, hx_md],
    )
    ed_btn.click(
        run_eddington,
        [ed_ell, ed_kappa, ed_lambda, ed_flywheels, ed_steps, ed_kick],
        [ed_plot, ed_md],
    )
    an_btn.click(
        run_analytic_coupling,
        [an_ell, an_kappa, an_steps, an_lambda],
        [an_img, an_md],
    )

    demo.load(
        _sync_from_vqc,
        inputs=[vqc_ell, vqc_kappa, vqc_lmax],
        outputs=[em_ell, em_kappa, em_lmax, em_sync_badge],
    )

if __name__ == "__main__":
    demo.launch()