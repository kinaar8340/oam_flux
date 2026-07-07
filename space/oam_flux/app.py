#!/usr/bin/env python3
"""OAM–Flux Hugging Face Space — Gradio demo."""

from __future__ import annotations

import logging

import gradio as gr

from demo_core import (
    GITHUB_URL,
    HF_SPACE_URL,
    MYSTERY_URL,
    TOE_URL,
    VQC_URL,
    get_build_label,
    run_analytic_coupling,
    run_emergence,
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

**Orbital Angular Momentum coupled to gauged Hopf lattice flux flywheels.**

Synthesis of [toe]({TOE_URL}) (Hopf lattice PDE) · [vqc_sims_public]({VQC_URL}) (helical OAM) · [mystery]({MYSTERY_URL}) (φ/e/π emergence probes).

| Tab | What it runs |
|-----|----------------|
| **VQC Coupling** | Vectorized multi-ℓ propagation → Hopf fiber flux deposition → PDE relaxation |
| **Emergence** | κ/ℓ sweeps @ λt=2 vs R, e⁻², golden-angle analogs |
| **Analytic** | v0.1 OAM packet → flywheel kicks |

GitHub: [{GITHUB_URL}]({GITHUB_URL}) · {get_build_label()}
"""

with gr.Blocks(title="OAM–Flux", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(
        f"# 🌀 OAM–Flux\n"
        f"Helical photon flux on gauged Hopf lattice flywheels · "
        f"[GitHub]({GITHUB_URL}) · {get_build_label()}"
    )

    with gr.Tabs():
        with gr.Tab("VQC Coupling"):
            gr.Markdown("v0.2 — VQC vectorized OAM propagation deposited on lattice fibers.")
            with gr.Row():
                vqc_ell = gr.Slider(-8, 8, value=3, step=1, label="ℓ (OAM)")
                vqc_kappa = gr.Slider(0.75, 0.95, value=0.85, step=0.01, label="κ")
                vqc_kick = gr.Slider(0.01, 0.2, value=0.08, step=0.01, label="kick strength")
            with gr.Row():
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

            vqc_btn.click(
                run_vqc_coupling,
                [vqc_ell, vqc_kappa, vqc_kick, vqc_steps, vqc_lmax, vqc_turb],
                [vqc_ts, vqc_heat, vqc_kick_img, vqc_md],
            )

        with gr.Tab("Emergence"):
            gr.Markdown("v0.3 — Mystery analog probes under VQC pump-then-relax @ λt=2.")
            with gr.Row():
                em_ell = gr.Slider(-6, 6, value=3, step=1, label="probe ℓ")
                em_kappa = gr.Slider(0.80, 0.90, value=0.85, step=0.01, label="κ_doc")
                em_lmax = gr.Slider(3, 8, value=5, step=1, label="ℓ sweep L_max")
            with gr.Row():
                em_kpoints = gr.Slider(5, 21, value=11, step=2, label="κ sweep points")
                em_lambda_t = gr.Slider(1.0, 3.0, value=2.0, step=0.5, label="λt")
            em_btn = gr.Button("Run emergence probes", variant="primary")
            with gr.Row():
                em_kappa_plot = gr.Image(label="κ survival sweep")
                em_ell_plot = gr.Image(label="ℓ survival sweep")
            em_md = gr.Markdown()

            em_btn.click(
                run_emergence,
                [em_ell, em_kappa, em_kpoints, em_lmax, em_lambda_t],
                [em_kappa_plot, em_ell_plot, em_md],
            )

        with gr.Tab("Analytic"):
            gr.Markdown("v0.1 — analytic OAM packet coupling (fast baseline).")
            with gr.Row():
                an_ell = gr.Slider(-8, 8, value=3, step=1, label="ℓ")
                an_kappa = gr.Slider(0.75, 0.95, value=0.85, step=0.01, label="κ")
                an_steps = gr.Slider(50, 500, value=150, step=50, label="steps")
            an_btn = gr.Button("Run analytic coupling", variant="secondary")
            an_img = gr.Image(label="Timeseries")
            an_md = gr.Markdown()
            an_btn.click(run_analytic_coupling, [an_ell, an_kappa, an_steps], [an_img, an_md])

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

if __name__ == "__main__":
    demo.launch()