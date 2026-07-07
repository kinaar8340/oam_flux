#!/usr/bin/env python3
"""v0.2 probe: VQC vectorized OAM propagation → Hopf fiber flux deposition → PDE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oam_flux.constants import RESIDUAL_R, load_config
from oam_flux.flux_deposit import build_flux_kick, hopf_fiber_coords
from oam_flux.lattice import TwistLattice
from oam_flux.vqc_coupling import VQCCouplingState, run_vqc_coupling_step
from oam_flux.vqc_photonics import PhotonicsConfig


def main() -> None:
    cfg = load_config()
    lat_cfg = cfg["lattice"]
    ph_cfg = cfg["photon"]
    vqc_cfg = cfg["vqc"]
    cpl_cfg = cfg["coupling"]

    lattice = TwistLattice(
        nx=lat_cfg["nx"],
        dt=lat_cfg["dt"],
        D=lat_cfg["D"],
        kappa=lat_cfg["kappa"],
        delta_omega=lat_cfg["delta_omega"],
        theta_crit=lat_cfg["theta_crit"],
    )
    photonics = PhotonicsConfig(
        l_max=vqc_cfg["l_max"],
        w0=vqc_cfg["w0"],
        nr=vqc_cfg["nr"],
        z_start=vqc_cfg["z_start"],
        z_end=vqc_cfg["z_end"],
        n_z=vqc_cfg["n_z"],
        turbulence=vqc_cfg["turbulence"],
        chirp=vqc_cfg["chirp"],
        qec_suppression=vqc_cfg["qec_suppression"],
    )
    state = VQCCouplingState.from_config(
        lattice,
        photonics,
        ell=ph_cfg["ell"],
        coupling_cfg=cpl_cfg,
    )

    n_steps = min(lat_cfg["nt"], state.propagation.n_z)
    print(f"VQC OAM–flux coupling | ℓ={state.ell} | L_max={photonics.l_max} | κ={lattice.kappa}")
    print(f"Propagation: {state.propagation.n_z} z-slices, {state.propagation.n_modes} modes")
    print(f"Residual R (mystery) = {RESIDUAL_R:.6f}")

    for step in range(n_steps):
        run_vqc_coupling_step(state, step)

    out_dir = ROOT / "outputs" / "vqc_coupling_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Timeseries ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    steps = [h["step"] for h in state.history]
    axes[0].plot(steps, [h["mean_twist"] for h in state.history], color="green", lw=1.2)
    axes[0].set_ylabel("⟨θ⟩ (rad)")
    axes[0].set_title(f"VQC-coupled lattice twist (ℓ={state.ell})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [h["mode_intensity"] for h in state.history], color="orange", lw=1.2)
    axes[1].set_ylabel(f"|ℓ={state.ell}| intensity")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, [h["momentum_ledger"] for h in state.history], color="purple", lw=1.2)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("momentum ledger")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    ts_path = out_dir / "vqc_coupling_timeseries.png"
    fig.savefig(ts_path, dpi=160)
    plt.close(fig)

    # --- VQC propagation heatmap (z × ℓ) ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im = ax2.imshow(
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
    ax2.set_xlabel("z (arb. units)")
    ax2.set_ylabel("ℓ")
    ax2.set_title("VQC multi-ℓ propagation intensity")
    fig2.colorbar(im, ax=ax2, label="intensity")
    fig2.tight_layout()
    heat_path = out_dir / "vqc_propagation_heatmap.png"
    fig2.savefig(heat_path, dpi=160)
    plt.close(fig2)

    # --- Mid-z flux deposition slice ---
    mid_z = state.propagation.n_z // 2
    kick, _ = build_flux_kick(
        lattice,
        state.propagation,
        ell=state.ell,
        z_index=mid_z,
        kick_strength=cpl_cfg["kick_strength"],
    )
    mid_slice = kick.shape[0] // 2
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    im3 = ax3.imshow(kick[mid_slice], origin="lower", cmap="RdBu_r")
    ax3.set_title(f"Flux kick slice (z_idx={mid_z}, ℓ={state.ell})")
    fig3.colorbar(im3, ax=ax3)
    fig3.tight_layout()
    kick_path = out_dir / "flux_kick_slice.png"
    fig3.savefig(kick_path, dpi=160)
    plt.close(fig3)

    summary = {
        "ell": state.ell,
        "l_max": photonics.l_max,
        "n_z": state.propagation.n_z,
        "n_modes": state.propagation.n_modes,
        "final_mean_twist": state.lattice.mean_twist,
        "final_twist_variance": state.lattice.twist_variance,
        "momentum_ledger": state.lattice.momentum_ledger,
        "mean_mode_intensity": float(np.mean([h["mode_intensity"] for h in state.history])),
        "residual_R": RESIDUAL_R,
        "n_steps": n_steps,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Final ⟨θ⟩ = {summary['final_mean_twist']:.4f}")
    print(f"Final twist σ² = {summary['final_twist_variance']:.4f}")
    print(f"Momentum ledger = {summary['momentum_ledger']:.4f}")
    print(f"Saved → {ts_path}")
    print(f"Saved → {heat_path}")
    print(f"Saved → {kick_path}")
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()