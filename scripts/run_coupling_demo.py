#!/usr/bin/env python3
"""First integration probe: OAM packet → flywheel kicks → PDE relaxation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oam_flux.constants import RESIDUAL_R, load_config
from oam_flux.coupling import CouplingState, run_coupling_step
from oam_flux.lattice import TwistLattice
from oam_flux.photon import OAMPacket


def main() -> None:
    cfg = load_config()
    lat_cfg = cfg["lattice"]
    ph_cfg = cfg["photon"]
    cpl_cfg = cfg["coupling"]

    lattice = TwistLattice(
        nx=lat_cfg["nx"],
        dt=lat_cfg["dt"],
        D=lat_cfg["D"],
        kappa=lat_cfg["kappa"],
        delta_omega=lat_cfg["delta_omega"],
        theta_crit=lat_cfg["theta_crit"],
    )
    photon = OAMPacket(
        ell=ph_cfg["ell"],
        lambda_nm=ph_cfg["lambda_nm"],
        w0=ph_cfg["w0"],
        energy_scale=ph_cfg["energy_scale"],
    )
    state = CouplingState(
        lattice=lattice,
        photon=photon,
        kick_strength=cpl_cfg["kick_strength"],
        flywheel_sites=cpl_cfg["flywheel_sites"],
        conserve_momentum=cpl_cfg["conserve_momentum"],
    )

    n_steps = lat_cfg["nt"]
    print(f"OAM–flux coupling demo | ℓ={photon.ell} | κ={lattice.kappa} | steps={n_steps}")
    print(f"Residual R (mystery) = {RESIDUAL_R:.6f}")

    for step in range(n_steps):
        run_coupling_step(state, step)

    out_dir = ROOT / "outputs" / "coupling_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    steps = [h["step"] for h in state.history]
    axes[0].plot(steps, [h["mean_twist"] for h in state.history], color="green", lw=1.2)
    axes[0].set_ylabel("⟨θ⟩ (rad)")
    axes[0].set_title("Lattice mean twist under OAM flywheel coupling")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [h["photon_momentum"] for h in state.history], label="photon p", color="orange")
    axes[1].plot(steps, [h["momentum_ledger"] for h in state.history], label="ledger", color="purple")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("momentum (norm.)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "coupling_timeseries.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "final_mean_twist": state.lattice.mean_twist,
        "final_twist_variance": state.lattice.twist_variance,
        "final_photon_energy": state.photon.energy_scale,
        "momentum_ledger": state.lattice.momentum_ledger,
        "residual_R": RESIDUAL_R,
        "n_steps": n_steps,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Final ⟨θ⟩ = {summary['final_mean_twist']:.4f}")
    print(f"Final twist σ² = {summary['final_twist_variance']:.4f}")
    print(f"Momentum ledger = {summary['momentum_ledger']:.4f}")
    print(f"Saved → {plot_path}")
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()