#!/usr/bin/env python3
"""CLI: mini-Eddington flywheel cluster probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oam_flux.constants import load_config
from oam_flux.eddington import run_eddington_probe


def main() -> None:
    cfg = load_config()
    lat = cfg.get("lattice", {})
    ph = cfg.get("photon", {})
    cpl = cfg.get("coupling", {})

    result = run_eddington_probe(
        kappa=float(lat.get("kappa", 0.85)),
        ell=int(ph.get("ell", 3)),
        lambda_nm=float(ph.get("lambda_nm", 1550.0)),
        n_flywheels=int(cpl.get("flywheel_sites", 6)),
        n_steps=80,
        kick_strength=float(cpl.get("kick_strength", 0.08)),
    )

    out = ROOT / "outputs" / "eddington_probe"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        [h["step"] for h in result.history],
        [h["cumulative_outward"] for h in result.history],
        color="#c9a227",
        lw=2,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative outward flux")
    ax.set_title(f"Mini-Eddington  κ={result.kappa}  ℓ={result.ell}")
    ax.grid(alpha=0.3)
    fig.savefig(out / "eddington_outward_flux.png", dpi=150)
    plt.close(fig)

    summary = result.to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()