# OAM–Flux

[![Repository](https://img.shields.io/badge/GitHub-oam__flux-blue)](https://github.com/kinaar8340/oam_flux)

**Orbital Angular Momentum coupled to gauged Hopf lattice flux flywheels.**

Synthesis layer unifying three existing simulation stacks:

| Repo | Role in `oam_flux` |
|------|-------------------|
| [toe](https://github.com/kinaar8340/toe) | Gauged Hopf lattice, twist PDE, flux flywheels, κ / W_g locks |
| [vqc_sims_public](https://github.com/kinaar8340/vqc_sims_public) | Helical OAM propagation, LG modes, OAM-flux qubit dynamics |
| [mystery](https://github.com/kinaar8340/mystery) | Emergent probes: residual R, golden angle, e⁻² survival @ λt=2 |

## Concept

Photons are modeled as propagating helical twist packets (OAM modes). The lattice background is a porous gauged Hopf medium with discrete flux-flywheel resonators. Momentum transfers from photon kinetic flux to flywheel twist increments while the PDE relaxes under global gauge torque `−κ⟨θ⟩`.

## Quick start

```bash
cd ~/Projects/oam_flux
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# v0.1 — analytic OAM packet coupling
.venv/bin/python scripts/run_coupling_demo.py

# v0.2 — VQC vectorized propagation + fiber flux deposition
.venv/bin/python scripts/run_vqc_coupling_demo.py

# Tests
.venv/bin/pytest -q
```

Outputs: `outputs/coupling_demo/` (v0.1) · `outputs/vqc_coupling_demo/` (v0.2).

## Module map

```
src/oam_flux/
├── constants.py     # W_g, κ_doc, κ_sim, κ*, residual R
├── lattice.py       # Twist PDE + helical IC (from toe pde_relaxation)
├── photon.py        # Analytic LG OAM packets (v0.1)
├── coupling.py      # Analytic momentum ledger + flywheel kicks (v0.1)
├── vqc_photonics.py # VQC vectorized multi-ℓ propagation (v0.2)
├── flux_deposit.py  # Hopf fiber coords + flux → twist kick
└── vqc_coupling.py  # z-resolved VQC ↔ lattice coupling (v0.2)
```

## Shared constants (`configs/default.yaml`)

| Symbol | Value | Source |
|--------|-------|--------|
| W_g | 350/π ≈ 111.408 | toe / mystery / vqc stable font |
| κ_doc | 0.85 | toe gauge damping |
| κ_sim | ≈ 0.89 | mystery survival optimum @ λt=2 |
| braiding | ≈ 0.814 | toe reproduction lock |
| γ₁ (BMGL) | 1.5 | vqc_sims_public p-wave inhibition |

## Roadmap

1. **v0.1** — analytic OAM packet + flywheel kicks + PDE relaxation ✅
2. **v0.2** — VQC vectorized multi-ℓ propagation; Hopf fiber flux deposition ✅
3. **v0.3** — mystery probes: golden-angle OAM quantization, κ* residual alignment
4. **v0.4** — Streamlit dashboard (lattice + helical beam overlay)

## Related repos

- [vqc_proto](https://github.com/kinaar8340/vqc_proto) — Orbital Braille embodiment (SLM-ready)
- [hfb](https://github.com/kinaar8340/hfb) — Hopf flux bubbles / topological defects

## License

Research code — CC-BY-NC-SA-4.0 (aligned with mystery / vqc_sims_public). TOE components remain MIT where forked.