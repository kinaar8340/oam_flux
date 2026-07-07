# OAM–Flux

[![Repository](https://img.shields.io/badge/GitHub-oam__flux-blue)](https://github.com/kinaar8340/oam_flux)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-live%20demo-yellow)](https://huggingface.co/spaces/kinaar111/oam_flux)

**Orbital Angular Momentum coupled to gauged Hopf lattice flux flywheels.**

Synthesis layer unifying three existing simulation stacks:

| Repo | Role in `oam_flux` |
|------|-------------------|
| [toe](https://github.com/kinaar8340/toe) | Gauged Hopf lattice, twist PDE, flux flywheels, κ / W_g locks |
| [vqc_sims_public](https://github.com/kinaar8340/vqc_sims_public) | Helical OAM propagation, LG modes, OAM-flux qubit dynamics |
| [mystery](https://github.com/kinaar8340/mystery) | Emergent probes: residual R, golden angle, e⁻² survival @ λt=2 |

## Try the live demo (zero install)

**[🤗 Hugging Face Space — OAM–Flux](https://huggingface.co/spaces/kinaar111/oam_flux)**

| Tab | What it runs |
|-----|----------------|
| **VQC Coupling** | Multi-ℓ propagation heatmap + flux kick on lattice fibers |
| **Emergence** | κ/ℓ survival sweeps vs R, e⁻², golden-angle @ λt=2 |
| **Analytic** | Fast v0.1 OAM packet baseline |

Deploy / update Space:

```bash
bash scripts/sync_hf_space.sh
bash scripts/deploy_hf_space.sh
```

---

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

# v0.3 — mystery emergence probes (κ sweep, ℓ sweep, analog matching)
.venv/bin/python scripts/run_emergence_probes.py

# Tests
.venv/bin/pytest -q
```

Outputs: `outputs/coupling_demo/` (v0.1) · `outputs/vqc_coupling_demo/` (v0.2) · `outputs/emergence_probes/` (v0.3).

## Module map

```
src/oam_flux/
├── constants.py     # W_g, κ_doc, κ_sim, κ*, residual R
├── lattice.py       # Twist PDE + helical IC (from toe pde_relaxation)
├── photon.py        # Analytic LG OAM packets (v0.1)
├── coupling.py      # Analytic momentum ledger + flywheel kicks (v0.1)
├── vqc_photonics.py # VQC vectorized multi-ℓ propagation (v0.2)
├── flux_deposit.py  # Hopf fiber coords + flux → twist kick
├── vqc_coupling.py  # z-resolved VQC ↔ lattice coupling (v0.2)
└── emergence.py     # Mystery analog probes @ λt=2 (v0.3)
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
3. **v0.3** — mystery emergence probes: κ/ℓ sweeps, R / e⁻² / golden analog matching ✅
4. **v0.4** — HF Space: helix 3D, mini-Eddington, cross-tab sync ✅

## Related repos

- [vqc_proto](https://github.com/kinaar8340/vqc_proto) — Orbital Braille embodiment (SLM-ready)
- [hfb](https://github.com/kinaar8340/hfb) — Hopf flux bubbles / topological defects

## License

Research code — CC-BY-NC-SA-4.0 (aligned with mystery / vqc_sims_public). TOE components remain MIT where forked.