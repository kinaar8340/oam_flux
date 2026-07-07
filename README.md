# OAM–Flux

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

# First integration probe
.venv/bin/python scripts/run_coupling_demo.py

# Tests
.venv/bin/pytest -q
```

Outputs land in `outputs/coupling_demo/`.

## Module map

```
src/oam_flux/
├── constants.py   # W_g, κ_doc, κ_sim, κ*, residual R
├── lattice.py     # Twist PDE + helical IC (from toe pde_relaxation)
├── photon.py      # LG OAM packets (from vqc photonics)
└── coupling.py    # Momentum ledger + flywheel kicks
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

1. **v0.1** — coupled kick + PDE relaxation (this repo)
2. **v0.2** — import `vqc_sims_public` vectorized propagation; deposit flux on lattice fibers
3. **v0.3** — mystery probes: golden-angle OAM quantization, κ* residual alignment
4. **v0.4** — Streamlit dashboard (lattice + helical beam overlay)

## Related repos

- [vqc_proto](https://github.com/kinaar8340/vqc_proto) — Orbital Braille embodiment (SLM-ready)
- [hfb](https://github.com/kinaar8340/hfb) — Hopf flux bubbles / topological defects

## License

Research code — CC-BY-NC-SA-4.0 (aligned with mystery / vqc_sims_public). TOE components remain MIT where forked.