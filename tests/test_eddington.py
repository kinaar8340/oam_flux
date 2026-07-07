import numpy as np
import pytest

from oam_flux.eddington import binding_at_site, run_eddington_probe


def test_binding_scales_with_kappa():
    b1 = binding_at_site(1.0, 0.85)
    b2 = binding_at_site(1.0, 0.90)
    assert b2 > b1


def test_eddington_probe_runs():
    result = run_eddington_probe(
        kappa=0.85,
        ell=2,
        n_flywheels=4,
        n_steps=30,
        kick_strength=0.1,
    )
    assert result.n_sites == 4
    assert len(result.history) == 30
    meta = result.to_dict()
    assert "total_outward_flux" in meta
    assert "wind_z" in meta


def test_eddington_kicks_scale_with_energy():
    low = run_eddington_probe(
        kappa=0.80, ell=4, n_flywheels=6, n_steps=60, kick_strength=0.12, energy_scale=1.0,
    )
    high = run_eddington_probe(
        kappa=0.80, ell=4, n_flywheels=6, n_steps=60, kick_strength=0.12, energy_scale=2.0,
    )
    assert high.effective_kick == pytest.approx(2.0 * low.effective_kick)
    assert high.total_outward_flux >= low.total_outward_flux


def test_eddington_wind_along_hopf_fiber():
    result = run_eddington_probe(
        kappa=0.80,
        ell=4,
        n_flywheels=6,
        n_steps=60,
        kick_strength=0.15,
    )
    if result.total_outward_flux > 0:
        w = result.wind_vector
        assert float(w[2]) > 0.0
        assert float(np.linalg.norm(w)) > 0.0