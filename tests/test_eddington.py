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
    assert "total_outward_flux" in result.to_dict()