import math

from oam_flux.emergence import (
    E_INV2,
    GOLDEN_FRACTION,
    RESIDUAL_R,
    EmergenceAnalogs,
    best_analog_match,
    bound_b,
    ell_sweep,
    emergence_report,
    golden_quantized_ells,
    kappa_sweep,
    lambda_t_steps,
    run_vqc_emergence_trial,
)


def test_lambda_t_steps():
    assert lambda_t_steps(0.85, 0.001, 2.0) == 2353


def test_bound_b_kappa_star_nulls_r():
    a = EmergenceAnalogs()
    assert abs(bound_b(a.kappa_star) - RESIDUAL_R) < 1e-10


def test_best_analog_match_finds_e_inv2():
    match = best_analog_match(E_INV2)
    assert match["best_analog"] == "e_inv2"
    assert match["delta_pct"] < 0.01


def test_golden_quantized_ells_nonempty():
    ells = golden_quantized_ells(6)
    assert len(ells) >= 1
    assert all(-6 <= e <= 6 for e in ells)


def test_vqc_emergence_trial_runs():
    trial = run_vqc_emergence_trial(kappa=0.85, ell=2, lambda_t=2.0, pump_fraction=0.5)
    assert trial.n_steps > 0
    assert 0.0 < trial.mean_survival < 2.0


def test_kappa_sweep_produces_rows():
    result = kappa_sweep(n_points=5, ell=2)
    assert len(result.rows) == 5
    best = result.best_kappa_for_analog("mean_survival", "R")
    assert 0.80 <= best["kappa"] <= 0.90


def test_ell_sweep_and_report():
    ell_res = ell_sweep(l_max=4, kappa=0.85)
    kap_res = kappa_sweep(n_points=5, ell=2)
    report = emergence_report(kappa_result=kap_res, ell_result=ell_res)
    assert "comparisons" in report
    assert len(report["golden_quantized_ells"]) >= 1
    assert report["analogs"]["R"] == RESIDUAL_R
    assert abs(report["analogs"]["golden_angle"] - GOLDEN_FRACTION) < 1e-9