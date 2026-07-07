import math

from oam_flux.constants import LatticeConstants, RESIDUAL_R, W_G_EXACT


def test_w_g_lock():
    assert abs(W_G_EXACT - 350.0 / math.pi) < 1e-9


def test_kappa_star_near_doc():
    c = LatticeConstants()
    assert abs(c.kappa_star - 0.8513) < 0.001
    assert abs(c.kappa_doc - c.kappa_star) < 0.02


def test_residual_positive():
    assert 0.13 < RESIDUAL_R < 0.14