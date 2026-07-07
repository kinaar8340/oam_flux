from oam_flux.helix_viz import build_helix_geometry, hopf_fiber_curve, render_helix_frame


def test_hopf_fiber_length():
    x, y, z = hopf_fiber_curve(100)
    assert len(x) == 100
    assert len(z) == 100


def test_helix_geometry_shapes():
    geom = build_helix_geometry(active_ell=3, l_inner=5, num_points=200, turns=4)
    assert len(geom.x_outer) == 200
    assert len(geom.x_hopf) == 200
    assert geom.active_ell == 3


def test_render_frame():
    geom = build_helix_geometry(active_ell=-2, num_points=100, turns=3)
    fig = render_helix_frame(geom)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)