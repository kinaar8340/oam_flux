import pytest

from oam_flux.helix_viz import (
    build_helix_geometry,
    figures_to_pil,
    frames_to_mp4,
    helix_animation_frames,
    hopf_fiber_curve,
    hopf_tangent_at_lattice_site,
    hopf_tangent_at_t,
    render_helix_frame,
)


def test_hopf_fiber_length():
    x, y, z = hopf_fiber_curve(100)
    assert len(x) == 100
    assert len(z) == 100


def test_hopf_tangent_unit_vector():
    tan = hopf_tangent_at_t(1.2)
    import numpy as np
    assert float(np.linalg.norm(tan)) == pytest.approx(1.0, rel=1e-6)
    tan_site = hopf_tangent_at_lattice_site((5, 5, 5), nx=20)
    assert float(np.linalg.norm(tan_site)) == pytest.approx(1.0, rel=1e-6)


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


def test_animation_mp4_export(tmp_path):
    import shutil

    if shutil.which("ffmpeg") is None:
        return
    geom = build_helix_geometry(active_ell=2, num_points=80, turns=3)
    pil_frames = figures_to_pil(helix_animation_frames(geom, n_frames=4))
    out = tmp_path / "helix.mp4"
    result = frames_to_mp4(pil_frames, str(out), fps=8.0)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0