"""Helix-within-helix + Hopf fiber 3D visualization (from VQC schematic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class HelixGeometry:
    x_outer: np.ndarray
    y_outer: np.ndarray
    z_outer: np.ndarray
    x_inner: np.ndarray
    y_inner: np.ndarray
    z_inner: np.ndarray
    x_hopf: np.ndarray
    y_hopf: np.ndarray
    z_hopf: np.ndarray
    x_knot: np.ndarray | None
    y_knot: np.ndarray | None
    z_knot: np.ndarray | None
    theta: np.ndarray
    l_outer: int
    l_inner: int
    active_ell: int


def _stevedore_knot(num_points: int, scale: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * np.pi, num_points)
    x = scale * np.cos(t) * (2 + np.cos(2 * t)) / 2
    y = scale * np.sin(t) * (2 + np.cos(2 * t)) / 2
    z = scale * np.sin(3 * t) / 2
    x += scale * 0.5 * np.sin(4 * t)
    y += scale * 0.3 * np.cos(3 * t)
    return x, y, z


def hopf_fiber_curve(
    num_points: int = 600,
    *,
    turns: float = 3.0,
    radius: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hopf S³→S² fiber projected to 3D propagation coordinates."""
    t = np.linspace(0, turns * 2 * np.pi, num_points)
    # Standard Hopf parameterization on fiber
    eta = t
    phi = t * 0.5
    x = radius * np.cos(phi) * np.cos(eta * 0.35)
    y = radius * np.sin(phi) * np.cos(eta * 0.35)
    z = t / (2 * np.pi)
    return x, y, z


def hopf_tangent_at_t(
    t: float,
    *,
    radius: float = 0.55,
) -> np.ndarray:
    """Unit tangent to the Hopf fiber at parameter t."""
    phi = t * 0.5
    eta = t
    dx = (
        -radius * np.sin(phi) * np.cos(eta * 0.35) * 0.5
        - radius * np.cos(phi) * np.sin(eta * 0.35) * 0.35
    )
    dy = (
        radius * np.cos(phi) * np.cos(eta * 0.35) * 0.5
        - radius * np.sin(phi) * np.sin(eta * 0.35) * 0.35
    )
    dz = 1.0 / (2.0 * np.pi)
    vec = np.array([dx, dy, dz], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def hopf_tangent_at_lattice_site(
    index: tuple[int, int, int],
    *,
    nx: int,
    turns: float = 3.0,
) -> np.ndarray:
    """Hopf fiber tangent at a lattice flywheel voxel (maps site → fiber parameter)."""
    i, j, k = index
    denom = max(nx - 1, 1)
    t_frac = float(i + j + k) / (3.0 * denom)
    t = t_frac * turns * 2.0 * np.pi
    return hopf_tangent_at_t(t)


def build_helix_geometry(
    *,
    l_outer: int = 3,
    l_inner: int | None = None,
    active_ell: int = 3,
    num_points: int = 800,
    turns: int = 6,
    knot_mod: bool = True,
    scale: float = 0.8,
) -> HelixGeometry:
    """Dual counter-propagating OAM helices + Hopf fiber backbone."""
    l_inner = l_inner if l_inner is not None else abs(active_ell)
    l_outer = l_outer if l_outer != 0 else max(1, abs(active_ell))

    theta = np.linspace(0, turns * 2 * np.pi, num_points)
    period = 2 * np.pi

    x_outer = np.cos(l_outer * theta)
    y_outer = np.sin(l_outer * theta)
    z_outer = theta / (2 * np.pi)

    sign = -1 if active_ell < 0 else 1
    inner_l = sign * abs(l_inner)
    x_inner = 0.4 * np.cos(inner_l * theta)
    y_inner = 0.4 * np.sin(inner_l * theta)
    z_inner = z_outer.copy()

    x_k, y_k, z_k = _stevedore_knot(num_points, scale=scale)
    x_knot = y_knot = z_knot = None

    if knot_mod:
        knot_t = np.linspace(0, 2 * np.pi, num_points)
        knot_pos = np.column_stack([x_k, y_k, z_k])
        eps = 1e-9
        knot_t_ext = np.hstack([knot_t - period - eps, knot_t, knot_t + period + eps])
        knot_pos_ext = np.vstack([knot_pos, knot_pos, knot_pos])
        knot_interp = np.zeros((len(theta), 3))
        for dim in range(3):
            f = interp1d(
                knot_t_ext,
                knot_pos_ext[:, dim],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            knot_interp[:, dim] = f(theta % period)
        mod_r = 0.14 * np.abs(knot_interp[:, 2])
        mod_phase = 0.01 * np.linspace(0, 1, num_points)
        x_inner += mod_r * np.cos(inner_l * theta + mod_phase)
        y_inner += mod_r * np.sin(inner_l * theta + mod_phase)
        z_inner += 0.07 * knot_interp[:, 2]
        x_knot, y_knot, z_knot = knot_interp[:, 0], knot_interp[:, 1], knot_interp[:, 2]

    x_hopf, y_hopf, z_hopf = hopf_fiber_curve(num_points, turns=float(turns))

    return HelixGeometry(
        x_outer=x_outer,
        y_outer=y_outer,
        z_outer=z_outer,
        x_inner=x_inner,
        y_inner=y_inner,
        z_inner=z_inner,
        x_hopf=x_hopf,
        y_hopf=y_hopf,
        z_hopf=z_hopf,
        x_knot=x_knot,
        y_knot=y_knot,
        z_knot=z_knot,
        theta=theta,
        l_outer=l_outer,
        l_inner=abs(l_inner),
        active_ell=active_ell,
    )


def render_helix_frame(
    geom: HelixGeometry,
    *,
    azim: float = 52.0,
    elev: float = 20.0,
    title: str | None = None,
):
    """Render single 3D matplotlib frame."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(geom.x_outer, geom.y_outer, geom.z_outer, color="#457b9d", lw=2.5,
            label=f"Outer ℓ=+{geom.l_outer}", alpha=0.95)
    inner_sign = "-" if geom.active_ell < 0 else "+"
    ax.plot(geom.x_inner, geom.y_inner, geom.z_inner, color="#e76f51", lw=2.5,
            label=f"Inner ℓ={inner_sign}{geom.l_inner}", alpha=0.92)
    ax.plot(geom.x_hopf, geom.y_hopf, geom.z_hopf, color="#2a9d8f", lw=2.0, ls="--",
            label="Hopf fiber", alpha=0.85)
    if geom.x_knot is not None:
        ax.plot(geom.x_knot[::8], geom.y_knot[::8], geom.z_knot[::8],
                color="#6a4c93", lw=1.5, ls=":", label="8₃ knot backbone", alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (propagation)")
    ax.set_title(title or "Helix-within-helix + Hopf fiber")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def helix_animation_frames(
    geom: HelixGeometry,
    *,
    n_frames: int = 36,
    elev: float = 20.0,
) -> list:
    """Generate rotated 3D frames for GIF export."""
    frames = []
    for i in range(n_frames):
        azim = 52.0 + (360.0 * i / n_frames)
        fig = render_helix_frame(geom, azim=azim, elev=elev)
        frames.append(fig)
    return frames


def figures_to_pil(figures: list, *, dpi: int = 100):
    """Rasterize matplotlib figures to RGB PIL images."""
    from io import BytesIO

    from PIL import Image

    images: list[Image.Image] = []
    for fig in figures:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    return images


def frames_to_gif(images: list, path: str, *, duration_ms: int = 80) -> str:
    """Save PIL RGB frames as animated GIF via Pillow."""
    from PIL import Image

    if not images:
        raise ValueError("No frames to save")
    palette_frames = [im.convert("P", palette=Image.ADAPTIVE) for im in images]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return path


def frames_to_mp4(images: list, path: str, *, fps: float = 11.0) -> str | None:
    """Encode PIL RGB frames to H.264 MP4 via ffmpeg (for gr.Video pause/play)."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if not images or shutil.which("ffmpeg") is None:
        return None

    out_path = Path(path)
    with tempfile.TemporaryDirectory(prefix="oam_helix_mp4_") as tmp:
        tmp_path = Path(tmp)
        for i, frame in enumerate(images):
            frame.save(tmp_path / f"frame_{i:03d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            str(tmp_path / "frame_%03d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "cfr",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    return str(out_path)