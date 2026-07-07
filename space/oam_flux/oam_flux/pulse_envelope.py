"""Temporal pump envelopes for pulse-train coupling."""

from __future__ import annotations

import math

PULSE_SHAPES = ("square", "gaussian")
DEFAULT_PULSE_SHAPE = "square"
ENVELOPE_MIN = 1e-3


def normalize_pulse_shape(shape: str) -> str:
    key = str(shape).strip().lower()
    if key not in PULSE_SHAPES:
        raise ValueError(f"Unknown pulse shape {shape!r}; choose from {PULSE_SHAPES}")
    return key


def build_pump_envelope(shape: str, n_steps: int, *, sigma_fraction: float = 0.22) -> list[float]:
    """
    Build a per-pump-step intensity envelope (peak = 1).

    Square: flat top over the pump window.
    Gaussian: centered on the pump window with σ ≈ sigma_fraction · n_steps.
    """
    n = max(int(n_steps), 0)
    if n == 0:
        return []
    key = normalize_pulse_shape(shape)
    if key == "square":
        return [1.0] * n
    center = (n - 1) / 2.0
    sigma = max(n * float(sigma_fraction), 0.5)
    raw = [math.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(n)]
    peak = max(raw)
    return [v / peak for v in raw] if peak > 0.0 else [0.0] * n