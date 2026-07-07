"""Shared invariants bridging toe, mystery, and vqc_sims_public."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

W_G_EXACT = 350.0 / 3.141592653589793
PHI = (1.0 + 5.0**0.5) / 2.0
RESIDUAL_R = PHI**2 + 2.718281828459045**2 - 3.141592653589793**2


@dataclass(frozen=True)
class LatticeConstants:
    W_g: float = W_G_EXACT
    kappa_doc: float = 0.85
    kappa_sim: float = 0.89
    braiding_phase: float = 0.814
    golden_angle_deg: float = 137.50776405003785

    @property
    def kappa_star(self) -> float:
        """Exact κ that nulls B(κ) = π²(e/π − κ) against residual R."""
        import math
        return math.e / math.pi - RESIDUAL_R / math.pi**2

    @property
    def holonomy_gap(self) -> float:
        import math
        return math.e / math.pi - self.kappa_doc


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}