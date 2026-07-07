"""Gauged Hopf lattice twist PDE — adapted from toe/scripts/pde_relaxation.py."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def helical_seed(nx: int, *, pitch: float = 0.35, amplitude: float = 1.2) -> np.ndarray:
    """Two-gyro helical IC retaining σ > 0 (mystery structured-IC class)."""
    coords = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return amplitude * (0.5 + 0.5 * np.sin(pitch * (x + 2 * y - z)))


@dataclass
class TwistLattice:
    nx: int = 24
    dt: float = 0.001
    D: float = 0.05
    kappa: float = 0.85
    delta_omega: float = 0.002
    theta_crit: float = 5.8
    theta: np.ndarray = field(init=False)
    theta_initial: np.ndarray = field(init=False)
    momentum_ledger: float = 0.0
    recovery_rate: float = 0.04

    def __post_init__(self) -> None:
        self.theta = helical_seed(self.nx)
        self.theta_initial = self.theta.copy()

    @property
    def mean_twist(self) -> float:
        return float(self.theta.mean())

    @property
    def twist_variance(self) -> float:
        return float(self.theta.var())

    def _laplacian(self) -> np.ndarray:
        t = self.theta
        return (
            np.roll(t, 1, 0) + np.roll(t, -1, 0)
            + np.roll(t, 1, 1) + np.roll(t, -1, 1)
            + np.roll(t, 1, 2) + np.roll(t, -1, 2)
            - 6 * t
        ) / (1.0 / self.nx) ** 2

    def _cot_term(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_sq = (
                np.gradient(self.theta, axis=0) ** 2
                + np.gradient(self.theta, axis=1) ** 2
                + np.gradient(self.theta, axis=2) ** 2
            )
            return (self.D / 2.0) * np.cos(self.theta / 2.0) / np.sin(self.theta / 2.0) * grad_sq

    def apply_kick(self, kick: np.ndarray, *, photon_momentum: float) -> None:
        """Discrete momentum handoff: photon OAM flux → local twist increment."""
        self.theta = np.clip(self.theta + kick, 0.01, 2 * np.pi - 0.01)
        self.momentum_ledger -= photon_momentum

    def relax_step(
        self,
        *,
        external_torque: np.ndarray | None = None,
        pump_active: bool = True,
    ) -> float:
        """Single PDE step: ∂θ/∂t = DΔθ + cot + Δω − κ⟨θ⟩ + burst + external."""
        lap = self._laplacian()
        cot = self._cot_term()
        gauge = -self.kappa * self.mean_twist
        burst = np.where(
            self.theta > self.theta_crit,
            -50.0 * (self.theta - self.theta_crit),
            0.0,
        )
        rhs = self.D * lap + cot + self.delta_omega + gauge + burst
        if external_torque is not None:
            rhs = rhs + external_torque
        self.theta = np.clip(self.theta + self.dt * rhs, 0.01, 2 * np.pi - 0.01)
        if not pump_active:
            self.theta = np.clip(
                self.theta + self.recovery_rate * (self.theta_initial - self.theta),
                0.01,
                2 * np.pi - 0.01,
            )
        return self.mean_twist

    def twist_load_vs_initial(self) -> float:
        """Mean |θ − θ₀| — drops during lattice recovery."""
        return float(np.mean(np.abs(self.theta - self.theta_initial)))

    def flywheel_indices(self, n_sites: int) -> list[tuple[int, int, int]]:
        """Place resonators on lattice diagonal (flux-flywheel anchors)."""
        stride = max(1, self.nx // (n_sites + 1))
        return [(stride * (i + 1), stride * (i + 1), stride * (i + 1)) for i in range(n_sites)]