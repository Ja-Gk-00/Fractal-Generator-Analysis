from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .lsystem import LSystem, LSystemSpec
from .ifs import IFS, Affine2D
from .utils import rot2d


@dataclass
class LevyCCurve:
    method: str = "lsystem"  # "lsystem" | "ifs"
    iterations: int = 12
    angle_deg: float = 45.0
    lsystem_rules: dict[str, str] | None = None
    n_points: int = 50_000

    def generate(self) -> list[tuple[float, float]]:
        m = self.method.lower()
        if m == "lsystem":
            return self._generate_lsystem()
        elif m == "ifs":
            return self._generate_ifs()
        else:
            raise ValueError("Unknown method: %r" % self.method)

    # --- L-system ---
    def _generate_lsystem(self) -> list[tuple[float, float]]:
        rules = self.lsystem_rules or {"F": "+F--F+"}
        spec = LSystemSpec(axiom="F", rules=rules, angle_deg=self.angle_deg)
        ls = LSystem(spec, iterations=self.iterations)
        ls.expand()
        return ls.interpret()

    # --- IFS ---
    def _generate_ifs(self) -> list[tuple[float, float]]:
        # Classic Lévy C-curve IFS: two rotations by ±45° and scale 1/sqrt(2)
        theta = np.deg2rad(45.0)
        s = 1.0 / np.sqrt(2.0)
        R1 = s * rot2d(+theta)
        R2 = s * rot2d(-theta)
        b1 = np.array([0.0, 0.0])
        b2 = np.array([0.5, 0.5])
        ifs = IFS(
            [
                Affine2D(R1, b1),
                Affine2D(R2, b2),
            ],
            probs=[0.5, 0.5],
        )
        return ifs.sample(n_points=self.n_points, discard=50, seed=123)
