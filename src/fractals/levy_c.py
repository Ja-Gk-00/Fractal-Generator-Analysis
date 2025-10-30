from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .lsystem import LSystem, LSystemSpec
from .ifs import IFS, Affine2D
from .utils import rot2d


@dataclass
class LevyCCurve:
    """
    Klasa generator dla krzywych Levy'ego C.

    Args:
        method: "lsystem" lub "ifs".
        iterations: Liczba iteracji L-systemu.
        angle_deg: Kat obrotu dla L-systemu.
        lsystem_rules: Slownik regul L-systemu (domyslnie {'F': '+F--F+'}).
        n_points: Liczba punktow dla metody IFS.

    Returns:
        Dla `generate()`: lista punktów (x, y) opisujaca krzywa/fraktal.
    """
    method: str = "lsystem"  # "lsystem" | "ifs"
    iterations: int = 12
    angle_deg: float = 45.0
    lsystem_rules: dict[str, str] | None = None
    n_points: int = 50_000

    def generate(self) -> list[tuple[float, float]]:
        """
        Wygeneruj punkty fraktala zgodnie z wybrana metodą.

        Returns:
            Lista krotek (x, y).
        """
        m = self.method.lower()
        if m == "lsystem":
            return self._generate_lsystem()
        elif m == "ifs":
            return self._generate_ifs()
        else:
            raise ValueError("Unknown method: %r" % self.method)

    # L-system
    def _generate_lsystem(self) -> list[tuple[float, float]]:
        """
        Zbuduj L-system i zinterpretuj go jako krzywa.

        Returns:
            Lista punktow (x, y).
        """
        rules = self.lsystem_rules or {"F": "+F--F+"}
        spec = LSystemSpec(axiom="F", rules=rules, angle_deg=self.angle_deg)
        ls = LSystem(spec, iterations=self.iterations)
        ls.expand()
        return ls.interpret()

    # IFS
    def _generate_ifs(self) -> list[tuple[float, float]]:
        """
        Probkowanie IFS klasycznej krzywej Levy'ego C.

        Returns:
            Lista punktow (x, y).
        """
        theta = np.deg2rad(45.0)
        s = 1.0 / np.sqrt(2.0)
        R1 = s * rot2d(+theta)
        R2 = s * rot2d(-theta)
        b1 = np.array([0.0, 0.0])
        b2 = np.array([0.5, 0.5])
        ifs = IFS
