from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class LSystemSpec:
    axiom: str
    rules: dict[str, str]
    angle_deg: float = 45.0


class LSystem:
    """Ogolny L-system 2D."""

    def __init__(self, spec: LSystemSpec, iterations: int = 12):
        self.spec = spec
        self.iterations = iterations
        self.expanded: str | None = None

    def expand(self) -> str:
        """Zwraca instrukcje po N iteracjach."""
        s = self.spec.axiom
        for _ in range(self.iterations):
            s = "".join(self.spec.rules.get(ch, ch) for ch in s)
        self.expanded = s
        return s

    def interpret(self, step: float | None = None) -> list[tuple[float, float]]:
        """Konwertuje instrukcje na punkty (x, y)."""
        if self.expanded is None:
            self.expand()
        instructions = self.expanded
        angle = math.radians(self.spec.angle_deg)
        if step is None:
            step = 1.0 / (2 ** (self.iterations / 2))

        x, y, heading = 0.0, 0.0, 0.0
        pts = [(x, y)]
        stack: list[tuple[float, float, float]] = []

        for ch in instructions:
            if ch in ("F", "G"):  # rysuj do przodu
                x += step * math.cos(heading)
                y += step * math.sin(heading)
                pts.append((x, y))
            elif ch == "f":  # ruch bez rysowania
                x += step * math.cos(heading)
                y += step * math.sin(heading)
            elif ch == "+":  # obrot w prawo
                heading -= angle
            elif ch == "-":  # obrot w lewo
                heading += angle
            elif ch == "[":  # zapisz stan
                stack.append((x, y, heading))
            elif ch == "]":  # przywroc stan
                x, y, heading = stack.pop()
                pts.append((x, y))
        return pts
