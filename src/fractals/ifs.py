from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import normalize_probs, seeded_rng


@dataclass
class Affine2D:
    A: np.ndarray  # macierz 2x2
    b: np.ndarray  # wektor (2,)


class IFS:
    """IFS 2D z probkowaniem"""

    def __init__(self, transforms: list[Affine2D], probs: list[float] | None = None):
        self.transforms = transforms
        self.probs = normalize_probs(probs if probs is not None else [1] * len(transforms))

    def sample(
        self,
        n_points: int = 50_000,
        discard: int = 100,
        seed: int | None = 1337,
        x0: np.ndarray | None = None,
    ) -> list[tuple[float, float]]:
        rng = seeded_rng(seed)  # losowe wybory transformacji
        x = np.array([0.0, 0.0], dtype=float) if x0 is None else np.asarray(x0, dtype=float)
        pts: list[tuple[float, float]] = []

        for i in range(n_points + discard):  # zbieranie punktow
            k = rng.choice(len(self.transforms), p=self.probs)
            T = self.transforms[k]
            x = T.A @ x + T.b
            if i >= discard:
                pts.append((float(x[0]), float(x[1])))
        return pts
