from __future__ import annotations
import numpy as np


__all__ = ["rot2d", "normalize_probs", "seeded_rng"]


def rot2d(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def normalize_probs(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        raise ValueError("Probabilities must sum to > 0")
    return p / s


def seeded_rng(seed: int | None):
    return np.random.default_rng(seed)
