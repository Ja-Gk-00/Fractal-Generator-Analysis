from __future__ import annotations
import numpy as np
from .utils import rot2d
from .ifs import IFS, Affine2D

# --- Variant 1: α-Lévy curve (generalized angle) ------------------------------


def levy_ifs_generalized(angle_deg: float = 45.0) -> IFS:
    """
    Return an IFS for a generalized Lévy-like curve using ±angle and scale 1/sqrt(2).
    For angle=45° it reduces to the classical Lévy C-curve.
    """
    theta = np.deg2rad(angle_deg)
    s = 1.0 / np.sqrt(2.0)
    R1 = s * rot2d(+theta)
    R2 = s * rot2d(-theta)
    # Translation chosen to keep endpoints roughly within [0,1]^2
    b1 = np.array([0.0, 0.0])
    b2 = np.array([0.5, 0.5])
    return IFS([Affine2D(R1, b1), Affine2D(R2, b2)], probs=[0.5, 0.5])


# --- Variant 2: stochastic L-system (probabilistic rule) ----------------------


def levy_lsystem_stochastic(p: float = 0.5) -> tuple[str, dict[str, str]]:
    """Return axiom and a pair of alternative rules applied with prob p/(1-p).

    Use with an L-system engine that supports random choice per symbol (implemented
    by the notebook demo or an extended interpreter). Here we only provide rules.
    """
    # Two slight perturbations of the classic rule
    rules_a = {"F": "+F--F+"}
    rules_b = {"F": "+F-+F-"}
    # The caller decides how to sample A/B per expansion step per symbol.
    return "F", {"A": "+F--F+", "B": "+F-+F-"}
