from __future__ import annotations
import numpy as np
from .utils import rot2d
from .ifs import IFS, Affine2D

# Wariant I, uogolniony kat
def levy_ifs_generalized(angle_deg: float = 45.0) -> IFS:
    """IFS krzywej typu Levy dla ±angle i skali 1/sqrt(2)."""
    theta = np.deg2rad(angle_deg)
    s = 1.0 / np.sqrt(2.0)
    R1 = s * rot2d(+theta)
    R2 = s * rot2d(-theta)
    # translacje trzymaja ksztalt w oknie [0,1]^2
    b1 = np.array([0.0, 0.0])
    b2 = np.array([0.5, 0.5])
    return IFS([Affine2D(R1, b1), Affine2D(R2, b2)], probs=[0.5, 0.5])


# Wariant II, stochastyczny wybor zasad dla L-systemu

def levy_lsystem_stochastic(p: float = 0.5) -> tuple[str, dict[str, str]]:
    """Zwraca aksjomat i alternatywne reguly stosowane z prawd. p/(1-p)."""
    # wybor A/B realizuje wywolujacy (tu tylko definicje)
    return "F", {"A": "+F--F+", "B": "+F-+F-"}

def lsystem_expand_stochastic(
    axiom: str = "F",
    iterations: int = 12,
    p: float = 0.25,
    rule_classic: str = "+F--F+",
    rule_variant: str = "+F-+F-",
    seed: int | None = 0,
) -> str:
    """Stochastyczny wybor: kazde 'F' -> wariant z p lub originalny z 1-p."""
    rng = np.random.default_rng(seed)
    s = axiom
    for _ in range(iterations):
        out = []
        for ch in s:
            if ch == "F":
                out.append(rule_variant if rng.random() < p else rule_classic)
            else:
                out.append(ch)
        s = "".join(out)
    return s
