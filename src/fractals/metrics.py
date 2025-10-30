from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class BoxCountResult:
    deltas: np.ndarray
    counts: np.ndarray
    slope: float
    intercept: float
    r_value: float


DefPoints = Iterable[tuple[float, float]]


def _occupied_boxes(points: DefPoints, delta: float) -> set[tuple[int, int]]:
    """Pomocnicza funkcja do lieczenia komorek: zwroc zbior zajetych komorek."""
    pts = np.asarray(list(points), dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / delta).astype(np.int64)
    iy = np.floor((y - ymin) / delta).astype(np.int64)
    return set(zip(ix.tolist(), iy.tolist()))


def box_count(points: DefPoints, delta: float) -> int:
    """Policz liczbe zajetych komorek dla danego delta."""
    return len(_occupied_boxes(points, delta))


def estimate_box_dimension(
    points: DefPoints,
    deltas: Sequence[float],
    title_prefix: str = "Box-counting",
) -> tuple[plt.Figure, BoxCountResult]:
    """Estymacja wymiaru komorek: dopasuj prosta do log N_delta, a log(1/delta)."""
    deltas = np.asarray(sorted(deltas), dtype=float)
    counts = np.array([box_count(points, d) for d in deltas], dtype=float)

    x = np.log(1.0 / deltas)
    y = np.log(counts)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # estymuj prosta w regresji
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_value = float(np.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot > 0 else 1.0

    # wykres
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_hat, label=f"fit: D≈{slope:.4f}, R≈{r_value:.3f}")
    ax.set_xlabel("log(1/delta)")
    ax.set_ylabel("log N_delta")
    ax.set_title(f"{title_prefix}: box-counting dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    res = BoxCountResult(
        deltas=deltas,
        counts=counts,
        slope=float(slope),
        intercept=float(intercept),
        r_value=r_value,
    )
    return fig, res


ArrayLikePoints = Iterable[tuple[float, float]]


def _normalize_to_unit(points: ArrayLikePoints) -> np.ndarray:
    """Normalizacja punktow do [0,1]^2."""
    P = np.asarray(list(points), dtype=float)
    if P.size == 0:
        raise ValueError("points must be non-empty")
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)
    spanx = max(1e-12, xmax - xmin)
    spany = max(1e-12, ymax - ymin)
    P[:, 0] = (P[:, 0] - xmin) / spanx
    P[:, 1] = (P[:, 1] - ymin) / spany
    return P


@dataclass
class CorrelationDimResult:
    radii: np.ndarray
    C: np.ndarray
    slope: float
    intercept: float


def estimate_correlation_dimension(
    points: ArrayLikePoints,
    radii: Sequence[float],
    max_pairs: int = 100_000,
    seed: int = 0,
    title_prefix: str = "Correlation dimension",
) -> tuple[plt.Figure, CorrelationDimResult]:
    """Wymiar korelacyjny (Grassberger-Procaccia): nachylenie log C(r), a log r."""
    rng = np.random.default_rng(seed)
    P = _normalize_to_unit(points)
    n = len(P)
    if n < 2:
        raise ValueError("Need at least 2 points")

    # probkowanie par dla szybkosci
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=num_pairs)
    j_idx = rng.integers(0, n, size=num_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    d = np.linalg.norm(P[i_idx] - P[j_idx], axis=1)

    radii = np.asarray(sorted(radii), dtype=float)
    C = np.array([(d <= r).mean() for r in radii], dtype=float)

    x = np.log(radii)
    y = np.log(np.clip(C, 1e-12, 1.0))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, slope * x + intercept, label=f"fit: D2≈{slope:.4f}")
    ax.set_xlabel("log r")
    ax.set_ylabel("log C(r)")
    ax.set_title(title_prefix)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, CorrelationDimResult(radii=radii, C=C, slope=float(slope), intercept=float(intercept))


@dataclass
class LacunarityResult:
    deltas: np.ndarray
    lacunarity: np.ndarray


def lacunarity_gliding_box(
    points: ArrayLikePoints,
    deltas: Sequence[float],
    title: str = "Lacunarity (gliding box)",
) -> tuple[plt.Figure, LacunarityResult]:
    """Lakunarosc: Lambda = (var/mu^2) + 1."""
    P = _normalize_to_unit(points)
    X, Y = P[:, 0], P[:, 1]
    deltas = np.asarray(sorted(deltas), dtype=float)
    Lambda = []

    for delta in deltas:
        # liczba pudelek na os ~ 1/delta
        nbin = max(1, int(np.round(1.0 / delta)))
        H, _, _ = np.histogram2d(X, Y, bins=nbin, range=[[0, 1], [0, 1]])
        m = H.ravel()
        mu = m.mean()
        var = m.var()
        lam = (var / (mu**2) + 1.0) if mu > 0 else np.nan
        Lambda.append(lam)

    Lambda = np.asarray(Lambda, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(np.log(1.0 / deltas), np.log(Lambda), "o-")
    ax.set_xlabel("log(1/delta)")
    ax.set_ylabel("log Lambda(delta)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, LacunarityResult(deltas=deltas, lacunarity=Lambda)
