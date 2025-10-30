from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass
class BoxCountResult:
    deltas: np.ndarray
    counts: np.ndarray
    slope: float
    intercept: float
    r_value: float


DefPoints = Iterable[tuple[float, float]]


def _occupied_boxes(points: DefPoints, delta: float) -> set[tuple[int, int]]:
    pts = np.asarray(list(points), dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / delta).astype(np.int64)
    iy = np.floor((y - ymin) / delta).astype(np.int64)
    return set(zip(ix.tolist(), iy.tolist()))


def box_count(points: DefPoints, delta: float) -> int:
    return len(_occupied_boxes(points, delta))


def estimate_box_dimension(
    points: DefPoints,
    deltas: Sequence[float],
    title_prefix: str = "Box-counting",
) -> tuple[plt.Figure, BoxCountResult]:
    deltas = np.asarray(sorted(deltas), dtype=float)
    counts = np.array([box_count(points, d) for d in deltas], dtype=float)

    x = np.log(1.0 / deltas)
    y = np.log(counts)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # simple R-value (correlation) for line fit
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_value = float(np.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot > 0 else 1.0

    # plot
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_hat, label=f"fit: D≈{slope:.4f}, R≈{r_value:.3f}")
    ax.set_xlabel("log(1/δ)")
    ax.set_ylabel("log N_δ")
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
