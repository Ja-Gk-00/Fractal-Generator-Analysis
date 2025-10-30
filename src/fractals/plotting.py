from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Iterable
from .levy_c import LevyCCurve
import math

__all__ = ["plot_polyline", "plot_scatter", "plot_iterations_grid", "plot_ifs_progression", "lsystem_interpret_turtle"]


def _clean_axes(ax):
    ax.set_aspect("equal", adjustable="box")  # proporcje 1:1
    ax.axis("off")


def plot_polyline(points: Iterable[tuple[float, float]], linewidth: float = 0.8, title: str = ""):
    """Wizualizuj krzywa z listy punktow."""
    xs, ys = zip(*points)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, linewidth=linewidth)
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_scatter(points: Iterable[tuple[float, float]], s: float = 0.2, title: str = ""):
    """Wykres dla punktow."""
    xs, ys = zip(*points)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=s)
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_iterations_grid(
    iter_start: int = 8,
    iter_stop: int = 12,
    angle_deg: float = 45.0,
    max_per_row: int = 6,
    cell_size: float = 3.5,
    linewidth: float = 0.8,
):
    """Wizualizacja L-systemu po iteracjach."""
    total = max(0, iter_stop - iter_start + 1)
    if total == 0:
        raise ValueError("iter_stop must be >= iter_start")

    cols = min(total, max(1, max_per_row))
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cell_size * cols, cell_size * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.ravel().tolist()

    for idx, it in enumerate(range(iter_start, iter_stop + 1)):
        lc = LevyCCurve(method="lsystem", iterations=it, angle_deg=angle_deg)  # generuj dla danej iteracji
        pts = lc.generate()
        xs, ys = zip(*pts)
        ax = axes_list[idx]
        ax.plot(xs, ys, linewidth=linewidth)
        ax.set_title(f"it={it}")
        _clean_axes(ax)

    for j in range(total, len(axes_list)):
        axes_list[j].axis("off")

    fig.suptitle(f"Levy C-curve iterations (angle={angle_deg}°)")
    fig.tight_layout()
    return fig


def plot_ifs_progression(
    point_counts=(1_000, 5_000, 20_000, 100_000),
    max_per_row: int = 6,
    s: float = 0.2,
    cell_size: float = 3.5,
    title_prefix: str = "Levy C-curve (IFS)",
):
    """Zwizualizuj IFS dla rosnącej liczby punktow."""
    counts = list(point_counts)
    total = len(counts)
    if total == 0:
        raise ValueError("point_counts must contain at least one value")

    cols = min(total, max(1, max_per_row))
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cell_size * cols, cell_size * rows))
    axes_list = [axes] if rows == 1 and cols == 1 else axes.ravel().tolist()

    for idx, n in enumerate(counts):
        lc = LevyCCurve(method="ifs", n_points=int(n))  # probkuj IFS
        pts = lc.generate()
        xs, ys = zip(*pts)
        ax = axes_list[idx]
        ax.scatter(xs, ys, s=s)
        ax.set_title(f"{title_prefix} — n={n:,}")
        ax.set_aspect("equal")
        ax.axis("off")

    for j in range(total, len(axes_list)):
        axes_list[j].axis("off")

    fig.tight_layout()
    return fig


def lsystem_interpret_turtle(
    instructions: str,
    angle_deg: float = 45.0,
    step: float = 1.0,
    start: tuple[float, float] = (0.0, 0.0),
    heading_deg: float = 0.0,
):
    """Zwraca liste punktow (x, y)."""
    x, y = start
    heading = math.radians(heading_deg)
    ang = math.radians(angle_deg)
    pts = [(x, y)]
    stack = []

    for ch in instructions:
        if ch in ("F", "G"):  # rysuj do przodu
            x += step * math.cos(heading)
            y += step * math.sin(heading)
            pts.append((x, y))
        elif ch == "f":  # ruch bez rysowania
            x += step * math.cos(heading)
            y += step * math.sin(heading)
        elif ch == "+":  # obrot w prawo
            heading -= ang
        elif ch == "-":  # obrot w lewo
            heading += ang
        elif ch == "[":  # zapisz stan
            stack.append((x, y, heading))
        elif ch == "]" and stack:  # przywroc stan
            x, y, heading = stack.pop()
            pts.append((x, y))

    return pts
