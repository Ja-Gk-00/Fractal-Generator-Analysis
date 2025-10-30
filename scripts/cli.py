from __future__ import annotations
import typer
from typing import Optional
from pathlib import Path


from fractals.levy_c import LevyCCurve
from fractals.plotting import plot_polyline, plot_scatter, plot_iterations_grid
from fractals.metrics import estimate_box_dimension


app = typer.Typer(help="Lévy C-curve: L-system + IFS + analysis")


def _save(fig, outfile: Optional[str]):
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight", dpi=300)


@app.command()
def draw(
    method: typer.Argument(str, help="lsystem | ifs"),
    iter: int = typer.Option(12, help="L-system iterations"),
    angle: float = typer.Option(45.0, help="L-system / variant turn angle in degrees"),
    n_points: int = typer.Option(80_000, help="IFS points (when method=ifs)"),
    linewidth: float = typer.Option(0.8, help="Polyline linewidth (lsystem)"),
    outfile: Optional[str] = typer.Option(None, help="Path to save figure"),
):
    """Draw Lévy C-curve using the chosen method."""
    method = method.lower()
    lc = LevyCCurve(method=method, iterations=iter, angle_deg=angle, n_points=n_points)
    pts = lc.generate()

    if method == "ifs":
        fig = plot_scatter(pts, title=f"Lévy C-curve (IFS)")
    else:
        fig = plot_polyline(
            pts, linewidth=linewidth, title=f"Lévy C-curve (L-system, angle={angle}°, it={iter})"
        )
        _save(fig, outfile)


@app.command()
def compare(
    iter: int = typer.Option(12, help="L-system iterations for the grid"),
    start: int = typer.Option(8, help="Start iteration"),
    stop: int = typer.Option(12, help="Stop iteration (inclusive)"),
    angle: float = typer.Option(45.0, help="Angle for L-system"),
    outfile: Optional[str] = typer.Option(None, help="Path to save figure"),
):
    """Show a grid of iterations for the classic Lévy C-curve."""
    fig = plot_iterations_grid(iter_start=start, iter_stop=stop, angle_deg=angle)
    _save(fig, outfile)


@app.command()
def dimension(
    iter: int = typer.Option(13, help="L-system iterations for point set"),
    angle: float = typer.Option(45.0, help="Angle for L-system"),
    outfile: Optional[str] = typer.Option(None, help="Path to save regression plot"),
):
    """Estimate box-counting (Minkowski) dimension for a Lévy C-curve polyline."""
    lc = LevyCCurve(method="lsystem", iterations=iter, angle_deg=angle)
    pts = lc.generate()
    deltas = [2 ** (-k) for k in range(3, 9)]  # a simple dyadic range
    fig, _ = estimate_box_dimension(pts, deltas=deltas, title_prefix="Lévy C-curve")
    _save(fig, outfile)


if __name__ == "__main__":
    app()
