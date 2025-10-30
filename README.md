# Fractal-Generator-Analysis
A cool project for the WUT course on fractals ;).

# Lévy C-Curve — Python project

This repo implements the Lévy C-curve in two ways: **L-system** and **IFS (chaos game)**, plus
plotting utilities and simple fractal metrics (box-counting/Minkowski estimate).

## Quickstart (with uv)

```bash
# from repo root
uv sync # create venv and install deps from pyproject
uv run levy-fractal --help

# Examples
uv run levy-fractal draw lsystem --iter 14 --angle 45 --outfile levy_lsystem.png
uv run levy-fractal draw ifs --n-points 80000 --outfile levy_ifs.png
uv run levy-fractal compare --iter 12 --angle 45 --outfile compare.png
uv run levy-fractal dimension --iter 13 --angle 45 --outfile dim.png