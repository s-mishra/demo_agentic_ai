# CLAUDE.md

Project conventions for dengue surveillance analysis.

## Environment

- Use `uv run` for all Python execution (creates fresh environment from `pyproject.toml`)
- Run commands from project root

## Modeling

- Use **Bambi** for Bayesian modeling (not raw PyMC)
- Rt prior: `LogNormal(0, 0.5)`
- Serial interval: `Gamma(mean=14.5, sd=4.5)` days

## Code Style

- Use **seaborn** with `viridis` colormap for all plots
- Prefer **scipy vectorized operations** over loops
- Use `pathlib.Path` for file paths

## Git

- Never commit data files (`data/` is for local use only)
