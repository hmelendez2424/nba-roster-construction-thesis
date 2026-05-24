# Optimizing NBA Roster Construction

A Unified Framework for Performance and Salary Predictive Modelling Using Dimensionality Reduction.

MSc thesis project, Tilburg University, Data Science & Society (2026).
**Author:** Hugo Meléndez Moriñigo
**Supervisor:** Dr. Fred Atilla

## Overview

This project builds an integrated machine learning pipeline that:

- Forecasts next-season player performance (PIE) and cap-adjusted salary share, both one season ahead.
- Discovers player archetypes via UMAP dimensionality reduction and KMeans clustering (K=7).
- Compares five model families (Ridge, ElasticNet, XGBoost, LightGBM, MLP) against a persistence baseline.
- Conducts stratified error analysis across player archetypes and age groups.

The dataset spans 20 NBA seasons (2005/06 to 2024/25), with a strict temporal split: train (2006–2019), validation (2020–2022), test (2023–2024).

## Repository contents

- `thesis_pipeline.ipynb` — main Jupyter notebook containing the full pipeline, from data loading to results and figures.
- `nba_merged_clean.csv` — cleaned merged player-season dataset used by the notebook.
- `requirements.txt` — Python dependencies.

## Dataset

`nba_merged_clean.csv` contains one row per player-season. Key columns used by the pipeline:

- `SEASON_END` — season end year (e.g. 2024 for the 2023/24 season).
- `PLAYER_NAME_NORM` — normalized player name (used as identifier).
- `PIE_ADVANCED` — Player Impact Estimate, the performance target.
- `salary_share` — player salary as a share of the team salary cap, the salary target.
- `SALARY_CAP` — league salary cap for the season (used to convert salary share back to dollars).

Additional columns include traditional and advanced box-score statistics, used as features.

## Reproducing the results

### Requirements

- Python 3.12.4 (other 3.10+ versions should also work).
- Dependencies listed in `requirements.txt`.

Install with:

```bash
pip install -r requirements.txt
```

### Run

1. Clone or download this repository.
2. Make sure `thesis_pipeline.ipynb` and `nba_merged_clean.csv` are in the same folder.
3. Open the notebook in Jupyter or VS Code.
4. Run all cells in order (Kernel → Restart and Run All).

### Reproducibility notes

- Global random seed is `RANDOM_STATE = 42`.
- For deterministic UMAP clustering, the implementation sets both `random_state=42` **and** `n_jobs=1`. Setting only the seed is not sufficient — UMAP's parallel execution introduces non-determinism unless `n_jobs=1` is also enforced.
- Optuna uses a TPE sampler with the same seed.
- MLP models are trained across multiple seeds (42, 0, 1, 7, 13, 99, 123, 200, 314, 777) and the best-validating seed is retained.

## Citation

If you use this code or data, please cite:

> Meléndez Moríñigo, H. (2026). *Optimizing NBA Roster Construction: A Unified Framework for Player Performance and Salary Predictive Modelling Using Dimensionality Reduction*. MSc thesis, Tilburg University.

## License

MIT License. See `LICENSE` for details.
