# Analysis structure

The repository keeps the two threshold methods separate:

- `empirical/` contains family-size empirical thresholds, extracted household inputs, and the five requested charts.
- `regression/model.py` contains the coefficient-based household regression model.
- `regression/reports/` contains standalone regression-based analyses.
- `regression/visualization/` contains the legacy numbered-graph engine.
- `shared/` contains presentation utilities used by both methods. It contains no threshold calculations.

Run the empirical chart set from the repository root:

```bash
python -m analysis.empirical.generate_charts
```

Run the regression flow through its stable compatibility entry point:

```bash
python run.py
```

Shared survey CSVs live in the repository-level `data/` directory. Empirical-only
extracted inputs remain beside the empirical method in `empirical/data/`.
