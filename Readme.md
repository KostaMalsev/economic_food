
# Project Files Overview

## Repository layout

- `data/`: shared survey and regression CSV inputs.
- `analysis/empirical/`: empirical household-size thresholds and charts.
- `analysis/regression/model.py`: regression threshold model.
- `analysis/regression/reports/`: regression-based reports.
- `analysis/regression/visualization/`: legacy numbered graphs.
- `analysis/shared/`: common paths and chart styling only.
- `docs/assets/`: reference images.

Run the empirical figures with `python -m analysis.empirical.generate_charts`.
The compatible regression entry point remains `python run.py`.

## Core Files
- `analysis/regression/visualization/bucketing.py`: consistent data bucketing and statistics calculations
- `analysis/regression/visualization/base.py`: base visualization class
- `analysis/regression/visualization/`: expenditure, sacrifice, sufficiency, detailed, and normalized numbered graphs
- `analysis/regression/visualization/manager.py`: coordinates the numbered visualizers
- `__init__.py`: Package initialization with exports
- `requirements.txt`: Required Python packages



## Ravallion graph:
1.Graph1 - Food Expenditure - Y, X - Total spending. all per capita, show 45 deg. line
2. Find Ravallion ZL:
   a. Find bucket of households where mean FoodNorm ~==  mean TotalExpenditure(C3)
   b. ZL = mean of housholds ZL
3. Find Ravallion ZU:
   a. Find bucket of households where mean FoodNorm ~== mean FoodActual
   b. ZU = mean TotalExpenditure(C3)


