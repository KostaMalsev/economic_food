
# Project Files Overview

## Core Files
- `bucketing_helper.py`: Contains the BucketingHelper class for consistent data bucketing and statistics calculations
- `base_visualizer.py`: Base visualization class with common functionality
- `graphs_expenditure.py`: ExpenditureVisualizer for graphs 1,2,13 (expenditure analysis),
- `graphs_sacrifice.py`: SacrificeVisualizer for graphs 5-6 (sacrifice analysis)
- `graphs_sufficiency.py`: SufficiencyVisualizer for graphs 7-8 (sufficiency analysis)
- `graphs_detailed.py`: DetailedVisualizer for graphs 9-10 (detailed analysis)
- `graphs_normalized.py`: NormalizedVisualizer for graphs 11-12 (normalized analysis)
- `visualization_manager.py`: Main class to coordinate all visualizers
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


      