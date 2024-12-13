import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
from datetime import datetime
import pandas as pd

class GroupVisualizer:
    def __init__(self, save_dir='./graphs/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def save_plot(self, filename):
        """Save plot with timestamp"""
        full_path = os.path.join(self.save_dir, f"{filename}_{self.timestamp}.png")
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot: {full_path}")

    def create_scatter_plot(self, df, x, y, title, overlays=None):
        """Create scatter plot with optional overlays"""
        plt.figure(figsize=(12, 8))
        
        # Main scatter plot
        plt.scatter(df[x], df[y], alpha=0.5, label=f'Base')
        
        if overlays:
            colors = plt.cm.Set2(np.linspace(0, 1, len(overlays)))
            for overlay, color in zip(overlays, colors):
                if isinstance(overlay, tuple):  # Tuple indicates function overlay (e.g., y = 3x)
                    x_vals = df[x]
                    y_vals = overlay[1](x_vals)
                    plt.plot(x_vals, y_vals, color=color, label=overlay[0], linestyle='--')
                else:
                    plt.scatter(df[x], df[overlay], 
                              alpha=0.3, 
                              color=color,
                              label=f'{overlay}')
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt

    def create_household_graphs(self, df, is_sedentary=False):
        """Create graphs for household analysis"""
        pop_type = "Sedentary" if is_sedentary else "Active"
        foodnorm_col = "FoodNorm-sendetary" if is_sedentary else "FoodNorm-active"
        filtered_df = df
        
        # 1. Total Expenditure vs Food Expenditure
        self.create_scatter_plot(
            filtered_df,
            'c3',
            'food_actual',
            f'Total Expenditure vs Food Expenditure ({pop_type} Households)'
        )
        self.save_plot(f'1_expenditure_food_{pop_type.lower()}')
        
        # 2. Total Expenditure vs Non-Food Expenditure
        filtered_df['non_food'] = filtered_df['c3'] - filtered_df['food_actual']
        self.create_scatter_plot(
            filtered_df,
            'c3',
            'non_food',
            f'Total Expenditure vs Non-Food Expenditure ({pop_type} Households)'
        )
        self.save_plot(f'2_expenditure_nonfood_{pop_type.lower()}')
        
        # 3. Actual vs Estimated Expenditure
        zu = "ZU-sendetary" if is_sedentary=="Sendetary" else "ZU-active"
        filtered_df['c3_diff'] = filtered_df[zu] - filtered_df['c3']
        self.create_scatter_plot(
            filtered_df,
            'c3',
            zu,
            f'Actual vs Estimated Expenditure ({pop_type} Households)',
            overlays=['c3_diff']
        )
        self.save_plot(f'3_actual_estimated_{pop_type.lower()}')
        
        # 4. Food Expenditure vs Food Norm
        filtered_df['food_norm_diff'] = filtered_df[foodnorm_col] - filtered_df['food_actual']
        self.create_scatter_plot(
            filtered_df,
            'food_actual',
            'food_actual',
            f'Food Expenditure vs Food Norm ({pop_type} Households)',
            overlays=[
                foodnorm_col,
                'food_norm_diff'
            ]
        )
        self.save_plot(f'4_food_norm_{pop_type.lower()}')
        
        # 6. ZL vs ZU
        #        pop_type = "Sedentary" if is_sedentary else "Active"
        self.create_scatter_plot(
            filtered_df,
            "ZL-sendetary" if is_sedentary=="Sendetary" else "ZL-active",
            "ZU-sendetary" if is_sedentary=="Sendetary" else "ZU-active",
            f'ZL vs ZU ({pop_type} Households)'
        )
        self.save_plot(f'6_zl_zu_{pop_type.lower()}')
        
        # 7. ZL vs Food Norm with 3x overlay
        self.create_scatter_plot(
            filtered_df,
            "ZL-sendetary" if is_sedentary=="Sendetary" else "ZL-active",
            foodnorm_col,
            f'ZL vs Food Norm ({pop_type} Households)',
            overlays=[('3x Line', lambda x: 3*x)]
        )
        self.save_plot(f'7_zl_foodnorm_{pop_type.lower()}')
        
        # 8. Food Actual - Food Norm
        filtered_df['food_diff'] = filtered_df['food_actual'] - filtered_df[foodnorm_col]
        self.create_scatter_plot(
            filtered_df,
            'c3',
            'food_diff',
            f'FoodActual - FoodNorm ({pop_type} Households)'
        )
        self.save_plot(f'8_food_diff_{pop_type.lower()}')

    def create_person_graphs(self, df, is_sedentary=False):
        """Create per-person graphs"""
        pop_type = "Sedentary" if is_sedentary else "Active"
        foodnorm_col = "FoodNorm-sendetary" if is_sedentary else "FoodNorm-active"
        filtered_df = df
        
        # Calculate per-person metrics
        metrics = {
            'c3': 'c3',
            'food_actual': "food_actual",
            'food_norm': foodnorm_col,
            'ZL': "ZL-sendetary" if is_sedentary=="Sendetary" else "ZL-active",
            'ZU': "ZU-sendetary" if is_sedentary=="Sendetary" else "ZU-active",
            'non_food_per_person' : 'non_food',
        }
        
        for metric_name, col in metrics.items():
            filtered_df[f'{metric_name}_per_person'] = filtered_df[col] / filtered_df['persons_count']
        
        # Create per-person versions of all graphs
        # 1. Total Expenditure vs Food Expenditure per person
        self.create_scatter_plot(
            filtered_df,
            'c3_per_person',
            'food_actual_per_person',
            f'Total Expenditure vs Food Expenditure per Person ({pop_type})'
        )
        self.save_plot(f'1p_expenditure_food_{pop_type.lower()}')
        
        # 2. Total Expenditure vs Food Expenditure per person
        self.create_scatter_plot(
            filtered_df,
            'c3_per_person',
            'non_food_per_person_per_person',
            f'Total Expenditure vs Non-Food Expenditure per Person ({pop_type})'
        )
        self.save_plot(f'2p_expenditure_nonfood_{pop_type.lower()}')
        
        # Continue with all other per-person graphs...@@TBD
        # [Add similar code blocks for graphs 2-8 with _per_person metrics]
        
        
        
        
def add_visualization_to_analyzer(FamilyGroupAnalyzer):
    """Add visualization methods to the analyzer class"""
    def plot_and_save_groups(self, plot_dir=None):
        """Create and save all visualization plots"""
        if self.df is None:
            print("Please run analysis first")
            return
            
        plot_dir = plot_dir or './graphs/'
        visualizer = GroupVisualizer(plot_dir)
        
        # Create household-level graphs
        print("\nGenerating active household graphs...")
        visualizer.create_household_graphs(self.df, is_sedentary=False)
        
        print("\nGenerating sedentary household graphs...")
        visualizer.create_household_graphs(self.df, is_sedentary=True)
        
        # Create per-person graphs
        print("\nGenerating active per-person graphs...")
        visualizer.create_person_graphs(self.df, is_sedentary=False)
        
        print("\nGenerating sedentary per-person graphs...")
        visualizer.create_person_graphs(self.df, is_sedentary=True)
        
        print(f"\nAll plots have been saved in: {plot_dir}")
    
    # Add the method to the analyzer class
    FamilyGroupAnalyzer.plot_and_save_groups = plot_and_save_groups