import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class GroupVisualizer:
    def __init__(self, save_dir='./graphs/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set consistent style for all plots
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['figure.figsize'] = [12, 8]
        self.colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']  # Colorblind-friendly palette
        
    def save_plot(self, filename):
        """Save plot with timestamp"""
        full_path = os.path.join(self.save_dir, f"{filename}_{self.timestamp}.png")
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot: {full_path}")

    def create_food_vs_expenses_plot(self, df, lifestyle, per_capita=False):
        """Plot 1: Total Expenditure (c3) vs Food Expenditure and FoodNorm"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure()
        
        # Define columns according to glossary
        x_col = f'c3{suffix}'  # Food ExpenditureX
        y_col = f'food_actual{suffix}'#-{lifestyle}{suffix}'  # Food ExpenditureY
        norm_col = f'FoodNorm-{lifestyle}{suffix}'
        
        # Plot metrics
        plt.scatter(df[x_col], df[y_col], alpha=0.5, color=self.colors[0], 
                   label='Food Expenditure')
        plt.scatter(df[x_col], df[norm_col], alpha=0.5, color=self.colors[1], 
                   label='FoodNorm')
        
        # Plot difference
        diff = df[y_col] - df[norm_col]
        plt.scatter(df[x_col], diff, alpha=0.5, color=self.colors[2], 
                   label='Difference')
        
        plt.xlabel('Total Expenditure (c3)')
        plt.ylabel('Food Expenditure')
        plt.title(f'Food Expenditure vs Total Expenditure - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_c3_zu_plot(self, df, lifestyle, per_capita=False):
        """Plot 2: c3 vs ZU (c^3) comparison"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure()
        
        x_col = f'c3{suffix}'
        zu_col = f'ZU-{lifestyle}{suffix}'
        
        # Plot c3 vs ZU (c^3)
        plt.scatter(df[x_col], df[x_col], alpha=0.5, color=self.colors[0], 
                   label='c3')
        plt.scatter(df[x_col], df[zu_col], alpha=0.5, color=self.colors[1], 
                   label=f'ZU {lifestyle} (c^3)')
        
        # Plot difference
        diff = df[zu_col] - df[x_col]
        plt.scatter(df[x_col], diff, alpha=0.5, color=self.colors[2], 
                   label='Difference')
        
        plt.xlabel('Total Expenditure (c3)')
        plt.ylabel('Value')
        plt.title(f'c3 vs ZU Comparison - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_food_comparison_plot(self, df, lifestyle, per_capita=False):
        """Plot 3: Food Expenditure vs FoodNorm comparison"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure()
        
        # Get correct columns based on lifestyle from glossary
        food_actual = f'food_actual{suffix}'#-{lifestyle}{suffix}'
        food_norm = f'FoodNorm-{lifestyle}{suffix}'
        
        # Plot metrics
        plt.scatter(df[food_actual], df[food_actual], alpha=0.5, color=self.colors[0], 
                   label='Food Expenditure')
        plt.scatter(df[food_actual], df[food_norm], alpha=0.5, color=self.colors[1], 
                   label='FoodNorm')
        
        # Plot difference
        diff = df[food_actual] - df[food_norm]
        plt.scatter(df[food_actual], diff, alpha=0.5, color=self.colors[2], 
                   label='Difference')
        
        plt.xlabel('Food Expenditure')
        plt.ylabel('Value')
        plt.title(f'Food Expenditure vs Norm Comparison - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_zl_zu_plot(self, df, lifestyle, per_capita=False):
        """Plot 4: ZL vs ZU relationship with FoodNorm"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure()
        
        # Get columns based on lifestyle
        zl_col = f'ZL-{lifestyle}{suffix}'
        zu_col = f'ZU-{lifestyle}{suffix}'
        food_norm_col = f'FoodNorm-{lifestyle}{suffix}'
        
        # Plot relationship
        plt.scatter(df[zl_col], df[zu_col], alpha=0.5, color=self.colors[0], 
                   label=f'ZU {lifestyle}')
        plt.scatter(df[zl_col], df[food_norm_col], alpha=0.5, color=self.colors[1], 
                   label='FoodNorm')
        
        # Add 3x reference line
        zl_range = np.array([df[zl_col].min(), df[zl_col].max()])
        plt.plot(zl_range, 3 * zl_range, '--', color=self.colors[2], 
                label='3x Reference')
        
        plt.xlabel('ZL')
        plt.ylabel('Value')
        plt.title(f'ZU and FoodNorm vs ZL - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_deprivation_plot(self, df, lifestyle, per_capita=False):
        """Plot 5: Household food deprivation distribution"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure()
        
        # Calculate food deprivation using correct fields from glossary
        food_actual = f'food_actual{suffix}'#-{lifestyle}{suffix}'
        food_norm = f'FoodNorm-{lifestyle}{suffix}'
        deprivation = df[food_actual] - df[food_norm]
        
        # Sort households by deprivation
        sorted_deprivation = np.sort(deprivation)
        households = np.arange(len(sorted_deprivation))
        
        # Calculate poverty statistics
        poor_households = (sorted_deprivation < 0).sum()
        #print(f'deprivation-{lifestyle}{suffix} {deprivation}')
        
        poor_pct = (poor_households / len(sorted_deprivation)) * 100
        
        # Create distribution plot
        plt.plot(households, sorted_deprivation, color=self.colors[0], 
                label='Food Deprivation')
        
        # Add reference lines and regions
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                   label='Zero Point (No Deprivation)')
        plt.axvline(x=poor_households, color='red', linestyle='--', alpha=0.3,
                   label=f'Poverty Line ({poor_pct:.1f}% households)')
        
        # Shade poverty region
        plt.fill_between(households[:poor_households], 
                        sorted_deprivation[:poor_households], 
                        0, color='red', alpha=0.1, 
                        label='Poor Households')
        
        plt.xlabel('Households (Ordered by Food Deprivation)')
        plt.ylabel('Food Expenditure - FoodNorm')
        plt.title(f'Household Food Deprivation Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

def add_visualization_to_analyzer(FamilyGroupAnalyzer):
    """Add visualization methods to the analyzer class"""
    def plot_and_save_groups(self, plot_dir=None):
        """Create and save all visualization plots"""
        if self.df is None:
            print("Please run analysis first")
            return
            
        plot_dir = plot_dir or './graphs/'
        visualizer = GroupVisualizer(plot_dir)
        
        print("\nGenerating plots...")
        # Generate plots for both lifestyles and both household/per-capita metrics
        for lifestyle in ['active', 'sedentary']:
            print(f"\nProcessing {lifestyle} lifestyle plots:")
            for per_capita in [False, True]:
                metric_type = 'per_capita' if per_capita else 'household'
                print(f"\n  Generating {metric_type} metrics:")
                
                try:
                    # 1. Food vs Total Expenses
                    print("    - Creating food vs expenses plot...")
                    plt = visualizer.create_food_vs_expenses_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'1_food_expenses_{lifestyle}_{metric_type}')
                    
                    # 2. c3 vs ZU
                    print("    - Creating c3 vs ZU plot...")
                    plt = visualizer.create_c3_zu_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'2_c3_zu_{lifestyle}_{metric_type}')
                    
                    # 3. Food Expenditure vs Norm
                    print("    - Creating food comparison plot...")
                    plt = visualizer.create_food_comparison_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'3_food_comparison_{lifestyle}_{metric_type}')
                    
                    # 4. ZL vs ZU relationship
                    print("    - Creating ZL vs ZU plot...")
                    plt = visualizer.create_zl_zu_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'4_zl_zu_{lifestyle}_{metric_type}')
                    
                    # 5. Household distribution
                    print("    - Creating deprivation plot...")
                    plt = visualizer.create_deprivation_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'5_deprivation_{lifestyle}_{metric_type}')
                    
                except Exception as e:
                    print(f"Error generating {metric_type} plots for {lifestyle} lifestyle: {str(e)}")
        
        print(f"\nAll plots have been saved in: {plot_dir}")
    
    # Add the method to the analyzer class
    FamilyGroupAnalyzer.plot_and_save_groups = plot_and_save_groups