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
        
        # Filter data for c3 <= 60000
        mask = df[f'c3{suffix}'] <= 20000#60000
        filtered_df = df[mask]
        
        # Define columns according to glossary
        x_col = f'c3{suffix}'  # Food ExpenditureX
        y_col = f'food_actual{suffix}'  # Food ExpenditureY
        norm_col = f'FoodNorm-{lifestyle}{suffix}'
        
        # Plot metrics with filtered data
        plt.scatter(filtered_df[x_col], filtered_df[y_col], alpha=0.5, color=self.colors[0], 
                label='Food Expenditure')
        plt.scatter(filtered_df[x_col], filtered_df[norm_col], alpha=0.5, color=self.colors[1], 
                label='FoodNorm')
        
        # Plot difference
        diff = filtered_df[y_col] - filtered_df[norm_col]
        plt.scatter(filtered_df[x_col], diff, alpha=0.5, color=self.colors[2], 
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
        diff = df[x_col] - df[zu_col]
        plt.scatter(df[x_col], diff, alpha=0.5, color=self.colors[2], 
                   label='C3-C3^')
        
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
                        label='Poor')
        
        plt.xlabel('Households (Ordered by Food Deprivation)')
        plt.ylabel('Food Expenditure - FoodNorm')
        plt.title(f'Household Food Deprivation Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt
    
    
    def create_non_food_deprivation_plot(self, df, lifestyle, per_capita=False):
        """Plot 6: Household food deprivation distribution"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure()
        
        
        # Calculate food deprivation using correct fields from glossary
        tot_exp = f'c3{suffix}'#-{lifestyle}{suffix}'
        norm_expendeture = f'ZU-{lifestyle}{suffix}' #c3^ - which is upper limit for food expendetures
        non_food_deprivation = df[tot_exp] - df[norm_expendeture]
        
        # Sort households by deprivation
        non_food_deprivation = np.sort(non_food_deprivation)
        households = np.arange(len(non_food_deprivation))
        
        # Calculate poverty statistics
        poor_households = (non_food_deprivation < 0).sum()
        
        poor_pct = (poor_households / len(non_food_deprivation)) * 100
        
        # Create distribution plot
        plt.plot(households, non_food_deprivation, color=self.colors[0], 
                label='Food Deprivation')
        
        # Add reference lines and regions
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                   label='Zero Point (No Deprivation)')
        plt.axvline(x=poor_households, color='red', linestyle='--', alpha=0.3,
                   label=f'Poverty Line ({poor_pct:.1f}% households)')
        
        # Shade poverty region
        plt.fill_between(households[:poor_households], 
                        non_food_deprivation[:poor_households], 
                        0, color='red', alpha=0.1, 
                        label='Poor')
        
        plt.xlabel('Households (Ordered by Non Food Deprivation)')
        plt.ylabel('Total Expenditure - C3^')
        plt.title(f'Household Non Food Deprivation Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt
    
    def create_above_norm_percentage_plot(self, df, lifestyle, per_capita=False):
        """Plot percentage of households above FoodNorm per 1000-unit bucket of c3, merging small buckets"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure(figsize=(15, 8))
        
        # Filter data for c3 <= 20000
        mask = df[f'c3{suffix}'] <= 20000
        filtered_df = df[mask]
        
        # Define columns
        x_col = f'c3{suffix}'
        y_col = f'food_actual{suffix}'
        norm_col = f'FoodNorm-{lifestyle}{suffix}'
        
        # Create initial buckets of 1000 units
        bucket_size = 1000
        max_c3 = 20000
        buckets = range(0, max_c3 + bucket_size, bucket_size)
        
        # Initialize lists for merged bucket data
        merged_percentages = []
        merged_bucket_centers = []
        merged_bucket_counts = []
        merged_bucket_ranges = []
        
        current_bucket_start = 0
        current_above_norm = 0
        current_total = 0
        
        # Process buckets and merge when needed
        for start in buckets[:-1]:
            end = start + bucket_size
            bucket_mask = (filtered_df[x_col] >= start) & (filtered_df[x_col] < end)
            bucket_df = filtered_df[bucket_mask]
            
            if len(bucket_df) > 0:
                above_norm = (bucket_df[y_col] > bucket_df[norm_col]).sum()
                total = len(bucket_df)
                
                current_above_norm += above_norm
                current_total += total
                
                # If we have >= 50 samples or this is the last bucket, save the merged bucket
                if current_total >= 50 or end >= max_c3:
                    percentage = (current_above_norm / current_total) * 100
                    merged_percentages.append(percentage)
                    merged_bucket_centers.append((current_bucket_start + end) / 2)
                    merged_bucket_counts.append(current_total)
                    merged_bucket_ranges.append((current_bucket_start, end))
                    
                    # Reset accumulators
                    current_bucket_start = end
                    current_above_norm = 0
                    current_total = 0
        
        # Plot merged buckets
        bar_width = [r[1] - r[0] for r in merged_bucket_ranges]
        plt.bar(merged_bucket_centers, merged_percentages, width=[w*0.8 for w in bar_width], 
                alpha=0.6, color=self.colors[0])
        
        # Add trend line (קו מגמה)
        z = np.polyfit(merged_bucket_centers, merged_percentages, 2)  # Quadratic fit
        p = np.poly1d(z)
        x_trend = np.linspace(min(merged_bucket_centers), max(merged_bucket_centers), 100)
        plt.plot(x_trend, p(x_trend), '--', color='red', label='Trend Line')
        
        plt.xlabel('Total Expenditure (c3)')
        plt.ylabel('Percentage of Households Above FoodNorm (%)')
        plt.title(f'Households Above FoodNorm by C3 Range - {lifestyle.capitalize()} {pop_type}')
        
        # Add percentage and count labels on bars
        for i, (pct, count, center, bucket_range) in enumerate(zip(
                merged_percentages, merged_bucket_counts, merged_bucket_centers, merged_bucket_ranges)):
            # Percentage on top
            plt.text(center, pct + 1, f'{pct:.1f}%', ha='center')
            # Count and range in middle of bar
            range_text = f'{bucket_range[0]}-{bucket_range[1]}'
            plt.text(center, pct/2, f'n={count}\n{range_text}', 
                    ha='center', va='center')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        
        return plt
    
    
    def create_zu_percentage_plot(self, df, lifestyle, per_capita=False, min_c3=4000, max_c3=10000, bucket_size=100):
        """
        Plot percentage of households where c3 > ZU per bucket of c3, merging small buckets
        
        Parameters:
        - min_c3: Minimum c3 value to include
        - max_c3: Maximum c3 value to include
        - bucket_size: Size of each bucket before merging
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        plt.figure(figsize=(15, 8))
        
        # Filter data for c3 between min and max
        mask = (df[f'c3{suffix}'] >= min_c3) & (df[f'c3{suffix}'] <= max_c3)
        filtered_df = df[mask]
        
        # Define columns
        x_col = f'c3{suffix}'
        zu_col = f'ZU-{lifestyle}{suffix}'
        
        # Create buckets
        buckets = range(min_c3, max_c3 + bucket_size, bucket_size)
        
        # Initialize lists for merged bucket data
        merged_percentages = []
        merged_bucket_centers = []
        merged_bucket_counts = []
        merged_bucket_ranges = []
        
        current_bucket_start = min_c3  # Start from min_c3
        current_above_zu = 0
        current_total = 0
        
        # Process buckets and merge when needed
        for start in buckets[:-1]:
            end = start + bucket_size
            bucket_mask = (filtered_df[x_col] >= start) & (filtered_df[x_col] < end)
            bucket_df = filtered_df[bucket_mask]
            
            if len(bucket_df) > 0:
                above_zu = (bucket_df[x_col] > bucket_df[zu_col]).sum()
                total = len(bucket_df)
                
                current_above_zu += above_zu
                current_total += total
                
                # If we have >= 50 samples or this is the last bucket, save the merged bucket
                if current_total >= 50 or end >= max_c3:
                    percentage = (current_above_zu / current_total) * 100
                    merged_percentages.append(percentage)
                    merged_bucket_centers.append((current_bucket_start + end) / 2)
                    merged_bucket_counts.append(current_total)
                    merged_bucket_ranges.append((current_bucket_start, end))
                    
                    # Reset accumulators
                    current_bucket_start = end
                    current_above_zu = 0
                    current_total = 0
        
        # Plot merged buckets
        bar_width = [r[1] - r[0] for r in merged_bucket_ranges]
        plt.bar(merged_bucket_centers, merged_percentages, width=[w*0.8 for w in bar_width], 
                alpha=0.6, color=self.colors[0])
        
        # Add trend line 
        z = np.polyfit(merged_bucket_centers, merged_percentages, 2)  # Quadratic fit
        p = np.poly1d(z)
        x_trend = np.linspace(min(merged_bucket_centers), max(merged_bucket_centers), 100)
        plt.plot(x_trend, p(x_trend), '--', color='red', label='Trend Line')
        
        plt.xlabel('Total Expenditure (c3)')
        plt.ylabel('Percentage of Households where c3 > ZU (%)')
        plt.title(f'Households with c3 > ZU by C3 Range ({min_c3}-{max_c3}) - {lifestyle.capitalize()} {pop_type}')
        
        # Add percentage and count labels on bars
        '''for i, (pct, count, center, bucket_range) in enumerate(zip(
                merged_percentages, merged_bucket_counts, merged_bucket_centers, merged_bucket_ranges)):
            # Percentage on top
            plt.text(center, pct + 1, f'{pct:.1f}%', ha='center')
            # Count and range in middle of bar
            range_text = f'{bucket_range[0]}-{bucket_range[1]}'
            plt.text(center, pct/2, f'n={count}\n{range_text}', 
                    ha='center', va='center')
        '''
        for i, (pct, center) in enumerate(zip(merged_percentages, merged_bucket_centers)):
            # Percentage on top
            plt.text(center, pct + 1, f'{pct:.1f}%', ha='center')
        
        plt.grid(True, axis='y', alpha=0.3)
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
                    
                    # 6. Houshold distribution for non food expendeture
                    print("    - Creating deprivation plot...")
                    plt = visualizer.create_non_food_deprivation_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'6_non_food_deprivation_{lifestyle}_{metric_type}')
                    
                    # 7. Percentage of households/per capita above foodnorm
                    print("    - Creating  plot...")
                    plt = visualizer.create_above_norm_percentage_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'7_percentage_above_foodnorm_{lifestyle}_{metric_type}')
                    
                    # 8. Percentage of households/per capita where c3 is above c3^
                    print("    - Creating  plot...")
                    plt = visualizer.create_zu_percentage_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'8_percentage_c3_above_c3gag_{lifestyle}_{metric_type}')
                    
                    
                except Exception as e:
                    print(f"Error generating {metric_type} plots for {lifestyle} lifestyle: {str(e)}")
        
        print(f"\nAll plots have been saved in: {plot_dir}")
    
    # Add the method to the analyzer class
    FamilyGroupAnalyzer.plot_and_save_groups = plot_and_save_groups