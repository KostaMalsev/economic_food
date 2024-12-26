import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd


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
        
        # Create figure with adjusted size
        plt.figure(figsize=(15, 8))
        plt.rcParams.update({'font.size': 9})  # Base font size
        
        # Filter data for c3 <= 20000
        mask = df[f'c3{suffix}'] <= 20000
        filtered_df = df[mask]
        
        # Define columns
        x_col = f'c3{suffix}'
        y_col = f'food_actual{suffix}'
        norm_col = f'FoodNorm-{lifestyle}{suffix}'
        
        # Calculate deprivation for each household
        filtered_df['deprivation'] = filtered_df[y_col] - filtered_df[norm_col]
        
        # Create initial buckets of 1000 units
        bucket_size = 1000
        max_c3 = 20000
        buckets = range(0, max_c3 + bucket_size, bucket_size)
        
        # Initialize lists for merged bucket data
        merged_percentages = []
        merged_bucket_centers = []
        merged_bucket_counts = []
        merged_bucket_ranges = []
        merged_mean_deprivation = []
        merged_mean_household_size = []  # New list for average household size
        
        current_bucket_start = 0
        current_above_norm = 0
        current_total = 0
        current_deprivation_sum = 0
        current_household_size_sum = 0  # For calculating mean household size
        
        # Process buckets and merge when needed
        for start in buckets[:-1]:
            end = start + bucket_size
            bucket_mask = (filtered_df[x_col] >= start) & (filtered_df[x_col] < end)
            bucket_df = filtered_df[bucket_mask]
            
            if len(bucket_df) > 0:
                above_norm = (bucket_df[y_col] > bucket_df[norm_col]).sum()
                total = len(bucket_df)
                deprivation_sum = bucket_df['deprivation'].sum()
                household_size_sum = bucket_df['persons_count'].sum()  # Sum of household sizes
                
                current_above_norm += above_norm
                current_total += total
                current_deprivation_sum += deprivation_sum
                current_household_size_sum += household_size_sum
                
                if current_total >= 50 or end >= max_c3:
                    percentage = (current_above_norm / current_total) * 100
                    mean_deprivation = current_deprivation_sum / current_total
                    mean_household_size = current_household_size_sum / current_total
                    
                    merged_percentages.append(percentage)
                    merged_bucket_centers.append((current_bucket_start + end) / 2)
                    merged_bucket_counts.append(current_total)
                    merged_bucket_ranges.append((current_bucket_start, end))
                    merged_mean_deprivation.append(mean_deprivation)
                    merged_mean_household_size.append(mean_household_size)
                    
                    current_bucket_start = end
                    current_above_norm = 0
                    current_total = 0
                    current_deprivation_sum = 0
                    current_household_size_sum = 0
        
        # Plot merged buckets
        bar_width = [r[1] - r[0] for r in merged_bucket_ranges]
        bars = plt.bar(merged_bucket_centers, merged_percentages, 
                      width=[w*0.8 for w in bar_width], 
                      alpha=0.6, color=self.colors[0])
        
        # Add trend line
        z = np.polyfit(merged_bucket_centers, merged_percentages, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(min(merged_bucket_centers), max(merged_bucket_centers), 100)
        plt.plot(x_trend, p(x_trend), '--', color='red', label='Trend Line')
        
        # Set font sizes
        plt.xlabel(f'Total Expenditure (c3) {pop_type}', fontsize=10)
        plt.ylabel('Percentage of Households Above FoodNorm (%)', fontsize=10)
        plt.title(f'Households Above FoodNorm by C3 Range - {lifestyle.capitalize()} {pop_type}', 
                 fontsize=12, pad=20)
        
        # Add labels with improved positioning
        for i, (pct, count, center, bucket_range, mean_dep, mean_size) in enumerate(zip(
                merged_percentages, merged_bucket_counts, merged_bucket_centers, 
                merged_bucket_ranges, merged_mean_deprivation, merged_mean_household_size)):
            
            # Percentage on top of the bar
            plt.text(center, pct + 0.5, f'{pct:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
            
            # Mean deviation below percentage
            plt.text(center, pct - 3, f'Mean Dev: {mean_dep:.1f}', 
                    ha='center', va='top', fontsize=8)
            
            # Mean household size
            plt.text(center, pct - 5, f'Mean HH Size: {mean_size:.1f}', 
                    ha='center', va='top', fontsize=8)
            
            # Sample size
            plt.text(center, 2, f'n={count}', 
                    ha='center', va='bottom', fontsize=8)
            
            # Range at bottom
            range_text = f'{bucket_range[0]}-{bucket_range[1]}'
            plt.text(center, 2, range_text, 
                    ha='center', va='top', fontsize=8, rotation=45)
        
        # Adjust layout
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        
        # Set y-axis limit to accommodate labels
        plt.ylim(-5, max(merged_percentages) + 5)
        
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
    
    
    def create_deprivation_detailed_plot(self, df, lifestyle, per_capita=False, window_size=1000):
        """
        Plot Household food deprivation distribution with statistically significant 
        overrepresented household types in the deprived population.
        
        Parameters:
        - df: DataFrame with the data
        - lifestyle: 'active' or 'sedentary'
        - per_capita: Boolean for per capita metrics
        - window_size: Size of moving average window for smoothing
        """
        from scipy.stats import chi2_contingency, fisher_exact
        
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        # Calculate food deprivation
        food_actual = f'food_actual{suffix}'
        food_norm = f'FoodNorm-{lifestyle}{suffix}'
        deprivation = df[food_actual] - df[food_norm]
        
        # Create DataFrame with all relevant information
        age_columns = [
            "0 -4 min1", "0 -4 min2",    # 0-4 years
            "5 - 9 min1", "5 - 9 min2",  # 5-9 years
            "10-14 min1", "10-14 min2",  # 10-14 years
            "15 - 17 min1", "15 - 17 min2",  # 15-17 years
            "18 -29 min1", "18 -29 min2",  # 18-29 years
            "30 - 49 min1", "30 - 49 min2",  # 30-49 years
            "50+ min1", "50+ min2"        # 50+ years
        ]
        
        plot_df = pd.DataFrame({
            'deprivation': deprivation,
            'c3': df[f'c3{suffix}'],
            'household_size': df['persons_count'],
            **{col: df[col] for col in age_columns}
        })
        
        def get_household_type(row):
            """Create a standardized household type description"""
            composition = []
            age_groups = [
                ("0-4", ["0 -4 min1", "0 -4 min2"]),
                ("5-9", ["5 - 9 min1", "5 - 9 min2"]),
                ("10-14", ["10-14 min1", "10-14 min2"]),
                ("15-17", ["15 - 17 min1", "15 - 17 min2"]),
                ("18-29", ["18 -29 min1", "18 -29 min2"]),
                ("30-49", ["30 - 49 min1", "30 - 49 min2"]),
                ("50+", ["50+ min1", "50+ min2"])
            ]
            
            for age_group, cols in age_groups:
                count = sum(row[col] for col in cols)
                if count > 0:
                    composition.append(f"{int(count)}x{age_group}")
            
            return " + ".join(sorted(composition))
        
        # Add household type and deprivation status
        plot_df['household_type'] = plot_df.apply(get_household_type, axis=1)
        plot_df['is_deprived'] = plot_df['deprivation'] < 0
        
        # Initialize list to store statistical results
        stat_results = []
        
        # Get unique household types with at least 5 total households
        household_types = plot_df['household_type'].value_counts()[plot_df['household_type'].value_counts() >= 5].index
        
        # Perform statistical test for each household type
        for htype in household_types:
            # Create contingency table
            type_data = plot_df['household_type'] == htype
            contingency = pd.crosstab(type_data, plot_df['is_deprived'])
            
            try:
                # Use Fisher's exact test for small samples, Chi-square for larger ones
                if contingency.min().min() < 5:
                    oddsratio, pvalue = fisher_exact(contingency)
                else:
                    chi2, pvalue, _, _ = chi2_contingency(contingency)
                    oddsratio = (contingency.iloc[1,1] * contingency.iloc[0,0]) / \
                              (contingency.iloc[1,0] * contingency.iloc[0,1])
                
                # Calculate proportions
                type_deprived = (type_data & plot_df['is_deprived']).sum()
                type_total = type_data.sum()
                overall_deprived_rate = plot_df['is_deprived'].mean()
                type_deprived_rate = type_deprived / type_total
                
                # Store results
                stat_results.append({
                    'household_type': htype,
                    'pvalue': pvalue,
                    'odds_ratio': oddsratio,
                    'total_count': type_total,
                    'deprived_count': type_deprived,
                    'type_rate': type_deprived_rate,
                    'overall_rate': overall_deprived_rate,
                    'relative_risk': type_deprived_rate / overall_deprived_rate
                })
                
            except:
                continue
        
        # Convert to DataFrame and sort by p-value
        results_df = pd.DataFrame(stat_results)
        results_df = results_df[results_df['relative_risk'] > 1]  # Only overrepresented types
        results_df = results_df.sort_values('pvalue')
        
        # Print analysis
        print(f"\nStatistically Significant Overrepresented Household Types in Food Deprivation")
        print(f"({lifestyle} {pop_type})")
        print("=" * 80)
        
        # Print top 5 most significant results
        for i, row in results_df.head(5).iterrows():
            if row['pvalue'] > 0.05:  # Skip if not significant at 0.05 level
                continue
                
            print(f"\nType #{i+1}: {row['household_type']}")
            print(f"Statistical Significance:")
            print(f"  p-value: {row['pvalue']:.6f}")
            print(f"  Odds Ratio: {row['odds_ratio']:.2f}")
            
            print(f"Representation:")
            print(f"  Count in deprived population: {int(row['deprived_count'])} " \
                  f"of {int(row['total_count'])} ({row['type_rate']*100:.1f}%)")
            print(f"  Overall deprivation rate: {row['overall_rate']*100:.1f}%")
            print(f"  Relative Risk: {row['relative_risk']:.2f}x more likely to be deprived")
            
            # Get average deprivation and expenditure for this type
            type_households = plot_df[plot_df['household_type'] == row['household_type']]
            deprived_type = type_households[type_households['is_deprived']]
            if len(deprived_type) > 0:
                avg_deprivation = deprived_type['deprivation'].mean()
                avg_c3 = deprived_type['c3'].mean()
                print(f"Severity:")
                print(f"  Average deprivation: {avg_deprivation:.2f}")
                print(f"  Average expenditure (c3): {avg_c3:.2f}")
        
        # Sort DataFrame for visualization
        plot_df = plot_df.sort_values('deprivation').reset_index(drop=True)
        households = np.arange(len(plot_df))
        
        # Create figure with axes
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()  # Second y-axis for c3
        ax3 = ax1.twinx()  # Third y-axis for household size
        
        # Offset the right spines for visibility
        ax3.spines['right'].set_position(('outward', 60))
        
        # Calculate moving averages
        c3_smooth = plot_df['c3'].rolling(window=window_size, center=True).mean()
        size_smooth = plot_df['household_size'].rolling(window=window_size, center=True).mean()
        
        # Calculate poverty statistics
        poor_households = (plot_df['deprivation'] < 0).sum()
        poor_pct = (poor_households / len(plot_df)) * 100
        
        # Plot deprivation on left y-axis
        deprivation_scatter = ax1.scatter(households, plot_df['deprivation'], 
                                        color='purple', alpha=0.5, s=10,
                                        label='Food Deprivation')
        
        # Plot smoothed c3 on first right y-axis
        c3_line = ax2.plot(households, c3_smooth, 
                        color='green', alpha=0.7, linewidth=1,
                        label=f'Total Expenditure (c3) - {window_size}-point MA')
        
        # Plot smoothed household size on second right y-axis
        size_line = ax3.plot(households, size_smooth, 
                            color='blue', alpha=0.7, linewidth=1,
                            label=f'Household Size - {window_size}-point MA')
        
        # Add reference lines and regions
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Zero Point (No Deprivation)')
        ax1.axvline(x=poor_households, color='red', linestyle='--', alpha=0.3,
                    label=f'Poverty Line ({poor_pct:.1f}% households)')
        
        # Shade poverty region
        ax1.fill_between(households[:poor_households],
                        plot_df['deprivation'][:poor_households],
                        0, color='red', alpha=0.1,
                        label='Poor')
        
        # Set labels and title
        ax1.set_xlabel('Households (Ordered by Food Deprivation)')
        ax1.set_ylabel(f'Food Expenditure - FoodNorm {pop_type}', color='purple')
        ax2.set_ylabel(f'Total Expenditure (c3) {pop_type}', color='green')
        ax3.set_ylabel('Household Size', color='blue')
        plt.title(f'Household Food Deprivation Distribution - {lifestyle.capitalize()} {pop_type}')
        
        # Color the tick labels
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        # Add grid
        ax1.grid(True, alpha=0.2)
        
        # Combine legends from all axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, 
                  labels1 + labels2 + labels3,
                  loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
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
                    '''
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
                   '''
                    
                    
                    # 9. Household distribution
                    print("    - Creating  plot...")
                    plt = visualizer.create_deprivation_detailed_plot(self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'9_create_deprivation_detailed_{lifestyle}_{metric_type}')
                   
                    
                except Exception as e:
                    print(f"Error generating {metric_type} plots for {lifestyle} lifestyle: {str(e)}")
        
        print(f"\nAll plots have been saved in: {plot_dir}")
    
    # Add the method to the analyzer class
    FamilyGroupAnalyzer.plot_and_save_groups = plot_and_save_groups