import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
from scipy import stats

class GroupVisualizer:
    def __init__(self, save_dir='./graphs/'):
        """Initialize visualizer with consistent styling"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set consistent style
        plt.style.use('default')
        plt.rcParams.update({
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.figsize': [15, 8],
            'font.size': 10
        })
        self.colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        
    def save_plot(self, filename):
        """Save plot with timestamp"""
        full_path = os.path.join(self.save_dir, f"{filename}_{self.timestamp}.png")
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot: {full_path}")

    def _get_display_type(self, per_capita):
        """Helper to get consistent display strings"""
        return 'Per Capita' if per_capita else 'Household'

    def create_graph1(self, df, lifestyle, per_capita=False):
        """Graph #1: Food Expenditure vs Total Expenditure Analysis 
        * X-axis: Total Expenditure (c3)
        * Y-axes: Food Expenditure and Food Norm
        * Additional: Gap between actual food expenditure and norm
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure(figsize=(10,6))  
        mask = df[f'c3{suffix}'] <= 20000
        df_filtered = df[mask].copy()
        
        # Plot metrics
        x = df_filtered[f'c3{suffix}']
        y_actual = df_filtered[f'food_actual{suffix}']
        y_norm = df_filtered[f'FoodNorm-{lifestyle}{suffix}']
        
        plt.scatter(x, y_actual, s=0.1, alpha=0.6, color=self.colors[0], 
                    label='Food Expenditure')
        plt.scatter(x, y_norm, s=0.1, alpha=0.6, color=self.colors[1],
                    label='Food Norm')
        
        # Plot gap  
        gap = y_actual - y_norm
        plt.scatter(x, gap, s=1, alpha=0.6, color=self.colors[2], 
                    label='Gap')
        
        # Add trend lines
        z_actual = np.polyfit(x, y_actual, 1)
        p_actual = np.poly1d(z_actual)
        plt.plot(x, p_actual(x), color=self.colors[0], linestyle='--', alpha=0.8)
        
        z_norm = np.polyfit(x, y_norm, 1)
        p_norm = np.poly1d(z_norm)
        plt.plot(x, p_norm(x), color=self.colors[1], linestyle='--', alpha=0.8)
        
        z_gap = np.polyfit(x, gap, 1)
        p_gap = np.poly1d(z_gap)
        plt.plot(x, p_gap(x), color=self.colors[2], linestyle='--', alpha=0.8)
        
        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Value') 
        plt.title(f'Graph #1: Food Expenditure vs Total Expenditure - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt
    
    

    def create_graph2(self, df, lifestyle, per_capita=False):
        """Graph #2: Total Expenditure vs Upper Poverty Line Comparison
        
        * X-axis: Total Expenditure (c3)
        * Y-axes: Total Expenditure (c3) and Upper Poverty Line (ZU)
        * Additional: Gap analysis between c3 and ZU
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        x = df[f'c3{suffix}']
        zu = df[f'ZU-{lifestyle}{suffix}']
        
        plt.scatter(x, x, alpha=0.5, color=self.colors[0], 
                   label='Total Expenditure (c3)')
        plt.scatter(x, zu, alpha=0.5, color=self.colors[1], 
                   label='Upper Poverty Line (ZU)')
        
        # Plot gap
        gap = x - zu
        plt.scatter(x, gap, alpha=0.5, color=self.colors[2], 
                   label='Gap')
        
        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Value')
        plt.title(f'Graph #2: Total Expenditure vs Upper Poverty Line - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_graph3(self, df, lifestyle, per_capita=False):
        """Graph #3: Food Expenditure vs Food Norm Relationship
        
        * X-axis: Food Expenditure
        * Y-axes: Food Expenditure and Food Norm
        * Additional: Deviation from norm analysis
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        x = df[f'food_actual{suffix}']
        norm = df[f'FoodNorm-{lifestyle}{suffix}']
        
        plt.scatter(x, x, alpha=0.5, color=self.colors[0], 
                   label='Food Expenditure')
        plt.scatter(x, norm, alpha=0.5, color=self.colors[1], 
                   label='Food Norm')
        
        # Plot difference
        diff = x - norm
        plt.scatter(x, diff, alpha=0.5, color=self.colors[2], 
                   label='Deviation')
        
        plt.xlabel(f'Food Expenditure {pop_type}')
        plt.ylabel('Value')
        plt.title(f'Graph #3: Food Expenditure vs Food Norm - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_graph4(self, df, lifestyle, per_capita=False):
        """Graph #4: Poverty Lines Relationship
        
        * Analysis of ZL (Lower Poverty Line) vs ZU (Upper Poverty Line)
        * Include Food Norm overlay
        * Test for 3x relationship between ZL and ZU
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        zl = df[f'ZL-{lifestyle}{suffix}']
        zu = df[f'ZU-{lifestyle}{suffix}']
        norm = df[f'FoodNorm-{lifestyle}{suffix}']
        
        plt.scatter(zl, zu, alpha=0.5, color=self.colors[0], 
                   label='Upper Poverty Line (ZU)')
        plt.scatter(zl, norm, alpha=0.5, color=self.colors[1], 
                   label='Food Norm')
        
        # Add 3x reference line
        zl_range = np.array([zl.min(), zl.max()])
        plt.plot(zl_range, 3 * zl_range, '--', color=self.colors[2], 
                label='3x Reference Line')
        
        plt.xlabel(f'Lower Poverty Line (ZL) {pop_type}')
        plt.ylabel('Value')
        plt.title(f'Graph #4: Poverty Lines Relationship - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_graph5(self, df, lifestyle, per_capita=False):
        """Graph #5: Food Sacrifice Distribution
        
        * X-axis: Households ordered by food sacrifice level
        * Y-axis: Food sacrifice (Food Expenditure - Food Norm)
        * Features: Poverty threshold at zero
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        
        # Calculate food sacrifice
        actual = df[f'food_actual{suffix}']
        norm = df[f'FoodNorm-{lifestyle}{suffix}']
        sacrifice = actual - norm
        sorted_sacrifice = np.sort(sacrifice)
        households = np.arange(len(sorted_sacrifice))
        
        # Calculate statistics
        sacrificing = (sorted_sacrifice < 0).sum()
        sacrificing_pct = (sacrificing / len(sorted_sacrifice)) * 100
        
        plt.plot(households, sorted_sacrifice, color=self.colors[0], 
                label='Food Sacrifice')
        
        # Add reference lines
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                   label='Zero Point (No Sacrifice)')
        plt.axvline(x=sacrificing, color='red', linestyle='--', alpha=0.3,
                   label=f'Sacrifice Line ({sacrificing_pct:.1f}% households)')
        
        plt.fill_between(households[:sacrificing], sorted_sacrifice[:sacrificing],
                        0, color='red', alpha=0.1, label='Food Sacrificing')
        
        plt.xlabel('Households (Ordered by Food Sacrifice)')
        plt.ylabel(f'Food Expenditure - Food Norm {pop_type}')
        plt.title(f'Graph #5: Household Food Sacrifice Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt

    def create_graph6(self, df, lifestyle, per_capita=False):
        """Graph #6: Total Expenditure Sacrifice Distribution
        
        * X-axis: Households ordered by expenditure sacrifice
        * Y-axis: Expenditure gap (c3 - ZU)
        * Similar structure to Graph #5 but focusing on total expenditure
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        
        # Calculate expenditure sacrifice
        c3 = df[f'c3{suffix}']
        zu = df[f'ZU-{lifestyle}{suffix}']
        sacrifice = c3 - zu
        sorted_sacrifice = np.sort(sacrifice)
        households = np.arange(len(sorted_sacrifice))
        
        # Calculate statistics
        sacrificing = (sorted_sacrifice < 0).sum()
        sacrificing_pct = (sacrificing / len(sorted_sacrifice)) * 100
        
        plt.plot(households, sorted_sacrifice, color=self.colors[0], 
                label='Total Expenditure Sacrifice')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                   label='Zero Point (No Sacrifice)')
        plt.axvline(x=sacrificing, color='red', linestyle='--', alpha=0.3,
                   label=f'Sacrifice Line ({sacrificing_pct:.1f}% households)')
        
        plt.fill_between(households[:sacrificing], sorted_sacrifice[:sacrificing],
                        0, color='red', alpha=0.1, label='Expenditure Sacrificing')
        
        plt.xlabel('Households (Ordered by Expenditure Sacrifice)')
        plt.ylabel(f'Total Expenditure - Upper Poverty Line {pop_type}')
        plt.title(f'Graph #6: Expenditure Sacrifice Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt
    
    
    def plot_bucket_histogram(self, df, lifestyle, per_capita=True, start_bucket=0, end_bucket=9, max_value=7000):
        """
        Plot overlaid histograms of food expenditure values for each bucket in the specified range
        with improved layout and visibility
        
        Parameters:
        - df: DataFrame containing the data
        - lifestyle: 'active' or 'sedentary'
        - per_capita: Boolean for per capita metrics
        - start_bucket: Starting bucket ID
        - end_bucket: Ending bucket ID
        - max_value: Maximum value for x-axis (default: 12000)
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'
        
        # Create figure with two subplots - main histogram and sample sizes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                    gridspec_kw={'height_ratios': [4, 1]})
        bucket_size = 250
        
        # Filter and sort data
        mask = df[f'c3{suffix}'] <= 20000
        df_filtered = df[mask].copy().sort_values(f'c3{suffix}')
        
        # Create fixed-width buckets
        min_c3 = df_filtered[f'c3{suffix}'].min()
        max_c3 = df_filtered[f'c3{suffix}'].max()
        n_buckets = len(df_filtered) // bucket_size
        bucket_width = (max_c3 - min_c3) / n_buckets
        
        # Create bucket boundaries and assign buckets
        bucket_edges = np.arange(min_c3, max_c3 + bucket_width, bucket_width)
        df_filtered['bucket'] = pd.cut(df_filtered[f'c3{suffix}'],
                                    bins=bucket_edges,
                                    labels=False,
                                    include_lowest=True)
        
        # Set up color cycle for different buckets
        colors = plt.cm.viridis(np.linspace(0, 1, end_bucket - start_bucket + 1))
        
        # Store statistics for legend
        legend_stats = []
        bucket_stats = []
        
        # Plot histograms for each bucket
        for i, bucket_id in enumerate(range(start_bucket, end_bucket + 1)):
            bucket_data = df_filtered[df_filtered['bucket'] == bucket_id]
            
            if len(bucket_data) > 0:
                # Calculate bucket statistics
                mean_food = bucket_data[f'food_actual{suffix}'].mean()
                median_food = bucket_data[f'food_actual{suffix}'].median()
                std_food = bucket_data[f'food_actual{suffix}'].std()
                n_samples = len(bucket_data)
                c3_range = f"{bucket_data[f'c3{suffix}'].min():.0f}-{bucket_data[f'c3{suffix}'].max():.0f}"
                pct_above_norm = (bucket_data[f'food_actual{suffix}'] > 
                                bucket_data[f'FoodNorm-{lifestyle}{suffix}']).mean() * 100
                
                # Plot histogram with reduced alpha for better visibility
                counts, bins, patches = ax1.hist(bucket_data[f'food_actual{suffix}'],
                                            bins=30, density=True, alpha=0.2,
                                            histtype='stepfilled', color=colors[i],
                                            label=f'Bucket {bucket_id}',
                                            range=(0, max_value))  # Set range for histogram
                
                # Plot kernel density estimate with increased line width
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(bucket_data[f'food_actual{suffix}'])
                x_range = np.linspace(0, max_value, 100)  # Use max_value for KDE range
                ax1.plot(x_range, kde(x_range), color=colors[i], linewidth=2)
                
                # Store statistics for this bucket
                bucket_stats.append({
                    'bucket_id': bucket_id,
                    'n_samples': n_samples,
                    'mean': mean_food,
                    'median': median_food,
                    'std': std_food,
                    'pct_above_norm': pct_above_norm,
                    'c3_range': c3_range
                })
                
                # Format legend entry
                legend_stats.append(
                    f'Bucket {bucket_id}:\n'
                    f'n={n_samples}\n'
                    f'mean={mean_food:.0f}\n'
                    f'median={median_food:.0f}\n'
                    f'{pct_above_norm:.1f}% above norm'
                )
        
        # Plot sample sizes in bottom subplot
        bucket_ids = [stat['bucket_id'] for stat in bucket_stats]
        sample_sizes = [stat['n_samples'] for stat in bucket_stats]
        
        ax2.bar(bucket_ids, sample_sizes, color=colors)
        ax2.set_xlabel('Bucket ID')
        ax2.set_ylabel('Sample Size')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations and styling to main plot
        ax1.set_xlabel(f'Food Expenditure {pop_type}')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis limit for main plot
        ax1.set_xlim(0, max_value)
        
        # Create title with total sample information
        total_samples = sum(stat['n_samples'] for stat in bucket_stats)
        title = (f'Distribution of Food Expenditure by C3 Bucket\n'
                f'{lifestyle.capitalize()} {pop_type}\n'
                f'Total Samples: {total_samples}')
        fig.suptitle(title, y=0.95)
        
        # Create separate legend figure on the right
        legend = ax1.legend(legend_stats, 
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        borderaxespad=0.,
                        fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Make room for the legend
        plt.subplots_adjust(right=0.85)
        
        return plt



    def create_graph7(self, df, lifestyle, per_capita=False):
        """Graph #7: Food Sufficiency Analysis by Income Level"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        
        # Filter and sort data
        mask = df[f'c3{suffix}'] <= 20000
        df_filtered = df[mask].copy().sort_values(f'c3{suffix}')
        
        # Create fixed-width buckets instead of quantile-based
        min_c3 = df_filtered[f'c3{suffix}'].min()
        max_c3 = df_filtered[f'c3{suffix}'].max()
        bucket_size = 250 #70
        n_buckets = len(df_filtered) // bucket_size  # Minimum bucket size of 70
        bucket_width = (max_c3 - min_c3) / n_buckets
        
        # Create bucket boundaries
        bucket_edges = np.arange(min_c3, max_c3 + bucket_width, bucket_width)
        df_filtered['bucket'] = pd.cut(df_filtered[f'c3{suffix}'], 
                                     bins=bucket_edges, 
                                     labels=False, 
                                     include_lowest=True)
        
        # Calculate metrics within each bucket
        bucket_stats = []
        valid_buckets = df_filtered['bucket'].unique()
        
        for bucket in valid_buckets:
            bucket_df = df_filtered[df_filtered['bucket'] == bucket]
            
            # Only process buckets with at least 70 observations
            if len(bucket_df) >= bucket_size:
                above_norm = (bucket_df[f'food_actual{suffix}'] > 
                            bucket_df[f'FoodNorm-{lifestyle}{suffix}']).mean() * 100
                above_zu = (bucket_df[f'c3{suffix}'] > 
                           bucket_df[f'ZU-{lifestyle}{suffix}']).mean() * 100
                
                mean_food_gap = (bucket_df[f'food_actual{suffix}'] - 
                               bucket_df[f'FoodNorm-{lifestyle}{suffix}']).mean()
                mean_c3_gap = (bucket_df[f'c3{suffix}'] - 
                              bucket_df[f'ZU-{lifestyle}{suffix}']).mean()
                
                stats = {
                    'c3_mean': bucket_df[f'c3{suffix}'].mean(),
                    'c3_min': bucket_df[f'c3{suffix}'].min(),
                    'c3_max': bucket_df[f'c3{suffix}'].max(),
                    'above_norm': above_norm,
                    'above_zu': above_zu,
                    'mean_food_gap': mean_food_gap,
                    'mean_c3_gap': mean_c3_gap,
                    'count': len(bucket_df)
                }
                bucket_stats.append(stats)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(bucket_stats)
        
        # Plot percentages
        width = bucket_width * 0.4
        plt.bar(stats_df['c3_mean'], stats_df['above_norm'],
                width=width, alpha=0.6, color=self.colors[0],
                label='Above Food Norm')
        plt.bar(stats_df['c3_mean'] + width, stats_df['above_zu'],
                width=width, alpha=0.6, color=self.colors[1],
                label='Above C3^')
        
        # Add trend lines
        for y_col, color, label in [
            ('above_norm', self.colors[2], 'Food Norm Trend'),
            ('above_zu', self.colors[3], 'C3^ Trend')
        ]:
            z = np.polyfit(stats_df['c3_mean'], stats_df[y_col], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(stats_df['c3_mean'].min(), 
                                stats_df['c3_mean'].max(), 100)
            plt.plot(x_trend, p(x_trend), '--', color=color, label=label)
        
        # Add percentage and N labels
        for _, row in stats_df.iterrows():
            # Add percentage and gap values above bars
            plt.text(row['c3_mean'], row['above_norm'] + 2,
                    f'{row["above_norm"]:.1f}%\nΔ={row["mean_food_gap"]:.1f}',
                    ha='center', va='bottom', fontsize=8)
            plt.text(row['c3_mean'] + width, row['above_zu'] + 2,
                    f'{row["above_zu"]:.1f}%\nΔ={row["mean_c3_gap"]:.1f}',
                    ha='center', va='bottom', fontsize=8)
            
            # Add N at bottom
            plt.text(row['c3_mean'] + width/2, -5,
                    f'N={int(row["count"])}',
                    ha='center', va='top', fontsize=7)
            
            # Add exact c3 range from data
            range_text = f'{row["c3_min"]:.0f}-{row["c3_max"]:.0f}'
            plt.text(row['c3_mean'] + width/2, -10,
                    range_text,
                    ha='center', va='top', fontsize=7, rotation=45)
        
        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Percentage of households above food norm')
        plt.title(f'Graph #7: Food deprivation by expenditure - {lifestyle.capitalize()} {pop_type},min bucket size: {bucket_size}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust y-axis to accommodate labels
        plt.ylim(-15, plt.ylim()[1] + 10)
        plt.tight_layout()
        
        return plt

    def create_graph8(self, df, lifestyle, per_capita=False):
        """Graph #8: Expenditure Adequacy Analysis"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure()
        
        # Filter and prepare data
        mask = df[f'c3{suffix}'] <= 20000
        df_filtered = df[mask].copy().sort_values(f'c3{suffix}')
        
        # Create buckets with minimum N=70
        total_rows = len(df_filtered)
        bucket_size = max(70, total_rows // 20)
        n_buckets = total_rows // bucket_size
        
        # Add bucket column first
        df_filtered['bucket'] = pd.qcut(df_filtered[f'c3{suffix}'], 
                                      q=n_buckets, labels=False)
        
        # Calculate metrics within each bucket
        bucket_stats = []
        for bucket in df_filtered['bucket'].unique():
            bucket_df = df_filtered[df_filtered['bucket'] == bucket]
            
            # Calculate percentage above ZU and mean gap
            above_zu = (bucket_df[f'c3{suffix}'] > 
                       bucket_df[f'ZU-{lifestyle}{suffix}']).mean() * 100
            mean_gap = (bucket_df[f'c3{suffix}'] - 
                       bucket_df[f'ZU-{lifestyle}{suffix}']).mean()
            
            stats = {
                'c3_mean': bucket_df[f'c3{suffix}'].mean(),
                'above_zu': above_zu,
                'mean_gap': mean_gap,
                'count': len(bucket_df)
            }
            bucket_stats.append(stats)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(bucket_stats)
        
        # Plot percentages
        width = np.min(np.diff([0] + sorted(stats_df['c3_mean'].tolist()))) * 0.8
        plt.bar(stats_df['c3_mean'], stats_df['above_zu'],
                width=width, alpha=0.6, color=self.colors[0],
                label='Above Upper Poverty Line (ZU)')
        
        # Add trend line
        z = np.polyfit(stats_df['c3_mean'], stats_df['above_zu'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(stats_df['c3_mean'].min(), 
                            stats_df['c3_mean'].max(), 100)
        plt.plot(x_trend, p(x_trend), '--', color='red', 
                label='Trend')
        
        # Add percentage and N labels
        for _, row in stats_df.iterrows():
            # Add percentage and gap value above bar
            plt.text(row['c3_mean'], row['above_zu'] + 2,
                    f'{row["above_zu"]:.1f}%\nΔ={row["mean_gap"]:.1f}',
                    ha='center', va='bottom', fontsize=8)
            
            # Add N at bottom
            plt.text(row['c3_mean'], -5,
                    f'N={int(row["count"])}',
                    ha='center', va='top', fontsize=7)
            
            # Add c3 range
            c3_range = df_filtered[df_filtered['bucket'] == row.name][f'c3{suffix}']
            range_text = f'{c3_range.min():.0f}-{c3_range.max():.0f}'
            plt.text(row['c3_mean'], -10,
                    range_text,
                    ha='center', va='top', fontsize=7, rotation=45)
        
        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Percentage Above C3^')
        plt.title(f'Graph #8: Expenditure Adequacy Analysis - {lifestyle.capitalize()} {pop_type}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust y-axis to accommodate labels
        plt.ylim(-15, plt.ylim()[1] + 10)
        plt.tight_layout()
        
        return plt

    def create_graph9(self, df, lifestyle, per_capita=False):
        """Graph #9: Detailed Food Sacrifice Analysis
        
        * Enhanced version of Graph #5
        * Additional overlays: 
            * Household size distribution
            * Total expenditure distribution
            * Moving averages for trend analysis
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Calculate sacrifice and sort data
        actual = df[f'food_actual{suffix}']
        norm = df[f'FoodNorm-{lifestyle}{suffix}']
        sacrifice = actual - norm
        
        data = pd.DataFrame({
            'sacrifice': sacrifice,
            'c3': df[f'c3{suffix}'],
            'household_size': df['persons_count']
        }).sort_values('sacrifice')
        
        households = np.arange(len(data))
        
        # Plot metrics without connecting lines
        ax1.scatter(households, data['sacrifice'], color='purple', alpha=0.5, s=10,
                   label='Food Sacrifice')
        ax2.scatter(households, data['c3'], color='green', alpha=0.5, s=10,
                   label='Total Expenditure (c3)')
        ax3.scatter(households, data['household_size'], color='blue', alpha=0.5, s=10,
                   label='Household Size')
        
        # Set labels and colors
        ax1.set_xlabel('Households (Ordered by Food Sacrifice)')
        ax1.set_ylabel('Food Sacrifice', color='purple')
        ax2.set_ylabel('Total Expenditure (c3)', color='green')
        ax3.set_ylabel('Household Size', color='blue')
        
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f'Graph #9: Detailed Food Sacrifice Analysis - {lifestyle.capitalize()} {pop_type}')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3,
                  labels1 + labels2 + labels3,
                  loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return plt
    
    def create_graph10(self, df, lifestyle, per_capita=False):
        """Graph #10: Modified Detailed Food Sacrifice Analysis with Bucketed Means
        
        Similar to Graph #9 but using bucket means for c3 and household size
        Individual points for food sacrifice, with poverty line and poor households highlighted
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Calculate sacrifice and sort data
        actual = df[f'food_actual{suffix}']
        norm = df[f'FoodNorm-{lifestyle}{suffix}']
        sacrifice = actual - norm
        
        # Create DataFrame with all metrics and sort by sacrifice
        data = pd.DataFrame({
            'sacrifice': sacrifice,
            'c3': df[f'c3{suffix}'],
            'household_size': df['persons_count'],
            'is_poor': sacrifice < 0
        }).sort_values('sacrifice')
        
        # Add indices for bucketing
        data['index'] = range(len(data))
        bucket_size = 250
        data['bucket'] = data['index'] // bucket_size
        
        # Calculate means for each bucket
        bucket_means = data.groupby('bucket').agg({
            'c3': 'mean',
            'household_size': 'mean',
            'index': 'mean',
            'is_poor': 'sum'  # Count of poor households in bucket
        }).reset_index()
        
        # Calculate poverty statistics
        poor_households = (sacrifice < 0).sum()
        poor_pct = (poor_households / len(sacrifice)) * 100
        
        # Plot shaded region for poor households
        ax1.fill_between(range(poor_households), 
                        data['sacrifice'].iloc[:poor_households],
                        0, color='red', alpha=0.1,
                        label=f'Poor Households ({poor_pct:.1f}%)')
        
        # Plot individual sacrifice points with different colors based on poverty status
        households = np.arange(len(data))
        ax1.scatter(households[data['is_poor']], 
                   data['sacrifice'][data['is_poor']],
                   color='red', alpha=0.5, s=10, label='Poor')
        ax1.scatter(households[~data['is_poor']], 
                   data['sacrifice'][~data['is_poor']],
                   color='purple', alpha=0.5, s=10, label='Non-Poor')
        
        # Add poverty line
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                   label='Poverty Line')
        ax1.axvline(x=poor_households, color='red', linestyle='--', alpha=0.3)
        
        # Plot bucketed means for c3 and household size
        ax2.plot(bucket_means['index'], bucket_means['c3'], 
                color='green', linewidth=2, label='Mean Total Expenditure (c3)')
        ax2.scatter(bucket_means['index'], bucket_means['c3'],
                   color='green', s=50, alpha=0.7)
        
        ax3.plot(bucket_means['index'], bucket_means['household_size'],
                color='blue', linewidth=2, label='Mean Household Size')
        ax3.scatter(bucket_means['index'], bucket_means['household_size'],
                   color='blue', s=50, alpha=0.7)
        
        # Set labels and colors
        ax1.set_xlabel('Households (Ordered by Food Sacrifice)')
        ax1.set_ylabel('Food Sacrifice', color='purple')
        ax2.set_ylabel('Total Expenditure (c3)', color='green')
        ax3.set_ylabel('Household Size', color='blue')
        
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f'Graph #10: Detailed Food Sacrifice Analysis (Bucketed) - {lifestyle.capitalize()} {pop_type}')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3,
                  labels1 + labels2 + labels3,
                  loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return plt
    
    
    def create_graph11(self, df, lifestyle, per_capita=False):
        """Graph #11: Food Sufficiency by Total Expenditure"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        plt.figure(figsize=(12,8))
        
        # Calculate percentage differences
        df['food_pct_diff'] = (df[f'food_actual{suffix}'] - df[f'FoodNorm-{lifestyle}{suffix}']) / df[f'FoodNorm-{lifestyle}{suffix}'] * 100
        df['c3_pct_diff'] = (df[f'c3{suffix}'] - df[f'ZU-{lifestyle}{suffix}']) / df[f'ZU-{lifestyle}{suffix}'] * 100
        
        # Filter x-axis to show only values under 300%
        df = df[df['c3_pct_diff'] <= 450]
        
        # Sort by C3 percentage difference 
        df = df.sort_values('c3_pct_diff')
        
        # Create fixed-width buckets
        min_c3_diff = df['c3_pct_diff'].min()
        max_c3_diff = df['c3_pct_diff'].max()
        bucket_size = 250 # Min bucket size of 70
        n_buckets = len(df) // bucket_size  
        bucket_width = (max_c3_diff - min_c3_diff) / n_buckets
        
        bucket_edges = np.arange(min_c3_diff, max_c3_diff + bucket_width, bucket_width)
        df['bucket'] = pd.cut(df['c3_pct_diff'], bins=bucket_edges, labels=False, include_lowest=True)
        
        # Calculate metrics per bucket
        bucket_stats = df.groupby('bucket').agg(
            food_pct_above_norm=('food_pct_diff', lambda x: (x > 0).mean() * 100),
            c3_pct_diff_mean=('c3_pct_diff', 'mean'), 
            c3_pct_diff_min=('c3_pct_diff', 'min'),
            c3_pct_diff_max=('c3_pct_diff', 'max'),
            count=('c3_pct_diff', 'count')
        ).reset_index()
        
        # Bar plot of percentage above food norm
        plt.bar(bucket_stats['c3_pct_diff_mean'], bucket_stats['food_pct_above_norm'], 
                width=bucket_width*0.8, alpha=0.6, color=self.colors[0])
        
        # Add annotations
        for x, y, count, c3min, c3max in zip(
            bucket_stats['c3_pct_diff_mean'], bucket_stats['food_pct_above_norm'], 
            bucket_stats['count'], bucket_stats['c3_pct_diff_min'], bucket_stats['c3_pct_diff_max']
        ):
            plt.annotate(f'{y:.1f}%', xy=(x,y), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
            
            plt.annotate(f'{x/100:.1f}', xy=(x,y-3), 
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6)
        
        plt.xlabel(f'Mean % Difference from C3^ ((C3-C3^)/C3^) normalized')
        plt.ylabel('% of Households Above Food Norm')  
        plt.title(f'Food Sufficiency Analysis - {lifestyle.capitalize()}\n Minimum Bucket Size: {bucket_size} Households')
        
        # Add reference lines
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='C3^ Threshold')
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='All Above Food Norm')  
        plt.legend()
        
        plt.tight_layout()
    
        return plt
    

    def create_graph12(self, df, lifestyle, per_capita=True):
        """Graph #12: Sorted Percentage Differences Analysis with Bucket Means
        
        X-axis: Buckets sorted by mean (C3-C3^)/C3^ where C3^===ZU
        Y-axes: (FoodActual-FoodNorm)/FoodNorm percentage, mean C3 difference percentage, and mean household size
        Bucket size: 250 households
        """
        suffix = '_per_capita'  # Always per capita for this graph
        
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Calculate percentage differences
        data = pd.DataFrame({
            'food_pct_diff': (df[f'food_actual{suffix}'] - df[f'FoodNorm-{lifestyle}{suffix}']) / 
                            df[f'FoodNorm-{lifestyle}{suffix}'] * 100,
            'c3_pct_diff': (df[f'c3{suffix}'] - df[f'ZU-{lifestyle}{suffix}']) / 
                          df[f'ZU-{lifestyle}{suffix}'] * 100,
            'household_size': df['persons_count']
        })
        
        # Sort by food percentage difference and create buckets
        data = data.sort_values('food_pct_diff')
        data['bucket'] = np.arange(len(data)) // 250
        
        # Calculate bucket means
        bucket_means = data.groupby('bucket').agg({
            'food_pct_diff': 'mean',
            'c3_pct_diff': 'mean',
            'household_size': 'mean'
        }).reset_index()
        
        # Calculate bucket statistics
        bucket_means['bucket_center'] = bucket_means['bucket'] * 250 + 125  # Center of each bucket
        bucket_means['poor_count'] = data.groupby('bucket')['food_pct_diff'].apply(
            lambda x: (x < 0).sum())
        
        # Create connected line plots with points
        line1 = ax1.plot(bucket_means['bucket_center'], bucket_means['food_pct_diff'],
                        '-o', color='purple', linewidth=1, markersize=3,
                        label='Mean Food Norm % Diff')
        line2 = ax2.plot(bucket_means['bucket_center'], bucket_means['c3_pct_diff'],
                        '-o', color='green', linewidth=1, markersize=3,
                        label='Mean C3^ % Diff')
        line3 = ax3.plot(bucket_means['bucket_center'], bucket_means['household_size'],
                        '-o', color='blue', linewidth=1, markersize=3,
                        label='Mean Household Size')
        
        
        # Add reference line at 0%
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                   label='Threshold')
        
        # Set labels and colors
        ax1.set_xlabel('Households (Bucketed and Ordered by (FoodActual-FoodNorm)/FoodNorm)')
        ax1.set_ylabel('(FoodActual-FoodNorm)/FoodNorm percentage', color='purple')
        ax2.set_ylabel('C3^ % Difference', color='green')
        ax3.set_ylabel('Household Size', color='blue')
        
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        # Calculate overall statistics
        total_poor = (data['food_pct_diff'] < 0).sum()
        poor_pct = (total_poor / len(data)) * 100
        
        plt.title(f'Food sacrifice - {lifestyle.capitalize()} normalized\n'
                 f'Bucket Size: 250 Households, Total Poor: {poor_pct:.1f}%')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1))
        
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
        
        # Update per-capita calculations using sqrt of household size
        '''
        for col in ['food_actual', 'c3']:
            self.df[f'{col}_per_capita'] = self.df[col] / np.sqrt(self.df['persons_count'])
        
        for lifestyle in ['active', 'sedentary']:
            for col in ['FoodNorm', 'ZU', 'ZL']:
                self.df[f'{col}-{lifestyle}_per_capita'] = (
                    self.df[f'{col}-{lifestyle}'] / np.sqrt(self.df['persons_count'])
                )
        '''
        # Define plot configurations
        plots = [
            
            #('graph1', 'create_graph1'),
            #('graph2', 'create_graph2'),
            #('graph3', 'create_graph3'),
            #('graph4', 'create_graph4'),
            #('graph5', 'create_graph5'),
            #('graph6', 'create_graph6'),
            
            ('graph7', 'create_graph7'),
            ('graph7-1', 'plot_bucket_histogram'),
            
            #('graph8', 'create_graph8'),
            #('graph9', 'create_graph9'),
            #('graph10', 'create_graph10'), 
            #('graph11', 'create_graph11'),
            #('graph12', 'create_graph12')
            
        ]
        
        # Generate all plots for both lifestyles and metrics
        for lifestyle in ['active', 'sedentary']:
            print(f"\nProcessing {lifestyle} lifestyle plots:")
            #for per_capita in [False, True]:
            per_capita = True
            metric_type = 'per_capita' if per_capita else 'household'
            print(f"\n  Generating {metric_type} metrics:")
                
            try:
                for plot_name, plot_method in plots:
                    print(f"    - Creating {plot_name}...")
                    plt = getattr(visualizer, plot_method)(
                        self.df, lifestyle, per_capita)
                    visualizer.save_plot(f'{plot_name}_{lifestyle}_{metric_type}')
                
            except Exception as e:
                print(f"Error generating {metric_type} plots for {lifestyle} "
                          f"lifestyle: {str(e)}")
        
        print(f"\nAll plots have been saved in: {plot_dir}")
    
    # Add the method to the analyzer class
    FamilyGroupAnalyzer.plot_and_save_groups = plot_and_save_groups