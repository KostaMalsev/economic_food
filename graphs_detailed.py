import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer


class DetailedVisualizer(BaseVisualizer):
    
    def create_graph9(self, df, lifestyle, per_capita=False, aggregation='mean'):
        """Detailed Food Sacrifice Analysis"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        # Calculate food sacrifice
        df = df.copy()
        df['sacrifice'] = (df[f'FoodNorm-{lifestyle}{suffix}'] - df[f'food_actual{suffix}'])

        # Create buckets based on sacrifice
        df_sorted = df.sort_values('sacrifice')
        df_sorted['bucket_index'] = range(len(df_sorted))

        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(
            15, 15), sharex=True, height_ratios=[3, 1, 1])

        # Plot 1: Food sacrifice
        sacrificing = (df_sorted['sacrifice'] < 0).sum()
        sacrificing_pct = (sacrificing / len(df_sorted)) * 100

        ax1.scatter(
            df_sorted['bucket_index'],
            df_sorted['sacrifice'],
            color=self.colors[0],
            alpha=0.5,
            s=10,
            label='Food Sacrifice')

        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Zero Point')
        ax1.axvline(x=sacrificing, color='red', linestyle='--', alpha=0.3,
                    label=f'Sacrifice Line ({sacrificing_pct:.1f}%)')

        # Fill area for food sacrificing households
        ax1.fill_between(df_sorted['bucket_index'], df_sorted['sacrifice'], 0, where=df_sorted['sacrifice'] < 0, color='red', alpha=0.1, label='Food Sacrifice')

        # Plot 2: Total expenditure
        ax2.scatter(df_sorted['bucket_index'], df_sorted[f'c3{suffix}'],
                    color=self.colors[1], alpha=0.5, s=10,
                    label='Total Expenditure (c3)')

        # Plot 3: Household size
        ax3.scatter(df_sorted['bucket_index'], df_sorted['persons_count'],
                    color=self.colors[2], alpha=0.5, s=10,
                    label='Household Size')

        # Set labels and titles
        ax1.set_ylabel('Food Sacrifice')
        ax2.set_ylabel('Total Expenditure')
        ax3.set_ylabel('Household Size')
        ax3.set_xlabel('Households (Ordered by Food Sacrifice)')

        plt.suptitle(
            f'Detailed Food Sacrifice Analysis - {lifestyle.capitalize()} {pop_type}')

        # Add legends
        for ax in [ax1, ax2, ax3]:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt


    def create_graph10(self, df, lifestyle, per_capita=False, aggregation='mean'):
        """Modified Detailed Food Sacrifice Analysis with Bucketed Means using metrics dictionary"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        # Create a copy and calculate sacrifice
        df = df.copy()
        df['sacrifice'] = (-df[f'FoodNorm-{lifestyle}{suffix}'] + df[f'food_actual{suffix}'])
        df['is_poor'] = df['sacrifice'] < 0
        
        # Create fixed-width buckets
        bucket_size = 70
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, 'sacrifice', bucket_size=bucket_size, min_samples=bucket_size
        )
        
        # Define metrics dictionary
        metrics = {
            'sacrifice': {
                'columns': ['sacrifice'],
                'func': lambda x: x['sacrifice'].mean() if aggregation == 'mean' else x['sacrifice'].median()
            },
            'c3': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: x[f'c3{suffix}'].mean() if aggregation == 'mean' else x[f'c3{suffix}'].median()
            },
            'household_size': {
                'columns': ['persons_count'],
                'func': lambda x: x['persons_count'].mean() if aggregation == 'mean' else x['persons_count'].median()
            },
            'age': {
                'columns': ['mean_age'],
                'func': lambda x: x['mean_age'].mean() if aggregation == 'mean' else x['mean_age'].median()
            },
            'poor_count': {
                'columns': ['is_poor'],
                'func': lambda x: x['is_poor'].sum()
            },
            'sample_count': {
                'columns': ['sacrifice'],
                'func': lambda x: len(x)
            }
        }
        
        # Calculate bucket statistics
        stats = self.helper.calculate_bucket_stats(df_bucketed, metrics=metrics)
        
        # Calculate cumulative households for x-axis
        stats['cumulative_households'] = stats['sample_count'].cumsum()
        
        # Create plot with four y-axes
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()
        
        # Set positions for multiple right y-axes
        ax3.spines['right'].set_position(('outward', 60))
        ax4.spines['right'].set_position(('outward', 120))
        
        # Calculate poverty statistics
        poor_households = df['is_poor'].sum()
        poor_pct = (poor_households / len(df)) * 100
        
        # Use cumulative households for x-axis positions
        bucket_centers = stats['cumulative_households']
        
        # Add shaded regions for poor households
        legend_added = False
        for i, row in stats.iterrows():
            if row['sacrifice'] < 0:
                label = f'Poor Households ({poor_pct:.1f}%)' if not legend_added else None
                ax1.fill_betweenx([min(0, row['sacrifice']), 0],
                                bucket_centers[i] - bucket_size/2,
                                bucket_centers[i] + bucket_size/2,
                                color='red', alpha=0.1, label=label)
                legend_added = True
        
        # Calculate y-axis limits considering both food sacrifice and c3
        min_y = min(min(stats['sacrifice']), min(stats['c3']))
        max_y = max(max(stats['sacrifice']), max(stats['c3']))
        
        # Add some padding to the limits
        y_padding = (max_y - min_y) * 0.1
        y_min = min_y - y_padding
        y_max = max_y + y_padding
        
        # Set the same y-axis limits for both food sacrifice and c3
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        # Ensure gridlines are visible from both axes but not duplicated
        ax1.grid(True, alpha=0.3)
        ax2.grid(False)  # Turn off grid for second axis
        ax3.grid(False)  # Turn off grid for third axis
        ax4.grid(False)  # Turn off grid for fourth axis
        
        # Plot connected sacrifice line with points
        sacrifice_line = ax1.plot(bucket_centers, stats['sacrifice'],
                                color='purple', linewidth=1, alpha=0.8,
                                label='Food Sacrifice')
        ax1.scatter(bucket_centers, stats['sacrifice'],
                    c=['red' if s < 0 else 'purple' for s in stats['sacrifice']],
                    alpha=0.5, s=10)
        
        # Add vertical lines at bucket boundaries
        for center in bucket_centers:
            ax1.axvline(x=center, color='gray', linestyle='-', alpha=0.1)
        
        # Add reference lines
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Poverty Line')
        
        # Plot bucketed statistics
        line2 = ax2.plot(bucket_centers, stats['c3'],
                        color='green', linewidth=2,
                        label=f'{aggregation} total expenditure (c3)')
        line3 = ax3.plot(bucket_centers, stats['household_size'],
                        color='blue', linewidth=2,
                        label=f'{aggregation} household size')
        line4 = ax4.plot(bucket_centers, stats['age'],
                        color='orange', linewidth=2,
                        label=f'{aggregation} age')
        
        # Add sample count annotations
        y_min = min(stats['sacrifice'].min(), 0)
        y_max = max(stats['sacrifice'].max(), 0)
        
        for i, (count, cum_count) in enumerate(zip(stats['sample_count'], stats['cumulative_households'])):
            ax1.text(cum_count, y_min + (y_max - y_min) * 0.1,
                    f'n={count}\nTotal={cum_count}',
                    rotation=45,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=8)
        
        # Set labels and colors
        ax1.set_xlabel('Cumulative Number of Households')
        ax1.set_ylabel('Food Sacrifice', color='purple')
        ax2.set_ylabel('Total Expenditure (c3)', color='green')
        ax3.set_ylabel('Household Size', color='blue')
        ax4.set_ylabel('Mean Age', color='orange')
        
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax4.tick_params(axis='y', labelcolor='orange')
        
        plt.title(
            f'Food Sacrifice Analysis - {lifestyle.capitalize()} {pop_type}\n'
            f'Bucket Size: {bucket_size} Households'
        )
        
        # Combine legends
        lines = sacrifice_line + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return plt