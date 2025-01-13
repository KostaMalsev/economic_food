import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer


class DetailedVisualizer(BaseVisualizer):
    
    def create_graph9(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Detailed Food Sacrifice Analysis"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        # Calculate food sacrifice
        df = df.copy()
        df['sacrifice'] = (df[f'food_actual{suffix}'] -
                           df[f'FoodNorm-{lifestyle}{suffix}'])

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
        ax1.fill_between(range(sacrificing),
                         df_sorted['sacrifice'].iloc[:sacrificing],
                         0, color='red', alpha=0.1, label='Food Sacrificing')

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

    def create_graph10(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Modified Detailed Food Sacrifice Analysis with Bucketed Means"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        df = df.copy()
        df['sacrifice'] = (df[f'food_actual{suffix}'] -
                           df[f'FoodNorm-{lifestyle}{suffix}'])
        df['is_poor'] = df['sacrifice'] < 0

        # Create buckets based on sacrifice
        df_sorted = df.sort_values('sacrifice')
        bucket_size = 250
        n_buckets = len(df_sorted) // bucket_size
        df_sorted['bucket'] = pd.qcut(
            range(len(df_sorted)), n_buckets, labels=False)

        # Calculate bucket statistics
        bucket_stats = df_sorted.groupby('bucket').agg({
            'sacrifice': aggregation,
            f'c3{suffix}': aggregation,
            'persons_count': aggregation,
            'is_poor': 'sum'
        }).reset_index()

        bucket_stats['bucket_center'] = bucket_stats['bucket'] * \
            bucket_size + bucket_size / 2

        # Create plot with three y-axes
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))

        # Calculate poverty statistics
        poor_households = df['is_poor'].sum()
        poor_pct = (poor_households / len(df)) * 100

        # Plot shaded region for poor households
        ax1.fill_between(range(poor_households),
                         df_sorted['sacrifice'].iloc[:poor_households],
                         0, color='red', alpha=0.1,
                         label=f'Poor Households ({poor_pct:.1f}%)')

        # Plot sacrifice points with different colors based on poverty status
        poor_mask = df_sorted['is_poor']
        households = range(len(df_sorted))

        ax1.scatter(households, df_sorted['sacrifice'],
                    c=poor_mask.map({True: 'red', False: 'purple'}),
                    alpha=0.5, s=10)

        # Add reference lines
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Poverty Line')
        ax1.axvline(x=poor_households, color='red', linestyle='--', alpha=0.3)

        # Plot bucketed means
        line2 = ax2.plot(bucket_stats['bucket_center'],
                         bucket_stats[f'c3{suffix}'],
                         color='green',
                         linewidth=2,
                         label='Mean Total Expenditure (c3)')
        line3 = ax3.plot(
            bucket_stats['bucket_center'],
            bucket_stats['persons_count'],
            color='blue',
            linewidth=2,
            label='Mean Household Size')

        # Set labels and colors
        ax1.set_xlabel('Households (Ordered by Food Sacrifice)')
        ax1.set_ylabel('Food Sacrifice', color='purple')
        ax2.set_ylabel('Total Expenditure (c3)', color='green')
        ax3.set_ylabel('Household Size', color='blue')

        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')

        plt.title(
            f'Detailed Food Sacrifice Analysis - {lifestyle.capitalize()} {pop_type}\n' f'Bucket Size: {bucket_size} Households')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + line2 + line3,
                   labels1 + ['Mean Total Expenditure (c3)', 'Mean Household Size'],
                   loc='upper left', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        return plt
