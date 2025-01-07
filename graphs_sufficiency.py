import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bucketing_helper as bh


from base_visualizer import BaseVisualizer


class SufficiencyVisualizer(BaseVisualizer):

    def create_graph7(self, df, lifestyle, per_capita=False):
        """Food Sufficiency Analysis by Income Level"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', max_value=40000, bucket_size=70
        )

        metrics = {
            'above_norm': {
                'columns': [f'food_actual{suffix}', f'FoodNorm-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'food_actual{suffix}'] >
                                   x[f'FoodNorm-{lifestyle}{suffix}']).mean() * 100
            },
            'above_zu': {
                'columns': [f'c3{suffix}', f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'c3{suffix}'] >
                                   x[f'ZU-{lifestyle}{suffix}']).mean() * 100
            },
            'mean_food_gap': {
                'columns': [f'food_actual{suffix}', f'FoodNorm-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'food_actual{suffix}'] -
                                   x[f'FoodNorm-{lifestyle}{suffix}']).mean()
            },
            'mean_c3_gap': {
                'columns': [f'c3{suffix}', f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'c3{suffix}'] -
                                   x[f'ZU-{lifestyle}{suffix}']).mean()
            },
            'c3_mean': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: x[f'c3{suffix}'].mean()
            },
            'count': {
                'columns': [f'c3{suffix}'],
                'func': len
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)

        plt.figure()
        width = bucket_width * 0.4

        # Plot percentages
        plt.bar(stats['c3_mean'], stats['above_norm'],
                width=width, alpha=0.6, color=self.colors[0],
                label='Above Food Norm')
        plt.bar(stats['c3_mean'] + width, stats['above_zu'],
                width=width, alpha=0.6, color=self.colors[1],
                label='Above C3^')

        # Add trend lines
        for y_col, color, label in [
            ('above_norm', self.colors[2], 'Food Norm Trend'),
            ('above_zu', self.colors[3], 'C3^ Trend')
        ]:
            z = np.polyfit(stats['c3_mean'], stats[y_col], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(stats['c3_mean'].min(),
                                  stats['c3_mean'].max(), 100)
            plt.plot(x_trend, p(x_trend), '--', color=color, label=label)

        # Add labels
        for _, row in stats.iterrows():
            plt.text(row['c3_mean'], row['above_norm'] + 2,
                     f'{row["above_norm"]:.1f}%\nΔ={row["mean_food_gap"]:.1f}',
                     ha='center', va='bottom', fontsize=8)
            plt.text(row['c3_mean'] + width, row['above_zu'] + 2,
                     f'{row["above_zu"]:.1f}%\nΔ={row["mean_c3_gap"]:.1f}',
                     ha='center', va='bottom', fontsize=8)
            plt.text(row['c3_mean'] + width / 2, -5,
                     f'N={int(row["count"])}',
                     ha='center', va='top', fontsize=7)

        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Percentage of households above threshold')
        plt.title(
            f'Food Sufficiency Analysis - {lifestyle.capitalize()} {pop_type}')
        plt.legend()

        # Adjust y-axis to accommodate labels
        plt.ylim(-15, plt.ylim()[1] + 10)
        plt.tight_layout()

        return plt

    def create_graph8(self, df, lifestyle, per_capita=False):
        """Expenditure Adequacy Analysis"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', max_value=20000, bucket_size=70
        )

        metrics = {
            'above_zu': {
                'columns': [f'c3{suffix}', f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'c3{suffix}'] >
                                   x[f'ZU-{lifestyle}{suffix}']).mean() * 100
            },
            'mean_gap': {
                'columns': [f'c3{suffix}', f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'c3{suffix}'] -
                                   x[f'ZU-{lifestyle}{suffix}']).mean()
            },
            'c3_mean': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: x[f'c3{suffix}'].mean()
            },
            'count': {
                'columns': [f'c3{suffix}'],
                'func': len
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)

        plt.figure()

        # Plot percentages
        plt.bar(stats['c3_mean'], stats['above_zu'],
                width=bucket_width * 0.8, alpha=0.6, color=self.colors[0],
                label='Above Upper Poverty Line (ZU)')

        # Add trend line
        z = np.polyfit(stats['c3_mean'], stats['above_zu'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(stats['c3_mean'].min(),
                              stats['c3_mean'].max(), 100)
        plt.plot(x_trend, p(x_trend), '--', color='red',
                 label='Trend')

        # Add labels
        for _, row in stats.iterrows():
            plt.text(row['c3_mean'], row['above_zu'] + 2,
                     f'{row["above_zu"]:.1f}%\nΔ={row["mean_gap"]:.1f}',
                     ha='center', va='bottom', fontsize=8)
            plt.text(row['c3_mean'], -5,
                     f'N={int(row["count"])}',
                     ha='center', va='top', fontsize=7)

        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Percentage Above Upper Poverty Line')
        plt.title(
            f'Expenditure Adequacy Analysis - {lifestyle.capitalize()} {pop_type}')
        plt.legend()

        # Adjust y-axis to accommodate labels
        plt.ylim(-15, plt.ylim()[1] + 10)
        plt.tight_layout()

        return plt




    

    

    
    def create_graph70(
            self,
            df,
            lifestyle,
            per_capita=True,
            start_bucket=0,
            end_bucket=9,
            max_value=7000):
        """
        Plot histograms of food expenditure values for each bucket
        
        Parameters:
        - df: DataFrame containing the data
        - lifestyle: 'active' or 'sedentary'
        - per_capita: Boolean for per capita metrics
        - start_bucket: Starting bucket ID
        - end_bucket: Ending bucket ID
        - max_value: Maximum value for x-axis (default: 7000)
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = 'Per Capita' if per_capita else 'Household'

        # Create figure with two subplots - main histogram and sample sizes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12),
                                       gridspec_kw={'height_ratios': [4, 1]})

        # Filter data by max_value first
        mask = df[f'c3{suffix}'] <= max_value
        df_filtered = df[mask].copy()

        # Use BucketingHelper to create buckets
        bucket_helper = bh.BucketingHelper()
        df_bucketed, bucket_width = bucket_helper.create_fixed_width_buckets(
            df_filtered,
            value_column=f'c3{suffix}',
            max_value=max_value,
            bucket_size=200,  # Target size for each bucket
            min_samples=70    # Minimum samples before merging buckets
        )

        # Set up color cycle for different buckets
        colors = plt.cm.viridis(
            np.linspace(
                0,
                1,
                end_bucket -
                start_bucket +
                1))

        # Store statistics for legend
        legend_stats = []
        bucket_stats = []

        # Plot histograms for each bucket
        for i, bucket_id in enumerate(range(start_bucket, end_bucket + 1)):
            bucket_data = df_bucketed[df_bucketed['bucket'] == bucket_id]

            if len(bucket_data) > 0:
                # Calculate bucket statistics
                mean_food = bucket_data[f'food_actual{suffix}'].mean()
                median_food = bucket_data[f'food_actual{suffix}'].median()
                std_food = bucket_data[f'food_actual{suffix}'].std()
                n_samples = len(bucket_data)
                c3_range = f"{bucket_data[f'c3{suffix}'].min():.0f}-{bucket_data[f'c3{suffix}'].max():.0f}"
                pct_above_norm = (bucket_data[f'food_actual{suffix}'] >
                                  bucket_data[f'FoodNorm-{lifestyle}{suffix}']).mean() * 100

                # Get food expenditure data
                food_data = bucket_data[f'food_actual{suffix}'].values

                # Plot histogram
                ax1.hist(food_data, bins=30, 
                        alpha=0.6,  # Increased alpha for better visibility
                        color=colors[i],
                        label=f'Bucket {bucket_id}',
                        range=(0, max_value),
                        density=True,  # Normalize histogram
                        histtype='bar',
                        edgecolor='black',  # Add edge color for better separation
                        linewidth=0.5)      # Thin edges for histograms

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

                # Format legend entry with C3 range
                legend_stats.append(
                    f'Bucket {bucket_id} (C3: {c3_range}):\n'
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