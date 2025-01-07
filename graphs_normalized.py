import numpy as np
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer


class NormalizedVisualizer(BaseVisualizer):

    def create_graph11(self, df, lifestyle, per_capita=False):
        """Food Sufficiency by Total Expenditure (Normalized)"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        df = df.copy()

        # Calculate normalized differences
        df['food_pct_diff'] = ((df[f'food_actual{suffix}'] -
                                df[f'FoodNorm-{lifestyle}{suffix}']) /
                               df[f'FoodNorm-{lifestyle}{suffix}'] *
                               100)
        df['c3_pct_diff'] = ((df[f'c3{suffix}'] -
                              df[f'ZU-{lifestyle}{suffix}']) /
                             df[f'ZU-{lifestyle}{suffix}'] *
                             100)

        # Filter extreme values
        df = df[df['c3_pct_diff'] <= 450]

        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, 'c3_pct_diff', bucket_size=70
        )

        metrics = {
            'food_pct_above_norm': {
                'columns': ['food_pct_diff'],
                'func': lambda x: (x['food_pct_diff'] > 0).mean() * 100
            },
            'c3_pct_diff': {
                'columns': ['c3_pct_diff'],
                'func': lambda x: x['c3_pct_diff'].mean()
            },
            'food_pct_diff': {
                'columns': ['food_pct_diff'],
                'func': lambda x: x['food_pct_diff'].mean()
            },
            'count': {
                'columns': ['c3_pct_diff'],
                'func': len
            },
            'c3_pct_above_zu': {
                'columns': ['c3_pct_diff'],
                'func': lambda x: (x['c3_pct_diff'] > 0).mean() * 100
            },
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)

        plt.figure(figsize=(12, 8))

        # Bar plot of percentage above food norm
        plt.bar(stats['c3_pct_diff'], stats['food_pct_above_norm'],
                width=bucket_width * 0.8, alpha=0.6, color=self.colors[0])
        plt.bar(stats['c3_pct_diff'], stats['c3_pct_above_zu'],
                width=bucket_width * 0.8, alpha=0.6, color=self.colors[1])

        # Add annotations
        for _, row in stats.iterrows():
            plt.annotate(f'{row["food_pct_above_norm"]:.1f}%',
                         xy=(row['c3_pct_diff'], row['food_pct_above_norm']),
                         xytext=(0, 5), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)

            plt.annotate(f'{row["food_pct_diff"] / 100:.1f}',
                         xy=(row['c3_pct_diff'], row['food_pct_above_norm'] - 3),
                         xytext=(0, 2), textcoords='offset points',
                         ha='center', va='bottom', fontsize=6)
            
            plt.annotate(f'({int(row["count"])})',
                         xy=(row['c3_pct_diff'], row['food_pct_above_norm'] - 5),
                         xytext=(0, 1), textcoords='offset points',
                         ha='center', va='bottom', fontsize=6)
            

        plt.xlabel('Mean % Difference from Upper Poverty Line ((C3-ZU)/ZU)')
        plt.ylabel('% of Households Above Food Norm')
        plt.title(f'Food Sufficiency Analysis - {lifestyle.capitalize()}\n'
                  f'Minimum Bucket Size: {min(stats["count"])} Households')

        # Add reference lines
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7,
                    label='Upper Poverty Line Threshold')
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7,
                    label='All Above Food Norm')
        plt.legend()

        plt.tight_layout()
        return plt
    



    def create_graph12(self, df, lifestyle, per_capita=True):
        """Sorted Percentage Differences Analysis with Bucket Means"""
        suffix = '_per_capita'  # Always per capita for this graph
        
        df = df.copy()
        df['food_pct_diff'] = ((df[f'food_actual{suffix}'] - 
                                df[f'FoodNorm-{lifestyle}{suffix}']) /
                            df[f'FoodNorm-{lifestyle}{suffix}'] * 
                            100)
        df['c3_pct_diff'] = ((df[f'c3{suffix}'] - 
                            df[f'ZU-{lifestyle}{suffix}']) /
                            df[f'ZU-{lifestyle}{suffix}'] * 
                            100)
        
        bucket_size = 70
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, 'food_pct_diff', bucket_size=bucket_size
        )
        
        metrics = {
            'food_pct_diff': {
                'columns': ['food_pct_diff'],
                'func': lambda x: x['food_pct_diff'].mean()
            },
            'c3_pct_diff': {
                'columns': ['c3_pct_diff'],
                'func': lambda x: x['c3_pct_diff'].mean()
            },
            'household_size': {
                'columns': ['persons_count'],
                'func': lambda x: x['persons_count'].mean()
            },
            'poor_count': {
                'columns': ['food_pct_diff'],
                'func': lambda x: (x['food_pct_diff'] < 0).sum()
            },
            'sample_count': {  # Add new metric for sample counts
                'columns': ['food_pct_diff'],
                'func': lambda x: len(x)
            }
        }
        
        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)
        
        # Create plot with three y-axes
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        bucket_centers = np.arange(len(stats)) * 250 + 125
        
        # Create connected line plots
        line1 = ax1.plot(bucket_centers, stats['food_pct_diff'],
                        '-o', color='purple', linewidth=1, markersize=3,
                        label='Mean Food Norm % Diff')
        line2 = ax2.plot(bucket_centers, stats['c3_pct_diff'],
                        '-o', color='green', linewidth=1, markersize=3,
                        label='Mean Upper Poverty Line % Diff')
        line3 = ax3.plot(bucket_centers, stats['household_size'],
                        '-o', color='blue', linewidth=1, markersize=3,
                        label='Mean Household Size')
        
        # Add reference line at 0%
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Threshold')
        y_min = ax1.get_ylim()[0]
        y_max = ax1.get_ylim()[1]
        
        text_y_pos = y_min + (y_max - y_min) * 0.1

        # Add sample count annotations
        for i, count in enumerate(stats['sample_count']):
            ax1.text(bucket_centers[i], text_y_pos,
                    f'n={count}',
                    rotation=45,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=8)
        
        # Set labels
        ax1.set_xlabel(
            'Households (Bucketed and Ordered by (Food-Norm)/Norm %)')
        ax1.set_ylabel('Food Norm % Difference', color='purple')
        ax2.set_ylabel('Upper Poverty Line % Difference', color='green')
        ax3.set_ylabel('Household Size', color='blue')
        
        # Set colors
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        # Calculate total poor percentage
        total_poor = (df['food_pct_diff'] < 0).sum()
        poor_pct = (total_poor / len(df)) * 100
        
        plt.title(f'Normalized Food Sacrifice Analysis - {lifestyle.capitalize()}\n'
                f'Bucket Size: {bucket_size} Households, Total Poor: {poor_pct:.1f}%')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return plt