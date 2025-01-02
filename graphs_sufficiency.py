import numpy as np
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer

class SufficiencyVisualizer(BaseVisualizer):
    def create_graph7(self, df, lifestyle, per_capita=False):
        """Food Sufficiency Analysis by Income Level"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', max_value=40000, bucket_size=100
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
        
        stats = self.helper.calculate_bucket_stats(df_bucketed, metrics=metrics)
        
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
            plt.text(row['c3_mean'] + width/2, -5,
                    f'N={int(row["count"])}',
                    ha='center', va='top', fontsize=7)
        
        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Percentage of households above threshold')
        plt.title(f'Food Sufficiency Analysis - {lifestyle.capitalize()} {pop_type}')
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
            df, f'c3{suffix}', max_value=20000, bucket_size=100
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
        
        stats = self.helper.calculate_bucket_stats(df_bucketed, metrics=metrics)
        
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
        plt.title(f'Expenditure Adequacy Analysis - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        # Adjust y-axis to accommodate labels
        plt.ylim(-15, plt.ylim()[1] + 10)
        plt.tight_layout()
        
        return plt