import numpy as np
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

class ExpenditureVisualizer(BaseVisualizer):

    def create_graph1(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Food Expenditure vs Total Expenditure Analysis"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        max_value = 20000
        
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', max_value,bucket_size=70, min_samples=70
        )

        metrics = {
            'food_actual': {
                'columns': [f'food_actual{suffix}'],
                'func': lambda x: x[f'food_actual{suffix}'].mean() if aggregation == 'mean' else x[f'food_actual{suffix}'].median()
            },
            'food_norm': {
                'columns': [f'FoodNorm-{lifestyle}{suffix}'],
                'func': lambda x: x[f'FoodNorm-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'FoodNorm-{lifestyle}{suffix}'].median()
            },
            'gap': {
                'columns': [f'food_actual{suffix}', f'FoodNorm-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'food_actual{suffix}'] - x[f'FoodNorm-{lifestyle}{suffix}']).mean() if aggregation == 'mean' else (x[f'food_actual{suffix}'] - x[f'FoodNorm-{lifestyle}{suffix}']).median()
            },
            'c3': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: x[f'c3{suffix}'].mean() if aggregation == 'mean' else x[f'c3{suffix}'].median()
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)

        plt.figure(figsize=(10, 6))

        # Plot metrics with cleaner code
        for col, color, label in [
            ('food_actual', self.colors[0], 'Food Expenditure'),
            ('food_norm', self.colors[1], 'Food Norm'),
            ('gap', self.colors[2], 'Gap')
        ]:
            plt.scatter(stats['c3'], stats[col], s=20, alpha=0.6,
                        color=color, label=label)

            # Add trend lines
            z = np.polyfit(stats['c3'], stats[col], 1)
            p = np.poly1d(z)
            plt.plot(stats['c3'], p(stats['c3']), '--', color=color, alpha=0.8)

        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Value')
        plt.title(
            f'Food Expenditure vs Total Expenditure - {lifestyle.capitalize()} {pop_type}')
        plt.legend()

        return plt

    def create_graph2(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Total Expenditure vs Upper Poverty Line Comparison"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', 20000, 70
        )

        metrics = {
            'c3': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: x[f'c3{suffix}'].mean() if aggregation == 'mean' else x[f'c3{suffix}'].median()
            },
            'zu': {
                'columns': [f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: x[f'ZU-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'ZU-{lifestyle}{suffix}'].median()
            },
            'gap': {
                'columns': [f'c3{suffix}', f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: (x[f'c3{suffix}'] - x[f'ZU-{lifestyle}{suffix}']).mean() if aggregation == 'mean' else (x[f'c3{suffix}'] - x[f'ZU-{lifestyle}{suffix}']).median()
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)

        plt.figure()
        for col, color, label in [
            ('c3', self.colors[0], 'Total Expenditure (c3)'),
            ('zu', self.colors[1], 'Upper Poverty Line (ZU)'),
            ('gap', self.colors[2], 'Gap')
        ]:
            plt.scatter(stats['c3'], stats[col], alpha=0.5,
                        color=color, label=label)

        plt.xlabel(f'Total Expenditure (c3) {pop_type}')
        plt.ylabel('Value')
        plt.title(
            f'Total Expenditure vs Upper Poverty Line - {lifestyle.capitalize()} {pop_type}')
        plt.legend()

        return plt



    def create_graph13(self, df, lifestyle, per_capita=False, aggregation='median'):
        """
        Creates a graph comparing food expenditure vs C3 with key points and metrics.
        Shows the relationship between actual food expenditure, food norms, and total expenditure.
        """
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        max_value = 20000
        
        # Create buckets
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', max_value,bucket_size=30, min_samples=30
        )

        # Define metrics for bucket statistics
        metrics = {
            'food_actual': {
                'columns': [f'food_actual{suffix}'],
                'func': lambda x: x[f'food_actual{suffix}'].mean() if aggregation == 'mean' else x[f'food_actual{suffix}'].median()
            },
            'food_norm': {
                'columns': [f'FoodNorm-{lifestyle}{suffix}'],
                'func': lambda x: x[f'FoodNorm-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'FoodNorm-{lifestyle}{suffix}'].median()
            },
            'c3': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: x[f'c3{suffix}'].mean() if aggregation == 'mean' else x[f'c3{suffix}'].median()
            },
            'count': {
                'columns': [f'c3{suffix}'],
                'func': lambda x: len(x)
            },
            'zl': {
                'columns': [f'ZL-{lifestyle}{suffix}'],
                'func': lambda x: x[f'ZL-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'ZL-{lifestyle}{suffix}'].median()
            }
        }
        

        # Calculate statistics for each bucket
        stats = self.helper.calculate_bucket_stats(df_bucketed, metrics=metrics)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot food expenditure vs C3
        plt.scatter(stats['c3'], stats['food_actual'], s=20, alpha=0.6,
                    color=self.colors[0], label='Food Expenditure')
        
        # Add sample size labels with smaller font
        for i in range(len(stats['c3'])):
            plt.text(stats['c3'][i], stats['food_actual'][i], 
                    f'N={stats["count"][i]}',
                    fontsize=6, alpha=0.5,  # Smaller font and more transparent
                    horizontalalignment='right',
                    verticalalignment='bottom')
        
        # Add trend line for food expenditure
        z = np.polyfit(stats['c3'], stats['food_actual'], 1)
        p = np.poly1d(z)
        plt.plot(stats['c3'], p(stats['c3']), '--', color=self.colors[0], alpha=0.8)
        
        # Plot C3 line
        plt.scatter(stats['c3'], stats['c3'], s=20, alpha=0.6,
                    color='gray', label='C3')
        plt.plot(stats['c3'], stats['c3'], '--', color='gray', alpha=0.5)
        
        # Find intersection point and calculate ZU, ZL
        # Find bucket where FoodNorm is closest to C3
        diff_norm_to_c3 = abs(stats['food_norm'] - stats['c3'])
        intersection_bucket_idx = diff_norm_to_c3.argmin()
        
        # For ZL: get the mean ZL value from the intersection bucket
        zl = stats['zl'][intersection_bucket_idx]
        zl_x = stats['c3'][intersection_bucket_idx]
        zl_y = stats['c3'][intersection_bucket_idx]
        
        # For ZU: find bucket where FoodNorm is closest to FoodActual
        diff_norm_to_actual = abs(stats['food_norm'] - stats['food_actual'])
        zu_bucket_idx = diff_norm_to_actual.argmin()
        zu = stats['c3'][zu_bucket_idx]
        zu_x = stats['c3'][zu_bucket_idx]
        zu_y = stats['food_actual'][zu_bucket_idx]
        
        # Calculate percentage differences
        zu_pct = abs(stats['food_norm'][zu_bucket_idx] - stats['food_actual'][zu_bucket_idx]) / stats['food_actual'][zu_bucket_idx] * 100
        zl_pct = abs(stats['food_norm'][intersection_bucket_idx] - stats['c3'][intersection_bucket_idx]) / stats['c3'][intersection_bucket_idx] * 100
        
        # Mark ZU and ZL points
        plt.scatter([zu_x], [zu_y], color='red', s=100, 
                    label='ZU point', zorder=5)
        plt.scatter([zl_x], [zl_y], color='blue', s=100, 
                    label='ZL point', zorder=5)
        
        # Add legend with calculations and bucket size
        legend_text = (
            f'ZU (C3 where FoodNorm ≈ FoodActual): {zu:.2f}\n'
            f'   |FoodNorm - FoodActual|/FoodActual: {zu_pct:.1f}%\n'
            f'ZL (mean C3 where FoodNorm ≈ C3): {zl:.2f}\n'
            f'   |FoodNorm - C3|/C3: {zl_pct:.1f}%\n'
            f'   Bucket width: {bucket_width:.1f}'
        )
        
        # Add legend box
        plt.text(0.02, 0.98, legend_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                fontsize=10)
        
        # Set labels and title
        plt.xlabel(f'Total Expenditure (C3) {pop_type}')
        plt.ylabel('Food Expenditure')
        plt.title(f'Food Expenditure vs Total Expenditure - {lifestyle.capitalize()} {pop_type}')
        plt.legend()
        
        return plt