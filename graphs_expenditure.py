import numpy as np
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer


class ExpenditureVisualizer(BaseVisualizer):

    def create_graph1(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Food Expenditure vs Total Expenditure Analysis"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        max_value = 20000
        
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'c3{suffix}', max_value, min_samples=70
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


    def create_graph4(self, df, lifestyle, per_capita=False, aggregation='mean'):
        """Poverty Lines Relationship"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)
        bucket_size = 300
        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, f'ZL-{lifestyle}{suffix}', bucket_size=bucket_size,min_samples=bucket_size
        )

        metrics = {
            'zl': {
                'columns': [f'ZL-{lifestyle}{suffix}'],
                'func': lambda x: x[f'ZL-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'ZL-{lifestyle}{suffix}'].median()
            },
            'zu': {
                'columns': [f'ZU-{lifestyle}{suffix}'],
                'func': lambda x: x[f'ZU-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'ZU-{lifestyle}{suffix}'].median()
            },
            'norm': {
                'columns': [f'FoodNorm-{lifestyle}{suffix}'],
                'func': lambda x: x[f'FoodNorm-{lifestyle}{suffix}'].mean() if aggregation == 'mean' else x[f'FoodNorm-{lifestyle}{suffix}'].median()
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)

        plt.figure()
        plt.plot(stats['zl'], stats['zu'], alpha=0.5,
                 color=self.colors[0], label='Upper Poverty Line (ZU)')
        plt.plot(stats['zl'], stats['norm'], alpha=0.5,
                 color=self.colors[1], label='Food Norm')

        # Add 3x reference line
        zl_range = np.array([stats['zl'].min(), stats['zl'].max()])
        plt.plot(zl_range, 3 * zl_range, '--',
                 color=self.colors[2], label='3x Reference Line')

        plt.xlabel(f'Lower Poverty Line (ZL) {pop_type}')
        plt.ylabel('Value')
        plt.title(
            f'Poverty Lines Relationship - {lifestyle.capitalize()} {pop_type} bucket-size:{bucket_size}')
        plt.legend()

        return plt
