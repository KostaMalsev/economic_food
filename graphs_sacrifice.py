import numpy as np
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer


class SacrificeVisualizer(BaseVisualizer):

    def create_graph5(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Food Sacrifice Distribution"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        # Calculate food sacrifice
        df = df.copy()
        df['sacrifice'] = (df[f'FoodNorm-{lifestyle}{suffix}'] - df[f'food_actual{suffix}'])

        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, 'sacrifice', bucket_size=1
        )

        metrics = {
            'sacrifice': {
                'columns': ['sacrifice'],
                'func': lambda x: x['sacrifice'].mean() if aggregation == 'mean' else x['sacrifice'].median()
            },
            'poor_count': {
                'columns': ['sacrifice'],
                'func': lambda x: (x['sacrifice'] > 0).sum()
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)
        sorted_sacrifice = np.sort(stats['sacrifice'])

        plt.figure()
        sacrificing = (sorted_sacrifice < 0).sum()
        sacrificing_pct = (sacrificing / len(sorted_sacrifice)) * 100

        # Plot sorted sacrifice values
        plt.plot(range(len(sorted_sacrifice)), sorted_sacrifice,
                 color=self.colors[0], label='Food Sacrifice (FoodNorm - FoodActual)')

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Zero Point (No Sacrifice)')
        plt.axvline(
            x=sacrificing,
            color='red',
            linestyle='--',
            alpha=0.3,
            label=f'Sacrifice Line ({sacrificing_pct:.1f}% households)')

        plt.fill_between(range(len(sorted_sacrifice)),
                         sorted_sacrifice,
                         0, where=sorted_sacrifice > 0, color='red', alpha=0.1, label='Food Sacrificing')

        plt.xlabel('Households (Ordered by Food Sacrifice)')
        plt.ylabel(f'Food Norm - Food Expenditure{pop_type}')
        plt.title(
            f'Household Food Sacrifice Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()

        return plt

    def create_graph6(self, df, lifestyle, per_capita=False, aggregation='median'):
        """Total Expenditure Sacrifice Distribution"""
        suffix = '_per_capita' if per_capita else ''
        pop_type = self._get_display_type(per_capita)

        df = df.copy()
        df['sacrifice'] =  df[f'ZU-{lifestyle}{suffix}'] - df[f'c3{suffix}']

        df_bucketed, bucket_width = self.helper.create_fixed_width_buckets(
            df, 'sacrifice', bucket_size=1
        )

        metrics = {
            'sacrifice': {
                'columns': ['sacrifice'],
                'func': lambda x: x['sacrifice'].mean() if aggregation == 'mean' else x['sacrifice'].median()
            },
            'poor_count': {
                'columns': ['sacrifice'],
                'func': lambda x: (x['sacrifice'] < 0).sum()
            }
        }

        stats = self.helper.calculate_bucket_stats(
            df_bucketed, metrics=metrics)
        sorted_sacrifice = np.sort(stats['sacrifice'])

        plt.figure()
        sacrificing = (sorted_sacrifice > 0).sum()
        sacrificing_pct = (sacrificing / len(sorted_sacrifice)) * 100

        # Plot sorted sacrifice values
        plt.plot(range(len(sorted_sacrifice)), sorted_sacrifice,
                 color=self.colors[0], label='Total Expenditure Sacrifice (Zu-C3)')

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3,
                    label='Zero Point (No Sacrifice)')
        plt.axvline(
            x=sacrificing,
            color='red',
            linestyle='--',
            alpha=0.3,
            label=f'Sacrifice Line ({sacrificing_pct:.1f}% households)')

        plt.fill_between(range(len(sorted_sacrifice)),
                         sorted_sacrifice,
                         0,
                         where=sorted_sacrifice > 0,
                         color='red',
                         alpha=0.1,
                         label='Expenditure Sacrificing')

        plt.xlabel('Households (Ordered by Expenditure Sacrifice)')
        plt.ylabel(f'Total Expenditure - Upper Poverty Line {pop_type}')
        plt.title(
            f'Expenditure Sacrifice Distribution - {lifestyle.capitalize()} {pop_type}')
        plt.legend()

        return plt
