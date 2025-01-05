import pandas as pd
import numpy as np
from collections import defaultdict
from visualization_manager import VisualizationManager


class FamilyGroupAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

        # Active weights for ZL and ZU
        self.active_zl_weights = [
            176.4827575,  # base
            95.91524586, 384.5782373,    # 0-4
            250.5918009, 164.2829614,    # 5-9
            128.3428789, 208.2338264,    # 10-14
            313.5189142, 290.7941503,    # 15-17
            56.66946114, 289.0340474,    # 18-29
            152.2557543, 775.2361008,    # 30-49
            412.9771015, 616.6631982     # 50+
        ]

        self.active_zu_weights = [
            3097.0054841,                    # base
            1778.3197479, 2605.5969925,      # 0-4
            3687.3931666, 2312.422585,       # 5-9
            3060.8969612, 2898.5661122,      # 10-14
            2341.0918063, 4116.1657006,      # 15-17
            3508.5692799, 4451.6304668,      # 18-29
            4202.0822921, 5705.0959602,      # 30-49
            3561.6220116, 4707.0973191       # 50+
        ]

        # Sedentary weights for ZL and ZU
        self.sedentary_zl_weights = [
            5.684342e-13,     # base
            0, -124,          # 0-4
            -68.5, 1342,      # 5-9
            0, 0,             # 10-14
            283, 979,         # 15-17
            -4.583e-13, 206,  # 18-29
            124, 0,           # 30-49
            0, 0              # 50+
        ]

        self.sedentary_zu_weights = [
            3467.7127111,                    # base
            2060.1814015, 2533.9932615,      # 0-4
            3205.0854474, 1343.1611015,      # 5-9
            182.1488577, 2834.3453616,       # 10-14
            5978.3193323, 1850.2039152,      # 15-17
            2984.0997578, 1980.4124585,      # 18-29
            4274.668595, 6407.2484325,       # 30-49
            5433.0485529, 3862.4569651       # 50+
        ]

    def read_csv(self):
        """Read CSV file with proper encoding"""
        try:
            encodings = ['utf-8', 'cp1255', 'iso-8859-8']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(
                        self.csv_path,
                        encoding=encoding,
                        thousands=',',
                        quotechar='"',
                        na_values=['', 'NA', 'NaN']
                    )
                    print("\nColumns found in CSV:", self.df.columns.tolist())
                    break
                except UnicodeDecodeError:
                    continue

            if self.df is None:
                raise Exception(
                    "Could not read file with any of the attempted encodings")

            # Convert numeric columns
            for col in self.df.columns:
                if col != "misparmb":
                    self.df[col] = pd.to_numeric(
                        self.df[col], errors='coerce').fillna(0)

            return True

        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            return False

    def calculate_zl(self, row, is_sedentary):
        """Calculate ZL value based on household type"""
        age_columns = [
            "0 -4 min1", "0 -4 min2",
            "5 - 9 min1", "5 - 9 min2",
            "10-14 min1", "10-14 min2",
            "15 - 17 min1", "15 - 17 min2",
            "18 -29 min1", "18 -29 min2",
            "30 - 49 min1", "30 - 49 min2",
            "50+ min1", "50+ min2"
        ]

        values = [row[col] for col in age_columns]
        weights = self.sedentary_zl_weights if is_sedentary else self.active_zl_weights
        food_norm_col = 'FoodNorm-sedentary' if is_sedentary else 'FoodNorm-active'

        weighted_sum = sum(
            v * w for v, w in zip(values, weights[1:])) + weights[0]

        if not is_sedentary:
            food_norm = row[food_norm_col]
            weighted_sum = food_norm + (food_norm - weighted_sum)

        return weighted_sum

    def calculate_zu(self, row, is_sedentary):
        """Calculate ZU value based on household type"""
        age_columns = [
            "0 -4 min1", "0 -4 min2",
            "5 - 9 min1", "5 - 9 min2",
            "10-14 min1", "10-14 min2",
            "15 - 17 min1", "15 - 17 min2",
            "18 -29 min1", "18 -29 min2",
            "30 - 49 min1", "30 - 49 min2",
            "50+ min1", "50+ min2"
        ]

        values = [row[col] for col in age_columns]
        weights = self.sedentary_zu_weights if is_sedentary else self.active_zu_weights
        return sum(v * w for v, w in zip(values, weights[1:])) + weights[0]

    def calculate_persons_count(self, row):
        """Calculate total number of persons in household"""
        age_columns = [
            "0 -4 min1", "0 -4 min2",
            "5 - 9 min1", "5 - 9 min2",
            "10-14 min1", "10-14 min2",
            "15 - 17 min1", "15 - 17 min2",
            "18 -29 min1", "18 -29 min2",
            "30 - 49 min1", "30 - 49 min2",
            "50+ min1", "50+ min2"
        ]
        return sum(row[col] for col in age_columns)

    def calculate_children_under_10(self, row):
        """Calculate number of children under 10 years old in household"""
        children_columns = [
            # "0 -4 min1", "0 -4 min2",    # 0-4 years
            # "5 - 9 min1", "5 - 9 min2",  # 5-9 years
            # "10-14 min1", "10-14 min2",  # 10-14 years
            # "15 - 17 min1", "15 - 17 min2"  # 15-17 years
            "50+ min1", "50+ min2"
        ]
        return sum(row[col] for col in children_columns)

    def process_dataframe(self):
        """Process dataframe to calculate required metrics"""
        if self.df is None:
            return False

        print("Processing dataframe...")

        # Calculate number of persons in each household
        self.df['persons_count'] = self.df.apply(
            self.calculate_persons_count, axis=1)
        # Calculate number of children under 10
        self.df['children_under_10'] = self.df.apply(
            self.calculate_children_under_10, axis=1)
        print("Calculated household sizes and children counts")

        # Calculate ZL for both active and sedentary
        self.df['ZL-active'] = self.df.apply(
            lambda row: self.calculate_zl(
                row, False), axis=1)
        self.df['ZL-sedentary'] = self.df.apply(
            lambda row: self.calculate_zl(row, True), axis=1)

        # Calculate ZU (c^3) for active and sedentary
        self.df['ZU-active'] = self.df.apply(
            lambda row: self.calculate_zu(
                row, False), axis=1)
        self.df['ZU-sedentary'] = self.df.apply(
            lambda row: self.calculate_zu(row, True), axis=1)

        print("Calculated ZL and ZU values")

        # Calculate per-person metrics
        for lifestyle in ['active', 'sedentary']:
            # Get corresponding columns
            food_actual_col = f'food_actual'
            food_norm_col = f'FoodNorm-{lifestyle}'
            zl_col = f'ZL-{lifestyle}'
            zu_col = f'ZU-{lifestyle}'

            # Calculate per capita metrics
            for metric in ['c3', zl_col, zu_col]:
                self.df[f'{metric}_per_capita'] = self.df[metric] / \
                    (self.df['persons_count'])

            # Calculate food differences
            self.df[f'food_norm_diff_{lifestyle}'] = (
                self.df[food_actual_col] - self.df[food_norm_col]
            )

            # Per capita versions of food metrics
            self.df[f'{food_actual_col}_per_capita'] = (
                self.df[food_actual_col] / (self.df['persons_count'])
            )
            self.df[f'{food_norm_col}_per_capita'] = (
                self.df[food_norm_col] / (self.df['persons_count'])
            )
            self.df[f'food_norm_diff_{lifestyle}_per_capita'] = (
                self.df[f'{food_actual_col}_per_capita'] -
                self.df[f'{food_norm_col}_per_capita']
            )

        print("Calculated all metrics and differences")
        return True


def main():
    # Initialize analyzer
    analyzer = FamilyGroupAnalyzer("food_economics_2024.csv")

    # Read data
    print("\nReading data...")
    if not analyzer.read_csv():
        print("Failed to read data")
        return

    # Process data
    print("\nProcessing data...")
    if not analyzer.process_dataframe():
        print("Failed to process data")
        return

    # Initialize visualization manager and generate plots
    print("\nGenerating visualizations...")
    viz_manager = VisualizationManager('./graphs/')
    viz_manager.generate_all_plots(analyzer.df, None, True)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
