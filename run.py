import pandas as pd
import numpy as np
from collections import defaultdict

class FamilyGroupAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.groups = None
        
        # Define weights for ZL and ZU calculations
        self.zl_weights = [
            176.4827575,  # base
            95.91524586, 384.5782373,    # 0-4
            250.5918009, 164.2829614,    # 5-9
            128.3428789, 208.2338264,    # 10-14
            313.5189142, 290.7941503,    # 15-17
            56.66946114, 289.0340474,    # 18-29
            152.2557543, 775.2361008,    # 30-49
            412.9771015, 616.6631982     # 50+
        ]
        
        self.zu_weights = [
            3097.0054841,                    # base
            1778.3197479, 2605.5969925,      # 0-4
            3687.3931666, 2312.422585,       # 5-9
            3060.8969612, 2898.5661122,      # 10-14
            2341.0918063, 4116.1657006,      # 15-17
            3508.5692799, 4451.6304668,      # 18-29
            4202.0822921, 5705.0959602,      # 30-49
            3561.6220116, 4707.0973191       # 50+
        ]
    
    def read_csv(self):
        """Read CSV file with proper encoding for Hebrew characters"""
        try:
            encodings = ['utf-8', 'cp1255', 'iso-8859-8']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if self.df is None:
                raise Exception("Could not read file with any of the attempted encodings")
                
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            return True
            
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            return False
    
    def create_group_key(self, row):
        """Create a tuple key from age distribution columns"""
        age_columns = [
            '0 -4 min1', '0 -4 min2',
            '5 - 9 min1', '5 - 9 min2',
            '10-14 min1', '10-14 min2',
            '15 - 17 min1', '15 - 17 min2',
            '18 -29 min1', '18 -29 min2',
            '30 - 49 min1', '30 - 49 min2',
            '50+ min1', '50+ min2'
        ]
        return tuple(row[col] for col in age_columns)
    
    def calculate_weighted_zl(self, row):
        """Calculate ZL using weighted sum"""
        age_columns = [
            '0 -4 min1', '0 -4 min2',
            '5 - 9 min1', '5 - 9 min2',
            '10-14 min1', '10-14 min2',
            '15 - 17 min1', '15 - 17 min2',
            '18 -29 min1', '18 -29 min2',
            '30 - 49 min1', '30 - 49 min2',
            '50+ min1', '50+ min2'
        ]
        
        values = [row[col] for col in age_columns]
        # Calculate weighted sum using ZL weights (skipping the base weight)
        return sum(v * w for v, w in zip(values, self.zl_weights[1:])) + self.zl_weights[0]
    
    def calculate_weighted_zu(self, row):
        """Calculate ZU using weighted sum"""
        age_columns = [
            '0 -4 min1', '0 -4 min2',
            '5 - 9 min1', '5 - 9 min2',
            '10-14 min1', '10-14 min2',
            '15 - 17 min1', '15 - 17 min2',
            '18 -29 min1', '18 -29 min2',
            '30 - 49 min1', '30 - 49 min2',
            '50+ min1', '50+ min2'
        ]
        
        values = [row[col] for col in age_columns]
        # Calculate weighted sum using ZU weights (skipping the base weight)
        return sum(v * w for v, w in zip(values, self.zu_weights[1:])) + self.zu_weights[0]
    
    def analyze_groups(self):
        """Group families and calculate statistics with weighted ZL and ZU"""
        if self.df is None:
            print("Please load the CSV file first")
            return
        
        # Initialize groups dictionary
        groups = defaultdict(lambda: {
            'count': 0,
            'expenses': [],
            'zu': 0,
            'zl': 0,
            'families': []
        })
        
        # Process each row
        for _, row in self.df.iterrows():
            group_key = self.create_group_key(row)
            
            # Update group statistics
            groups[group_key]['count'] += 1
            groups[group_key]['expenses'].append(row['c3 - סך כל ההוצאות'])
            groups[group_key]['families'].append(row['misparmb'])
            
            # Calculate and add weighted ZU/ZL values
            groups[group_key]['zu'] += self.calculate_weighted_zu(row)
            groups[group_key]['zl'] += self.calculate_weighted_zl(row)
        
        self.groups = groups
        return groups
    
    def print_results(self):
        """Print analysis results"""
        if self.groups is None:
            print("Please run analysis first")
            return
            
        print("\nFamily Group Analysis Results (with Weighted ZL and ZU)")
        print("-" * 70)
        
        # Sort groups by count in descending order
        sorted_groups = sorted(self.groups.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for i, (pattern, data) in enumerate(sorted_groups, 1):
            print(f"\nGroup {i}:")
            print(f"Count: {data['count']} families")
            print(f"Weighted ZU: {data['zu']:.2f}")
            print(f"Weighted ZL: {data['zl']:.2f}")
            print("Expenses Statistics:")
            print(f"  Mean: {np.mean(data['expenses']):.2f}")
            print(f"  Min: {min(data['expenses']):.2f}")
            print(f"  Max: {max(data['expenses']):.2f}")
            print(f"Family IDs: {', '.join(map(str, data['families'][:5]))}...")
            print("-" * 50)
    
    def export_results(self, output_path):
        """Export results to CSV file"""
        if self.groups is None:
            print("Please run analysis first")
            return
            
        # Prepare data for export
        export_data = []
        for pattern, data in self.groups.items():
            row = {
                'count': data['count'],
                'weighted_zu': data['zu'],
                'weighted_zl': data['zl'],
                'mean_expenses': np.mean(data['expenses']),
                'min_expenses': min(data['expenses']),
                'max_expenses': max(data['expenses']),
                'families': ','.join(map(str, data['families']))
            }
            # Add pattern values
            for i, value in enumerate(pattern):
                row[f'pattern_{i}'] = value
            export_data.append(row)
            
        # Create and save DataFrame
        results_df = pd.DataFrame(export_data)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults exported to {output_path}")

def main():
    # File paths
    input_file = "food_economics_2024 - all_families.csv"  # Replace with your actual file path
    output_file = "family_groups_analysis_weighted.csv"
    
    # Initialize analyzer
    analyzer = FamilyGroupAnalyzer(input_file)
    
    # Read and process data
    if analyzer.read_csv():
        print("CSV file read successfully")
        
        # Perform analysis
        analyzer.analyze_groups()
        
        # Print results
        analyzer.print_results()
        
        # Export results
        analyzer.export_results(output_file)
    else:
        print("Failed to read CSV file")

if __name__ == "__main__":
    main()