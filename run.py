import pandas as pd
import numpy as np
from collections import defaultdict
from plot import add_visualization_to_analyzer 

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
                    # Define expected columns
                    columns = [
                        "misparmb",
                        "c3 - סך כל ההוצאות",
                        "F_norm Active cheapest - סך הוצאה נמוכה למספר הסלים",
                        "0 -4 min1", "0 -4 min2",
                        "5 - 9 min1", "5 - 9 min2",
                        "10-14 min1", "10-14 min2",
                        "15 - 17 min1", "15 - 17 min2",
                        "18 -29 min1", "18 -29 min2",
                        "30 - 49 min1", "30 - 49 min2",
                        "50+ min1", "50+ min2"
                    ]
                    
                    # Read CSV with specified columns
                    self.df = pd.read_csv(self.csv_path, encoding=encoding, skiprows=1, names=columns)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if self.df is None:
                raise Exception("Could not read file with any of the attempted encodings")
            
            # Convert numeric columns (all except misparmb)
            numeric_columns = [col for col in columns if col != "misparmb"]
            
            # Convert string numbers to float, handling potential commas in numbers
            for col in numeric_columns:
                self.df[col] = self.df[col].apply(lambda x: float(str(x).replace(',', '')) if pd.notnull(x) else 0)
                
            # Ensure misparmb is string
            self.df["misparmb"] = self.df["misparmb"].astype(str)
            
            return True
            
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            return False

    def create_group_key(self, row):
        """Create a tuple key from age distribution columns"""
        age_columns = [
            "0 -4 min1", "0 -4 min2",
            "5 - 9 min1", "5 - 9 min2",
            "10-14 min1", "10-14 min2",
            "15 - 17 min1", "15 - 17 min2",
            "18 -29 min1", "18 -29 min2",
            "30 - 49 min1", "30 - 49 min2",
            "50+ min1", "50+ min2"
        ]
        return tuple(row[col] for col in age_columns)

    def calculate_weighted_zl(self, row):
        """Calculate ZL using weighted sum and food norm"""
        age_columns = [
            "0 -4 min1", "0 -4 min2",
            "5 - 9 min1", "5 - 9 min2",
            "10-14 min1", "10-14 min2",
            "15 - 17 min1", "15 - 17 min2",
            "18 -29 min1", "18 -29 min2",
            "30 - 49 min1", "30 - 49 min2",
            "50+ min1", "50+ min2"
        ]
        
        # Calculate c30_c31
        values = [row[col] for col in age_columns]
        c30_c31 = sum(v * w for v, w in zip(values, self.zl_weights[1:])) + self.zl_weights[0]
        
        # Get food norm
        food_norm = row["F_norm Active cheapest - סך הוצאה נמוכה למספר הסלים"]
        
        # Calculate new ZL using the formula: food_norm + (food_norm - c30_c31)
        new_zl = food_norm + (food_norm - c30_c31)
        
        return new_zl

    def calculate_weighted_zu(self, row):
        """Calculate ZU using weighted sum"""
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
        return sum(v * w for v, w in zip(values, self.zu_weights[1:])) + self.zu_weights[0]

    def analyze_groups(self):
        """Group families and calculate statistics including poverty line"""
        if self.df is None:
            print("Please load the CSV file first")
            return
        
        # Initialize groups dictionary
        groups = defaultdict(lambda: {
            'count': 0,
            'expenses': [],
            'zu': 0,
            'zl': 0,
            'food_norms': [],
            'families': [],
            'poverty_line': 0,
            'below_poverty': 0
        })
        
        # First pass: collect expenses and basic stats
        for _, row in self.df.iterrows():
            group_key = self.create_group_key(row)
            
            # Update group statistics
            groups[group_key]['count'] += 1
            groups[group_key]['expenses'].append(row["c3 - סך כל ההוצאות"])
            groups[group_key]['families'].append(row["misparmb"])
            groups[group_key]['food_norms'].append(
                row["F_norm Active cheapest - סך הוצאה נמוכה למספר הסלים"]
            )
            
            # Calculate and add weighted ZU/ZL values
            groups[group_key]['zu'] += self.calculate_weighted_zu(row)
            groups[group_key]['zl'] += self.calculate_weighted_zl(row)
        
        # Second pass: calculate poverty lines and related statistics
        for group_key, data in groups.items():
            expenses = np.array(data['expenses'])
            median_exp = np.median(expenses)
            data['poverty_line'] = median_exp / 2
            data['below_poverty'] = sum(expenses < data['poverty_line'])
            data['poverty_rate'] = (data['below_poverty'] / data['count']) * 100
            
            # Add additional poverty-related statistics
            data['poverty_gap'] = np.mean([
                (data['poverty_line'] - exp) / data['poverty_line'] 
                for exp in expenses if exp < data['poverty_line']
            ]) if data['below_poverty'] > 0 else 0
            
            # Calculate severity of poverty (squared poverty gap)
            data['poverty_severity'] = np.mean([
                ((data['poverty_line'] - exp) / data['poverty_line']) ** 2 
                for exp in expenses if exp < data['poverty_line']
            ]) if data['below_poverty'] > 0 else 0
            
            # Add food norm statistics
            data['mean_food_norm'] = np.mean(data['food_norms'])
            
        self.groups = groups
        return groups

    def print_results(self):
        """Print analysis results including poverty statistics"""
        if self.groups is None:
            print("Please run analysis first")
            return
            
        print("\nFamily Group Analysis Results (including Poverty Analysis)")
        print("-" * 80)
        
        sorted_groups = sorted(self.groups.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for i, (pattern, data) in enumerate(sorted_groups, 1):
            print(f"\nGroup {i}:")
            print(f"Count: {data['count']} families")
            print(f"Weighted ZU: {data['zu']:.2f}")
            print(f"Weighted ZL: {data['zl']:.2f}")
            print(f"Mean Food Norm: {data['mean_food_norm']:.2f}")
            print("\nExpenditure Statistics:")
            print(f"  Mean: {np.mean(data['expenses']):.2f}")
            print(f"  Median: {np.median(data['expenses']):.2f}")
            print(f"  Min: {min(data['expenses']):.2f}")
            print(f"  Max: {max(data['expenses']):.2f}")
            print("\nPoverty Analysis:")
            print(f"  Poverty Line: {data['poverty_line']:.2f}")
            print(f"  Families Below Poverty: {data['below_poverty']} ({data['poverty_rate']:.1f}%)")
            print(f"  Poverty Gap: {data['poverty_gap']:.3f}")
            print(f"  Poverty Severity: {data['poverty_severity']:.3f}")
            print("-" * 60)

    
    def get_family_description(self, pattern):
        """
        Generate a human-readable description of the family composition
        """
        age_groups = [
            ('0-4', pattern[0], pattern[1]),
            ('5-9', pattern[2], pattern[3]),
            ('10-14', pattern[4], pattern[5]),
            ('15-17', pattern[6], pattern[7]),
            ('18-29', pattern[8], pattern[9]),
            ('30-49', pattern[10], pattern[11]),
            ('50+', pattern[12], pattern[13])
        ]
        
        description_parts = []
        for age_group, males, females in age_groups:
            if males > 0 or females > 0:
                gender_parts = []
                if males > 0:
                    gender_parts.append(f"{males}M")
                if females > 0:
                    gender_parts.append(f"{females}F")
                description_parts.append(f"{age_group}:{'+'.join(gender_parts)}")
        
        return ", ".join(description_parts)
    
    
    
    def export_results(self, output_path):
        """Export results to CSV file with specified fields"""
        if self.groups is None:
            print("Please run analysis first")
            return
            
        # Prepare data for export
        export_data = []
        for pattern, data in self.groups.items():
            row = {
                'Family Composition': self.get_family_description(pattern),
                'ZL': data['zl'],
                'ZU': data['zu'],
                'Mean Expenditures': np.mean(data['expenses']),
                'Median Expenditures': np.median(data['expenses']),
                'Poverty Rate (%)': data['poverty_rate'],
                'Count': data['count']
            }
            export_data.append(row)
            
        # Create DataFrame
        results_df = pd.DataFrame(export_data)
        
        # Sort by Count (descending) for better readability
        results_df = results_df.sort_values('Count', ascending=False)
        
        # Export to CSV
        results_df.to_csv(output_path, index=False)
        print(f"\nResults exported to {output_path}")

def main():
    # File paths
    input_file = "food_economics_2024 - all_families.csv"
    output_file = "family_groups_analysis_weighted.csv"
    
    # Initialize analyzer
    analyzer = FamilyGroupAnalyzer(input_file)
    
    # Add visualization capability
    add_visualization_to_analyzer(FamilyGroupAnalyzer)
    
    # Read and process data
    if analyzer.read_csv():
        print("CSV file read successfully")
        
        # Perform analysis
        analyzer.analyze_groups()
        
        # Print results
        analyzer.print_results()
        
        # Create and save plots
        analyzer.plot_and_save_groups()
        
        # Export results
        analyzer.export_results(output_file)
    else:
        print("Failed to read CSV file")

if __name__ == "__main__":
    main()