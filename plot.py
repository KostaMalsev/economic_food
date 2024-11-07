import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from datetime import datetime
import os

class GroupVisualizer:
    @staticmethod
    def create_expenditure_plot(groups, num_groups=5, plot_dir=None):
        """
        Create and save visualization including poverty line
        
        Args:
            groups: Dictionary containing group data
            num_groups: Number of top groups to visualize
            plot_dir: Optional directory path for saving plots. If None, uses current directory
        """
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            save_path = lambda filename: os.path.join(plot_dir, filename)
        else:
            save_path = lambda filename: filename
            
        sorted_groups = sorted(groups.items(), key=lambda x: x[1]['count'], reverse=True)
        top_groups = sorted_groups[:num_groups]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        colors = sns.color_palette("husl", num_groups)
        
        # Main distribution plot
        plt.figure(figsize=(15, 10))
        plt.grid(True, alpha=0.3)
        plt.title('Expenditure Distribution with ZL/ZU Values and Poverty Line', pad=20)
        plt.xlabel('Expenditure Amount')
        plt.ylabel('Density')
        
        for idx, (pattern, data) in enumerate(top_groups):
            expenses = np.array(data['expenses'])
            avg_zl = data['zl'] / data['count']
            avg_zu = data['zu'] / data['count']
            poverty_line = data['poverty_line']
            poverty_count = sum(expenses < poverty_line)
            poverty_rate = (poverty_count / len(expenses)) * 100
            
            kernel = stats.gaussian_kde(expenses)
            x_range = np.linspace(min(expenses), max(expenses), 200)
            density = kernel(x_range)
            
            plt.plot(x_range, density, color=colors[idx], 
                    label=f'Group {idx+1} (n={data["count"]}, {poverty_rate:.1f}% below poverty)')
            
            mean_exp = np.mean(expenses)
            plt.axvline(mean_exp, color=colors[idx], linestyle='--', alpha=0.5)
            plt.axvline(avg_zl, color=colors[idx], linestyle=':', alpha=0.7)
            plt.axvline(avg_zu, color=colors[idx], linestyle='-.', alpha=0.7)
            plt.axvline(poverty_line, color=colors[idx], linestyle='-', alpha=0.3)
            
            y_pos = max(density) * (0.8 - idx * 0.15)
            plt.text(mean_exp, y_pos, f'Mean: {mean_exp:,.0f}', 
                    color=colors[idx], ha='right', va='bottom')
            plt.text(avg_zl, y_pos, f'ZL: {avg_zl:,.0f}', 
                    color=colors[idx], ha='right', va='top')
            plt.text(avg_zu, y_pos, f'ZU: {avg_zu:,.0f}', 
                    color=colors[idx], ha='left', va='top')
            plt.text(poverty_line, y_pos * 0.9, f'Poverty: {poverty_line:,.0f}', 
                    color=colors[idx], ha='right', va='bottom')
        
        plt.legend(title='Family Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.figtext(1.15, 0.5, 
                   'Line Styles:\n' +
                   '--  Mean Expenditure\n' +
                   ':   ZL (Lower Bound)\n' +
                   '-.  ZU (Upper Bound)\n' +
                   '-   Poverty Line',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        filename = save_path(f'expenditure_distribution_{timestamp}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved expenditure distribution plot: {filename}")
        
        # Individual group plots
        for idx, (pattern, data) in enumerate(top_groups):
            plt.figure(figsize=(10, 6))
            
            expenses = np.array(data['expenses'])
            avg_zl = data['zl'] / data['count']
            avg_zu = data['zu'] / data['count']
            poverty_line = data['poverty_line']
            poverty_count = sum(expenses < poverty_line)
            poverty_rate = (poverty_count / len(expenses)) * 100
            
            sns.histplot(expenses, kde=True, color=colors[idx])
            
            mean_exp = np.mean(expenses)
            plt.axvline(mean_exp, color='black', linestyle='--', label='Mean')
            plt.axvline(avg_zl, color='red', linestyle=':', label='ZL')
            plt.axvline(avg_zu, color='green', linestyle='-.', label='ZU')
            plt.axvline(poverty_line, color='purple', linestyle='-', label='Poverty Line')
            
            plt.title(f'Group {idx+1} Distribution (n={data["count"]}, {poverty_rate:.1f}% below poverty)')
            plt.xlabel('Expenditure Amount')
            plt.ylabel('Count')
            
            # Add shading for poverty area
            plt.axvspan(min(expenses), poverty_line, color='red', alpha=0.1, 
                       label=f'Below Poverty ({poverty_count} families)')
            
            plt.text(mean_exp, plt.ylim()[1], f'Mean: {mean_exp:,.0f}', 
                    rotation=90, va='top')
            plt.text(avg_zl, plt.ylim()[1], f'ZL: {avg_zl:,.0f}', 
                    rotation=90, va='top', color='red')
            plt.text(avg_zu, plt.ylim()[1], f'ZU: {avg_zu:,.0f}', 
                    rotation=90, va='top', color='green')
            plt.text(poverty_line, plt.ylim()[1], f'Poverty: {poverty_line:,.0f}', 
                    rotation=90, va='top', color='purple')
            
            plt.legend()
            
            filename = save_path(f'group_{idx+1}_distribution_{timestamp}.png')
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved group {idx+1} plot: {filename}")
        
        # Box plot with poverty lines
        plt.figure(figsize=(12, 6))
        box_data = []
        labels = []
        poverty_lines = []
        
        for idx, (pattern, data) in enumerate(top_groups):
            box_data.append(data['expenses'])
            poverty_rate = (sum(data['expenses'] < data['poverty_line']) / len(data['expenses'])) * 100
            labels.append(f'Group {idx+1}\n(n={data["count"]}\n{poverty_rate:.1f}% poor)')
            poverty_lines.append(data['poverty_line'])
        
        plt.boxplot(box_data, labels=labels)
        
        # Add poverty lines to box plot
        for idx, poverty_line in enumerate(poverty_lines):
            plt.hlines(y=poverty_line, xmin=idx+0.7, xmax=idx+1.3, 
                      color='red', linestyle='--', alpha=0.5)
        
        plt.title('Expenditure Distribution Comparison with Poverty Lines')
        plt.ylabel('Expenditure Amount')
        plt.grid(True, axis='y', alpha=0.3)
        
        filename = save_path(f'expenditure_boxplot_{timestamp}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved box plot: {filename}")
        
        return plot_dir if plot_dir else "current directory"

def add_visualization_to_analyzer(FamilyGroupAnalyzer):
    """Add visualization method to the analyzer class"""
    def plot_and_save_groups(self, plot_dir=None):
        """
        Create and save plots for group analysis
        
        Args:
            plot_dir: Optional directory path for saving plots. 
                     If None, saves in current directory.
        """
        if self.groups is None:
            print("Please run analysis first")
            return
        
        save_location = GroupVisualizer.create_expenditure_plot(self.groups, plot_dir=plot_dir)
        print(f"\nAll plots have been saved in: {save_location}")
    
    # Add the method to the class
    FamilyGroupAnalyzer.plot_and_save_groups = plot_and_save_groups