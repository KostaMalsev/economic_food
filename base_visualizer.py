import matplotlib.pyplot as plt
import os
from datetime import datetime
from bucketing_helper import BucketingHelper


class BaseVisualizer:
    def __init__(self, save_dir='./graphs/'):
        """Initialize base visualizer with consistent styling"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        plt.style.use('default')
        plt.rcParams.update({
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.figsize': [15, 8],
            'font.size': 10
        })
        self.colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        self.helper = BucketingHelper()

    def save_plot(self, filename):
        """Save plot with timestamp"""
        full_path = os.path.join(self.save_dir, f"{filename}_{self.timestamp}.png")
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot: {full_path}")

    def _get_display_type(self, per_capita):
        """Helper to get consistent display strings"""
        return 'Per Capita' if per_capita else 'Household'

    def create_graph(self, graph_num, *args, **kwargs):
        """Generic method to create graphs by number"""
        method_name = f'create_graph{graph_num}'
        if hasattr(self, method_name):
            return getattr(self, method_name)(*args, **kwargs)
        else:
            raise ValueError(
                f"Graph {graph_num} not implemented in {self.__class__.__name__}")
