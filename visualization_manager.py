from graphs_expenditure import ExpenditureVisualizer
from graphs_sacrifice import SacrificeVisualizer
from graphs_sufficiency import SufficiencyVisualizer
from graphs_detailed import DetailedVisualizer
from graphs_normalized import NormalizedVisualizer


class VisualizationManager:
    def __init__(self, save_dir='./graphs/'):
        """Initialize visualization manager with all visualizers"""
        self.save_dir = save_dir
        self.visualizers = {
            'expenditure': ExpenditureVisualizer(save_dir),
            'sacrifice': SacrificeVisualizer(save_dir),
            'sufficiency': SufficiencyVisualizer(save_dir),
            'detailed': DetailedVisualizer(save_dir),
            'normalized': NormalizedVisualizer(save_dir)
        }

        # Map graph numbers to visualizer types
        self.graph_mapping = {
            1: 'expenditure',
            2: 'expenditure',
            3: 'expenditure',
            4: 'expenditure',
            5: 'sacrifice',
            6: 'sacrifice',
            7: 'sufficiency',
            8: 'sufficiency',
            9: 'detailed',
            10: 'detailed',
            11: 'normalized',
            12: 'normalized',
            70: 'sufficiency',
        }

    def get_visualizer(self, graph_num):
        """Get appropriate visualizer for a graph number"""
        viz_type = self.graph_mapping.get(graph_num)
        if viz_type is None:
            raise ValueError(f"No visualizer found for graph {graph_num}")
        return self.visualizers[viz_type]

    def generate_all_plots(self, df, lifestyles=None, per_capita_options=None):
        """Generate all plots for specified lifestyles and per capita options"""
        if lifestyles is None:
            lifestyles = ['active', 'sedentary']
        if per_capita_options is None:
            per_capita_options = [True, False]
        else:
            per_capita_options = [True]

        for lifestyle in lifestyles:
            print(f"\nGenerating plots for {lifestyle} lifestyle:")
            for per_capita in per_capita_options:
                metric_type = 'per_capita' if per_capita else 'household'
                print(f"\nGenerating {metric_type} metrics:")

                try:
                    for graph_num in (list(range(1, 13)) + [70]): #[70]: #range(1,13):
                        print(f"  Creating graph {graph_num}...")
                        visualizer = self.get_visualizer(graph_num)
                        plt = visualizer.create_graph(
                            graph_num, df, lifestyle, per_capita)
                        visualizer.save_plot(
                            f'graph{graph_num}_{lifestyle}_{metric_type}')

                except Exception as e:
                    print(f"Error generating {metric_type} plots for {lifestyle} " f"lifestyle: {str(e)}")

        print(f"\nAll plots have been saved in: {self.save_dir}")

    def create_graph(self, graph_num, df, lifestyle, per_capita=False):
        """Create a specific graph"""
        visualizer = self.get_visualizer(graph_num)
        return visualizer.create_graph(graph_num, df, lifestyle, per_capita)

    def save_plot(self, graph_num, filename):
        """Save a specific plot"""
        visualizer = self.get_visualizer(graph_num)
        visualizer.save_plot(filename)
