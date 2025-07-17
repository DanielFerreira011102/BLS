import argparse
import logging
from pathlib import Path
import joblib

# Import classes from main module
from utils.helpers import load_json, save_json, setup_logging
from rq1.formula.phase3 import ResultsVisualizer, LassoSelector, ElasticNetSelector, BootstrapSelector

# Initialize logging
logger = setup_logging()

class ReplotManager:
    """Manager for replotting previous analysis."""
    
    def __init__(self, plot_data_path: Path, output_dir: Path = None):
        """Initialize with path to saved plot data."""
        self.plot_data_path = Path(plot_data_path)
        self.output_dir = Path(output_dir) if output_dir else self.plot_data_path.parent.parent / "replotted"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.methods = {}
        self.ordered_variants = {}
        self.evaluation_results = {}
        self.optimal_counts = {}
        self.correlation_info = {}
    
    def load_data(self) -> bool:
        """Load plot data from file."""
        logger.info(f"Loading plot data from {self.plot_data_path}")
        
        if not self.plot_data_path.exists():
            logger.error(f"Plot data file not found: {self.plot_data_path}")
            return False
        
        plot_data = joblib.load(self.plot_data_path)
        
        # Extract data
        self.methods = self._reconstruct_methods(plot_data.get("methods", {}))
        self.ordered_variants = plot_data.get("ordered_variants", {})
        self.evaluation_results = plot_data.get("evaluation_results", {})
        self.optimal_counts = plot_data.get("optimal_counts", {})
        self.correlation_info = plot_data.get("correlation_info", {})
        
        logger.info(f"Loaded data for {len(self.methods)} methods")
        return True
    
    def _reconstruct_methods(self, method_data: dict) -> dict:
        """Reconstruct method objects from serialized data."""
        methods = {}
        
        for method_id, data in method_data.items():
            # Determine method class
            if method_id.startswith("bootstrap"):
                method = self._create_bootstrap_method(method_id, data)
            elif method_id == "lasso":
                method = self._create_lasso_method(method_id, data)
            elif method_id == "elasticnet":
                method = self._create_elasticnet_method(method_id, data)
            else:
                logger.warning(f"Unknown method type: {method_id}")
                continue
                
            methods[method_id] = method
            
        return methods
    
    def _create_bootstrap_method(self, method_id: str, data: dict) -> BootstrapSelector:
        """Create a bootstrap method from data."""
        # Parse components from method_id
        parts = method_id.split("_")
        regularization = parts[1] if len(parts) > 1 else "elasticnet"
        variant = parts[2] if len(parts) > 2 else None
        selection = parts[3] if len(parts) > 3 else None
        
        method = BootstrapSelector(
            X=None,  # Dummy value, not used for plotting
            y=None,  # Dummy value, not used for plotting
            regularization=regularization,
            variant=variant,
            selection=selection
        )
        
        # Set properties from data
        method.name = method_id
        method.features = data.get("features", [])
        method.importances = data.get("importances", {})
        method.filtering_history = data.get("filtering_history", {})
        method.threshold_info = data.get("threshold_info", {})
        
        return method
    
    def _create_lasso_method(self, method_id: str, data: dict) -> LassoSelector:
        """Create a LASSO method from data."""
        method = LassoSelector(
            X=None,  # Dummy value, not used for plotting
            y=None,  # Dummy value, not used for plotting
        )
        
        # Set properties from data
        method.name = method_id
        method.features = data.get("features", [])
        method.importances = data.get("importances", {})
        method.filtering_history = data.get("filtering_history", {})
        method.params = data.get("params", {})
        
        return method
    
    def _create_elasticnet_method(self, method_id: str, data: dict) -> ElasticNetSelector:
        """Create an ElasticNet method from data."""
        method = ElasticNetSelector(
            X=None,  # Dummy value, not used for plotting
            y=None,  # Dummy value, not used for plotting
        )
        
        # Set properties from data
        method.name = method_id
        method.features = data.get("features", [])
        method.importances = data.get("importances", {})
        method.filtering_history = data.get("filtering_history", {})
        method.params = data.get("params", {})
        
        return method
    
    def replot(self) -> None:
        """Regenerate all visualizations."""
        if not self.methods:
            logger.error("No method data loaded. Call load_data() first.")
            return
        
        # Create visualizer
        visualizer = ResultsVisualizer(self.output_dir)
        
        # Regenerate all plots
        visualizer.plot_all(self.methods, self.evaluation_results, self.optimal_counts)
        
        logger.info(f"Visualizations regenerated in {self.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate plots from saved Phase 3 analysis data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--plot-data-path",
        type=str,
        required=True,
        help="Path to plot data file (plot_data.joblib)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for regenerated plots (default: parent_dir/replotted)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the replotting tool."""
    args = parse_args()
    
    # Initialize replot manager
    replot_manager = ReplotManager(args.plot_data_path, args.output_dir)
    
    # Load data and replot
    if replot_manager.load_data():
        replot_manager.replot()
        print(f"Plots regenerated in: {replot_manager.output_dir}")
    else:
        print("Failed to load plot data. See log for details.")


if __name__ == "__main__":
    main()