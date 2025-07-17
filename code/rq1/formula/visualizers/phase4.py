import argparse
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

# Import classes from main module
from utils.helpers import load_json, save_json, setup_logging
from rq1.formula.phase4 import DatasetManager, Visualizer

# Initialize logging
logger = setup_logging()

class PlotReloader:
    """Reloads and regenerates plots from saved data."""
    
    def __init__(self, experiment_dir: Path, output_dir: Path = None, 
               show_plot_titles: bool = True, feature_name_map: Dict[str, str] = None):
        """Initialize with paths to experiment data and output directory."""
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir) if output_dir else self.experiment_dir / "replotted"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization options
        self.show_plot_titles = show_plot_titles
        self.feature_name_map = feature_name_map
        
        # Data containers
        self.formula = {}
        self.plot_data = {}
        self.dataset_manager = None
        self.visualizer = None
    
    def load_data(self) -> bool:
        """Load formula and plot data from experiment directory."""
        # Load formula
        formula_path = self.experiment_dir / "formula.json"

        if not formula_path.exists():
            logger.error(f"Formula file not found: {formula_path}")
            return False

        self.formula = load_json(formula_path)
        logger.info(f"Loaded formula from {formula_path}")

        # Load plot data
        plot_data_path = self.experiment_dir / "data" / "plot_data.joblib"
        
        if not plot_data_path.exists():
            logger.error(f"Plot data file not found: {plot_data_path}")
            return False

        self.plot_data = joblib.load(plot_data_path)
        logger.info(f"Loaded plot data from {plot_data_path}")
        
        
        # Load datasets metadata
        summary_path = self.experiment_dir / "summary.json"

        if not summary_path.exists():
            logger.error(f"Summary file not found: {summary_path}")
            self.dataset_manager = self._create_dummy_dataset_manager([], [])

        summary = load_json(summary_path)
        phase1_dir = self.experiment_dir.parent
            
        # Initialize dataset manager with basic info
        self.dataset_manager = self._create_dummy_dataset_manager(
            summary.get("formula_info", {}).get("trained_on", []),
            list(self.plot_data.get("dataset_stats", {}).keys())
        )
        logger.info("Created dataset manager with metadata")
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.output_dir)
        
        return True
    
    def _create_dummy_dataset_manager(self, training_datasets, evaluation_datasets):
        """Create a dataset manager with metadata but no actual data loading."""
        dataset_manager = DatasetManager(Path("."))  # Path doesn't matter here
        
        # Register all training datasets
        for dataset in training_datasets:
            dataset_manager.register_dataset(dataset, category="training")
            
            # Also register the test split if it exists in evaluation datasets
            test_dataset = f"{dataset}_test"
            if test_dataset in evaluation_datasets:
                dataset_manager.register_dataset(
                    test_dataset, category="training_test", parent=dataset
                )
        
        # Register remaining evaluation datasets
        for dataset in evaluation_datasets:
            # Skip if already registered as a test split
            if any(dataset == f"{train}_test" for train in training_datasets):
                continue
            
            dataset_manager.register_dataset(dataset, category="evaluation")
        
        return dataset_manager
    
    def regenerate_plots(self):
        """Regenerate all plots from loaded data."""
        if not self.plot_data or not self.visualizer:
            logger.error("Plot data or visualizer not initialized")
            return {}
        
        viz_paths = {}
        
        # Plot formula coefficients
        viz_paths["coefficients"] = self.visualizer.plot_coefficients(
            self.formula, 
            show_title=self.show_plot_titles,
            feature_names=self.feature_name_map
        )
        
        # Extract data for plots
        combined_df = self.plot_data.get("combined_df")
        combined_y = self.plot_data.get("combined_y")
        combined_scores = self.plot_data.get("combined_scores")
        
        # Check if we have valid data
        if not self._has_valid_data(combined_df, combined_y, combined_scores):
            logger.warning("Missing or invalid plot data")
            return viz_paths
        
        # Plot overall distribution
        simple_scores = combined_scores[combined_y == 0]
        expert_scores = combined_scores[combined_y == 1]
        
        if len(simple_scores) > 0 and len(expert_scores) > 0:
            viz_paths["distributions"] = self.visualizer.plot_distributions(
                simple_scores, 
                expert_scores, 
                self.plot_data.get("overall_stats", {}),
                show_title=self.show_plot_titles
            )
        
        # Plot ROC curves
        individual_metrics = self.plot_data.get("individual_metrics", {})
        viz_paths["roc"] = self.visualizer.plot_roc_curves(
            combined_y, 
            combined_scores, 
            individual_metrics,
            show_title=self.show_plot_titles,
            feature_names=self.feature_name_map
        )
        
        # Plot dataset distributions
        dataset_stats = self.plot_data.get("dataset_stats", {})
        viz_paths["datasets"] = self.visualizer.plot_dataset_distributions(
            dataset_stats, 
            combined_df, 
            combined_scores, 
            combined_y, 
            self.dataset_manager,
            show_title=self.show_plot_titles
        )
        
        # Plot boxplot
        outlier_stats = self.plot_data.get("outlier_stats", {})
        if outlier_stats and "thresholds" in outlier_stats:
            viz_paths["boxplot"] = self.visualizer.plot_boxplot(
                combined_df, 
                combined_scores, 
                outlier_stats["thresholds"], 
                self.dataset_manager,
                show_title=self.show_plot_titles,
                show_legend=False  # Make the legend optional, default to not showing
            )
        
        logger.info(f"Regenerated {len(viz_paths)} visualizations")
        return viz_paths
    
    def _has_valid_data(self, df, y, scores):
        """Check if data is valid for visualization."""
        return (df is not None and len(df) > 0 and 
                y is not None and len(y) > 0 and
                scores is not None and len(scores) > 0)

def main():
    """Main function for running the replotting script."""
    parser = argparse.ArgumentParser(
        description="Regenerate plots from saved formula evaluation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--experiment-dir", required=True, type=str,
                      help="Path to the experiment directory containing formula results")
    parser.add_argument("--output-dir", type=str,
                      help="Directory for regenerated plots (default: EXPERIMENT_DIR/replotted)")
    # Add new visualization arguments
    parser.add_argument("--no-plot-titles", action="store_true",
                      help="Hide titles on plots")
    parser.add_argument("--feature-name-map", type=str,
                      help="Path to JSON file with feature name mappings")
    parser.add_argument("--show-boxplot-legend", action="store_true",
                      help="Show legend on boxplot (hidden by default)")
    
    args = parser.parse_args()
    
    # Load feature name map if provided
    feature_name_map = None
    if args.feature_name_map:
        feature_name_map = load_json(args.feature_name_map)
        logger.info(f"Loaded feature name mappings for {len(feature_name_map)} features")


    feature_name_map = {
        "jargon/roberta-large/abbr_general_density": "Abbreviation Density",
        "llm/mse_1.0+kl_0.0+ce_0.0+soft_0.0/dimension_scores/vocabulary_complexity": "Vocabulary Complexity",
        "textstat/dale_chall": "Dale-Chall Readability",
        "textstat/type_token_ratio": "Type-Token Ratio",
        "commonlit/albert": "Commonlit Albert",
        "jargon/roberta-large/jargon_count": "Jargon Count",
        "scigispy/PCREF_chunk_1p": "Referential Cohesion",
        "scigispy/WRDIC": "Information Content",
        "syntax/en_core_web_trf/embedding_depth": "Embedding Depth",
        "syntax/en_core_web_trf/verb_ratio": "Verb Ratio",
        "syntax/en_core_web_trf/function_ratio": "Function Word Ratio",
        "cluster/medreadme": "MedReadMe Cluster Score",
        "masked_prob/random/Bio_ClinicalBERT": "ClinicalBERT MLM Score",
        "syntax/en_core_web_trf/avg_dependency_distance": "Dependency Distance",
        "umls/en_core_web_trf/avg_cui_score": "CUI Score",
        "gispy/CoREF": "Coreference Resolution"
    }
    
    # Initialize reloader
    reloader = PlotReloader(
        args.experiment_dir, 
        args.output_dir,
        show_plot_titles=not args.no_plot_titles,
        feature_name_map=feature_name_map
    )
    
    # Load data
    if not reloader.load_data():
        logger.error("Failed to load required data. Exiting.")
        return 1
    
    # Regenerate plots
    viz_paths = reloader.regenerate_plots()

    if not viz_paths:
        logger.error("No visualizations generated. Exiting.")
        return 1

    # Save paths to generated visualizations
    for name, path in viz_paths.items():
        logger.info(f"Generated visualization: {name} -> {path}")
    return 0

if __name__ == "__main__":
    main()