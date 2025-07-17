import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from utils.helpers import setup_logging, load_json, save_json

# Initialize logging
logger = setup_logging()


class ScoreProcessor:
    """Processes model complexity scores into normalized, discretized, and percentile forms."""
    
    @staticmethod
    def normalize_scores(scores, min_score=None, max_score=None):
        """Normalize scores to a 0-100 scale."""
        if min_score is None:
            min_score = np.min(scores)
        
        if max_score is None:
            max_score = np.max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            logger.warning("Min and max scores are identical, returning default value of 50")
            return np.full_like(scores, 50, dtype=float)
        
        # Scale to 0-100 range
        normalized = (scores - min_score) / (max_score - min_score) * 100
        return normalized
    
    @staticmethod
    def discretize_scores(scores, interval=5):
        """Round scores to the nearest multiple of the specified interval."""
        return np.round(scores / interval) * interval
    
    @staticmethod
    def convert_to_percentiles(scores):
        """Convert scores to percentile ranks (0-100)."""
        # Calculate the percentile rank for each score
        percentiles = stats.rankdata(scores, "average") / len(scores) * 100
        return percentiles
    
    @staticmethod
    def bin_quantile(scores, num_bins=21, labels=None):
        """Bin scores into equal-sized groups using quantile binning."""
        # Reshape for KBinsDiscretizer
        scores_array = np.array(scores).reshape(-1, 1)
        
        # Create discretizer with quantile strategy
        kbd = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
        binned = kbd.fit_transform(scores_array).flatten().astype(int)
        
        # Return early if no custom labels
        if labels is None:
            return binned
        
        # Map to custom labels if provided
        label_map = {i: labels[i] for i in range(len(labels))}
        return np.array([label_map[b] for b in binned])


class ComplexityVisualizer:
    """Creates visualizations for complexity scores and binning methods."""
    
    def __init__(self, output_dir: Path):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            "distributions": self.output_dir / "distributions",
            "bin_ranges": self.output_dir / "bin_ranges"
        }
        
        # Create directories
        for path in self.subdirs.values():
            path.mkdir(exist_ok=True)
        
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    
    def create_all_visualizations(self, df: pd.DataFrame, score_name: str = "complexity_score"):
        """Create all visualizations for complexity scores."""
        if df.empty:
            logger.error("Cannot create visualizations: DataFrame is empty")
            return {}
        
        # Check if required columns exist
        required_columns = [
            f"{score_name}_raw",
            f"{score_name}_normalized",
            f"{score_name}_percentile",
            f"{score_name}_percentile_discretized",
            f"{score_name}_quantile"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return {}
        
        # Paths to visualizations
        viz_paths = {}
        
        # Create distribution plots for both methods
        viz_paths["quantile_distribution"] = self.plot_distribution_with_bins(
            df, 
            f"{score_name}_normalized", 
            f"{score_name}_quantile", 
            "Quantile"
        )
        
        viz_paths["percentile_distribution"] = self.plot_distribution_with_bins(
            df, 
            f"{score_name}_normalized", 
            f"{score_name}_percentile_discretized", 
            "Percentile"
        )
        
        # Create bin range plots for both methods
        viz_paths["quantile_bin_ranges"] = self.plot_bin_ranges(
            df, 
            f"{score_name}_normalized", 
            f"{score_name}_quantile", 
            "Quantile"
        )
        
        viz_paths["percentile_bin_ranges"] = self.plot_bin_ranges(
            df, 
            f"{score_name}_normalized", 
            f"{score_name}_percentile_discretized", 
            "Percentile"
        )
        
        # Note: Removed bin distribution plots as requested
        
        logger.info(f"Created {len(viz_paths)} visualizations")
        return viz_paths
    
    def plot_distribution_with_bins(self, df, score_col, bin_col, method_name):
        """Create a distribution plot with colored bins and stacked bar indicator."""
        # Get unique bin values and sort them
        unique_bins = sorted(df[bin_col].unique())
        n_bins = len(unique_bins)

        # Create a linear segmented colormap from green to red
        cmap = LinearSegmentedColormap.from_list(
            "green_red",
            ["green", "red"],
            N=n_bins
        )

        # Generate colors for each bin
        colors = [cmap(i/n_bins) for i in range(n_bins)]

        # Create figure with two subplots - main plot and color bar at bottom
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 0.5])
        ax1 = fig.add_subplot(gs[0, 0])  # Main plot (top 80%)
        ax2 = fig.add_subplot(gs[1, 0])  # Color bar ()
        
        # Plot main histogram with KDE
        sns.histplot(
            data=df,
            x=score_col,
            stat="density",
            kde=True,
            line_kws={"linewidth": 2},
            alpha=0.5,
            color="#444444",
            ax=ax1
        )
        
        # Plot individual histograms for each bin
        bin_data = []
        bin_ranges = []
        for i, bin_value in enumerate(unique_bins):
            # Get data for this bin
            mask = df[bin_col] == bin_value
            bin_scores = df.loc[mask, score_col]
            bin_min = bin_scores.min()
            bin_max = bin_scores.max()
            
            # Store for bar chart
            bin_data.append({
                'bin': bin_value,
                'color': colors[i],
                'count': len(bin_scores),
                'min': bin_min,
                'max': bin_max
            })
            bin_ranges.append((bin_min, bin_max))
            
            # Plot histogram for this bin
            if len(bin_scores) > 0:
                sns.histplot(
                    bin_scores,
                    stat="density",
                    alpha=0.5,
                    color=colors[i],
                    ax=ax1,
                    label=f"Bin {bin_value}"
                )
        
        # Style the main plot
        ax1.set_xlabel("")  # No x-label since it's shared with the bar below
        ax1.set_ylabel("Density", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Remove x-axis labels since they'll be on the bar chart
        ax1.set_xticklabels([])
        
        # Create the bin indicator bar
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(0, 1)
        
        # Draw colored segments for each bin - NO LABELS ON BAR
        for i, bin_info in enumerate(bin_data):
            bin_min = bin_info['min']
            bin_max = bin_info['max']
            color = bin_info['color']
            
            # Draw rectangle for this bin
            ax2.axvspan(bin_min, bin_max, 0, 1, color=color, alpha=0.7)
            
            # # Draw dividers (vertical lines) at bin boundaries
            # if i > 0:  # Don't draw divider at the very beginning
            #    ax2.axvline(x=bin_min, color='black', linewidth=1, ymin=0, ymax=1)
        
        # Style the bin indicator
        ax2.set_xlabel(f"Complexity Score (Normalized)", fontsize=14)
        ax2.set_yticks([])
        ax2.grid(False)
        
        # Add a note about the bin indicator with extra spacing from x-axis label
        fig.text(
            0.02, 0.005,  # Lower y-position to create gap from x-axis label
            "* Colored bar shows bin ranges", 
            fontsize=10, style='italic'
        )
        
        # Adjust spacing between subplots and add padding at bottom for the note
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05, bottom=0.08)  # Add more bottom space for the note
        
        # Save the figure
        filepath = self.subdirs['distributions'] / f"complexity_{method_name.lower()}_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved {method_name} distribution plot to {filepath}")
        return str(filepath)
    
    def plot_bin_ranges(self, df, score_col, bin_col, method_name):
        """Create a plot showing the distribution of scores within each bin."""
        # Create a new DataFrame for the plot
        plot_df = pd.DataFrame({
            'Bin': df[bin_col],
            'Score': df[score_col]
        })
        
        # Get unique bin values and sort them
        unique_bins = sorted(plot_df['Bin'].unique())
        
        # Use turbo colormap for consistency with distribution plot
        cmap = LinearSegmentedColormap.from_list(
            "green_red",
            ["green", "red"],
            N=len(unique_bins)
        )

        # Generate colors for each bin
        colors = [cmap(i/len(unique_bins)) for i in range(len(unique_bins))]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create violin plot with boxplot inside to show the distribution
        violin_parts = ax.violinplot(
            [plot_df[plot_df['Bin'] == bin_value]['Score'] for bin_value in unique_bins],
            positions=range(len(unique_bins)),
            showmeans=False,
            showextrema=False,
            showmedians=False
        )
        
        # Customize violin colors
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        
        # Add boxplots on top
        box_parts = ax.boxplot(
            [plot_df[plot_df['Bin'] == bin_value]['Score'] for bin_value in unique_bins],
            positions=range(len(unique_bins)),
            widths=0.1,
            patch_artist=True,
            boxprops=dict(facecolor='white', alpha=0.8),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5, color='black')
        )
        
        # Style the plot
        plt.xlabel("Bin Value", fontsize=14)
        plt.ylabel("Complexity Score", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-tick labels to bin values
        ax.set_xticks(range(len(unique_bins)))
        ax.set_xticklabels(unique_bins, rotation=45 if len(unique_bins) > 10 else 0)
        
        # Add minimal statistics that won't overlap
        for i, bin_value in enumerate(unique_bins):
            bin_data = plot_df[plot_df['Bin'] == bin_value]['Score']
            if len(bin_data) > 0:
                # Only show sample size at the top of each violin
                stats_text = f"n={len(bin_data)}"
                
                # Position text just above the violin, but ensure it's inside plot bounds
                y_pos = min(bin_data.max() + 2, ax.get_ylim()[1] - 5)
                
                ax.annotate(
                    stats_text,
                    xy=(i, y_pos),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5, edgecolor='grey')
                )
        
        # Ensure all text is within bounds
        plt.tight_layout()
        
        # Save the figure
        filepath = self.subdirs['bin_ranges'] / f"complexity_{method_name.lower()}_bin_ranges.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved {method_name} bin ranges plot to {filepath}")
        return str(filepath)


class Phase5Pipeline:
    """Generates complexity scores using a trained model pipeline."""
    
    def __init__(
        self,
        input_file: str,
        model_dir: str,
        output_dir: str = None,
        score_name: str = "complexity_score",
        num_bins: int = 21
    ):
        """Initialize with input/output directories and model directory."""
        self.input_file = Path(input_file)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_file.parent / "phase5"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.score_name = score_name
        self.num_bins = num_bins
        
        # Data and model attributes
        self.df = None
        self.pipeline = None
        self.features = []
        self.scores = None
        
        # Initialize visualizer
        self.visualizer = ComplexityVisualizer(self.output_dir / "vis")
    
    def load_data(self):
        """Load the filtered data from Phase 4."""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return False
        
        self.df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(self.df)} samples from {self.input_file}")
        return True
    
    def load_model(self):
        """Load the model pipeline and features."""
        # Load pipeline
        pipeline_path = self.model_dir / "complexity_pipeline.joblib"
        if not pipeline_path.exists():
            logger.error(f"Pipeline file not found: {pipeline_path}")
            return False
        
        self.pipeline = joblib.load(pipeline_path)
        
        # Load features
        features_path = self.model_dir / "model_features.json"
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            return False
        
        self.features = load_json(features_path).get("features", [])
        
        logger.info(f"Loaded model with {len(self.features)} features")
        return True
    
    def compute_scores(self):
        """Apply the pipeline to compute complexity scores."""
        # Verify required features are present
        missing_features = [f for f in self.features if f not in self.df.columns]
        
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        
        # Apply the model pipeline to get scores
        self.scores = self.pipeline.decision_function(self.df[self.features])
        logger.info(f"Computed {len(self.scores)} complexity scores")
        return True
    
    def process_scores(self):
        """Process raw scores into normalized, discretized, and percentile versions."""
        if self.scores is None or len(self.scores) == 0:
            logger.error("No scores to process")
            return False
        
        # Create normalized scores (0-100 scale)
        normalized_scores = ScoreProcessor.normalize_scores(self.scores)
        
        # Create discretized scores (multiples of 5)
        discretized_scores = ScoreProcessor.discretize_scores(normalized_scores)
        
        # Create percentile scores
        percentile_scores = ScoreProcessor.convert_to_percentiles(self.scores)
        
        # Labels for binned scores (0, 5, 10, ..., 100)
        bin_labels = list(range(0, 105, 5))
        
        # Create binned scores using quantile method
        quantile_binned = ScoreProcessor.bin_quantile(self.scores, self.num_bins, labels=bin_labels)
        
        # Add all score versions to the dataframe
        self.df[f"{self.score_name}_raw"] = self.scores
        self.df[f"{self.score_name}_normalized"] = normalized_scores
        self.df[f"{self.score_name}_discretized"] = discretized_scores
        self.df[f"{self.score_name}_percentile"] = percentile_scores
        self.df[f"{self.score_name}_percentile_discretized"] = ScoreProcessor.discretize_scores(percentile_scores)
        self.df[f"{self.score_name}_quantile"] = quantile_binned
        
        # Generate context_id from sample_id and dataset for training
        # Extract question ID (e.g., Q1 from Q1_A1)
        self.df["question_id"] = self.df["sample_id"].str.extract(r'(Q\d+)')
        # Extract answer ID (e.g., A1 from Q1_A1)
        self.df["answer_id"] = self.df["sample_id"].str.extract(r'_(A\d+)')
        # Combine with dataset for a unique context identifier
        self.df["context_id"] = self.df["dataset"] + "_" + self.df["question_id"]
        
        logger.info(f"Added raw, normalized, discretized, percentile, and binned scores to dataframe")
        logger.info(f"Generated context_ids from sample_id and dataset for {len(self.df['context_id'].unique())} unique contexts")
        
        return True
    
    def save_results(self):
        """Save the results to CSV and generate summary statistics."""
        output_path = self.output_dir / "scores_df.csv"
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved scored dataframe to {output_path}")
        
        # Save a sample if large
        if len(self.df) > 100:
            sample_path = self.output_dir / "scores_df_sample.csv"
            self.df.head(100).to_csv(sample_path, index=False)
            logger.info(f"Saved sample of 100 rows to {sample_path}")
        
        # Generate summary statistics
        self._save_summary()
        
        return True
    
    def _save_summary(self):
        """Save summary statistics of the scores."""
        # Initialize the base summary structure
        summary = self._create_base_summary()
        
        # Add bin threshold information for all binning methods
        self._add_bin_thresholds(summary)
        
        # Add detailed bin statistics (mean, median, range)
        self._add_detailed_bin_statistics(summary)
        
        # Add statistical tests between adjacent bins
        self._add_bin_comparison_tests(summary)
        
        # Add additional statistics if available
        self._add_group_statistics(summary)
        
        # Save as JSON
        summary_path = self.output_dir / "summary.json"
        save_json(summary, summary_path)
        logger.info(f"Saved score summary to {summary_path}")
        
        return True

    def _create_base_summary(self):
        """Create the base summary structure with overall statistics."""
        return {
            "count": len(self.scores),
            "raw_scores": {
                "min": np.min(self.scores),
                "max": np.max(self.scores),
                "mean": np.mean(self.scores),
                "median": np.median(self.scores),
                "std": np.std(self.scores)
            },
            "normalized_scores": {
                "min": np.min(self.df[f"{self.score_name}_normalized"]),
                "max": np.max(self.df[f"{self.score_name}_normalized"]),
                "mean": np.mean(self.df[f"{self.score_name}_normalized"]),
                "median": np.median(self.df[f"{self.score_name}_normalized"]),
                "std": np.std(self.df[f"{self.score_name}_normalized"])
            },
            "discretized_scores": {
                "distribution": self.df[f"{self.score_name}_discretized"].value_counts().sort_index().to_dict()
            },
            "percentile_scores": {
                "min": np.min(self.df[f"{self.score_name}_percentile"]),
                "max": np.max(self.df[f"{self.score_name}_percentile"]),
                "mean": np.mean(self.df[f"{self.score_name}_percentile"]),
                "median": np.median(self.df[f"{self.score_name}_percentile"]),
                "std": np.std(self.df[f"{self.score_name}_percentile"])
            },
            "percentile_discretized_scores": {
                "distribution": self.df[f"{self.score_name}_percentile_discretized"].value_counts().sort_index().to_dict()
            },
            "quantile_scores": {
                "num_bins": self.num_bins,
                "distribution": self.df[f"{self.score_name}_quantile"].value_counts().sort_index().to_dict()
            }
        }

    def _add_bin_thresholds(self, summary):
        """Add threshold information for each binning method."""
        # 1. Quantile binning thresholds
        quantile_bins = sorted(self.df[f"{self.score_name}_quantile"].unique())
        quantile_thresholds = self._calculate_bin_thresholds(
            quantile_bins, 
            f"{self.score_name}_quantile", 
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"],
            ["raw", "normalized"]
        )
        summary["quantile_scores"]["bin_thresholds"] = quantile_thresholds
        
        # 2. Discretized scores thresholds
        discretized_bins = sorted(self.df[f"{self.score_name}_discretized"].unique())
        discretized_thresholds = self._calculate_bin_thresholds(
            discretized_bins, 
            f"{self.score_name}_discretized", 
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"],
            ["raw", "normalized"]
        )
        summary["discretized_scores"]["bin_thresholds"] = discretized_thresholds
        
        # 3. Percentile discretized scores thresholds
        percentile_bins = sorted(self.df[f"{self.score_name}_percentile_discretized"].unique())
        percentile_thresholds = self._calculate_bin_thresholds(
            percentile_bins, 
            f"{self.score_name}_percentile_discretized", 
            [f"{self.score_name}_raw", f"{self.score_name}_normalized", f"{self.score_name}_percentile"],
            ["raw", "normalized", "percentile"]
        )
        summary["percentile_discretized_scores"]["bin_thresholds"] = percentile_thresholds

    def _calculate_bin_thresholds(self, bin_values, bin_column, threshold_columns, threshold_names):
        """Calculate min/max thresholds for each bin across multiple score types."""
        bin_thresholds = {}
        
        for bin_value in bin_values:
            bin_data = self.df[self.df[bin_column] == bin_value]
            
            if not bin_data.empty:
                thresholds = {}
                
                # Calculate min/max for each threshold type
                for col, name in zip(threshold_columns, threshold_names):
                    thresholds[f"{name}_min"] = bin_data[col].min()
                    thresholds[f"{name}_max"] = bin_data[col].max()
                    
                bin_thresholds[str(bin_value)] = thresholds
                
        return bin_thresholds
    
    def _add_detailed_bin_statistics(self, summary):
        """Add detailed statistics for each bin (mean, median, range)."""
        # Calculate detailed statistics for each binning method
        
        # 1. Quantile binning statistics
        quantile_bins = sorted(self.df[f"{self.score_name}_quantile"].unique())
        quantile_stats = self._calculate_detailed_bin_statistics(
            quantile_bins,
            f"{self.score_name}_quantile",
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"]
        )
        summary["quantile_scores"]["bin_statistics"] = quantile_stats
        
        # 2. Discretized scores statistics
        discretized_bins = sorted(self.df[f"{self.score_name}_discretized"].unique())
        discretized_stats = self._calculate_detailed_bin_statistics(
            discretized_bins,
            f"{self.score_name}_discretized",
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"]
        )
        summary["discretized_scores"]["bin_statistics"] = discretized_stats
        
        # 3. Percentile discretized scores statistics
        percentile_bins = sorted(self.df[f"{self.score_name}_percentile_discretized"].unique())
        percentile_stats = self._calculate_detailed_bin_statistics(
            percentile_bins,
            f"{self.score_name}_percentile_discretized",
            [f"{self.score_name}_raw", f"{self.score_name}_normalized", f"{self.score_name}_percentile"]
        )
        summary["percentile_discretized_scores"]["bin_statistics"] = percentile_stats
        
        logger.info("Added detailed bin statistics (mean, median, range) for all binning methods")
    
    def _calculate_detailed_bin_statistics(self, bin_values, bin_column, score_columns):
        """Calculate detailed statistics for each bin."""
        bin_statistics = {}
        
        for bin_value in bin_values:
            bin_data = self.df[self.df[bin_column] == bin_value]
            
            if not bin_data.empty:
                bin_stats = {
                    "count": len(bin_data)
                }
                
                # Calculate statistics for each score column
                for col in score_columns:
                    col_name = col.split('_')[-1]  # Extract score type (raw, normalized, etc.)
                    
                    # Calculate statistics
                    values = bin_data[col]
                    bin_stats[f"{col_name}_mean"] = np.mean(values)
                    bin_stats[f"{col_name}_median"] = np.median(values)
                    bin_stats[f"{col_name}_range"] = np.max(values) - np.min(values)
                    bin_stats[f"{col_name}_std"] = np.std(values)
                    bin_stats[f"{col_name}_q1"] = np.percentile(values, 25)
                    bin_stats[f"{col_name}_q3"] = np.percentile(values, 75)
                
                bin_statistics[str(bin_value)] = bin_stats
        
        return bin_statistics
    
    def _add_bin_comparison_tests(self, summary):
        """Add statistical tests comparing adjacent bins."""
        # Calculate statistical tests for each binning method
        
        # 1. Quantile binning comparisons
        quantile_bins = sorted(self.df[f"{self.score_name}_quantile"].unique())
        quantile_tests = self._calculate_bin_comparison_tests(
            quantile_bins,
            f"{self.score_name}_quantile",
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"]
        )
        summary["quantile_scores"]["bin_comparisons"] = quantile_tests
        
        # 2. Discretized scores comparisons
        discretized_bins = sorted(self.df[f"{self.score_name}_discretized"].unique())
        discretized_tests = self._calculate_bin_comparison_tests(
            discretized_bins,
            f"{self.score_name}_discretized",
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"]
        )
        summary["discretized_scores"]["bin_comparisons"] = discretized_tests
        
        # 3. Percentile discretized scores comparisons
        percentile_bins = sorted(self.df[f"{self.score_name}_percentile_discretized"].unique())
        percentile_tests = self._calculate_bin_comparison_tests(
            percentile_bins,
            f"{self.score_name}_percentile_discretized",
            [f"{self.score_name}_raw", f"{self.score_name}_normalized", f"{self.score_name}_percentile"]
        )
        summary["percentile_discretized_scores"]["bin_comparisons"] = percentile_tests
        
        logger.info("Added statistical tests comparing adjacent bins for all binning methods")
        
        # 4. ANOVA tests across all bins
        anova_tests = self._calculate_anova_tests(
            [
                (f"{self.score_name}_quantile", "quantile"),
                (f"{self.score_name}_discretized", "discretized"),
                (f"{self.score_name}_percentile_discretized", "percentile_discretized")
            ],
            [f"{self.score_name}_raw", f"{self.score_name}_normalized"]
        )
        summary["anova_tests"] = anova_tests
        
        logger.info("Added ANOVA tests across all bins for all binning methods")
    
    def _calculate_bin_comparison_tests(self, bin_values, bin_column, score_columns):
        """Calculate t-tests between adjacent bins."""
        if len(bin_values) < 2:
            logger.warning(f"Insufficient bins for comparison tests: {len(bin_values)} bins for {bin_column}")
            return {}
        
        comparison_tests = {}
        
        # Compare adjacent bins
        for i in range(len(bin_values) - 1):
            bin1 = bin_values[i]
            bin2 = bin_values[i + 1]
            
            bin1_data = self.df[self.df[bin_column] == bin1]
            bin2_data = self.df[self.df[bin_column] == bin2]
            
            # Skip if either bin is empty
            if bin1_data.empty or bin2_data.empty:
                continue
            
            comparison_key = f"{bin1}_vs_{bin2}"
            comparison_tests[comparison_key] = {}
            
            # Calculate t-tests for each score column
            for col in score_columns:
                col_name = col.split('_')[-1]  # Extract score type (raw, normalized, etc.)
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    bin1_data[col],
                    bin2_data[col],
                    equal_var=False  # Use Welch's t-test (don't assume equal variance)
                )
                
                # Store results
                comparison_tests[comparison_key][col_name] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,  # Flag if statistically significant at α=0.05
                    "bin1_mean": bin1_data[col].mean(),
                    "bin2_mean": bin2_data[col].mean(),
                    "mean_difference": bin2_data[col].mean() - bin1_data[col].mean()
                }
        
        return comparison_tests
    
    def _calculate_anova_tests(self, bin_columns, score_columns):
        """Calculate ANOVA tests across all bins for each binning method."""
        anova_results = {}
        
        for bin_col, bin_name in bin_columns:
            anova_results[bin_name] = {}
            
            for score_col in score_columns:
                score_name = score_col.split('_')[-1]  # Extract score type (raw, normalized, etc.)
                
                # Create groups based on bin values
                groups = []
                bin_values = sorted(self.df[bin_col].unique())
                
                for bin_value in bin_values:
                    bin_data = self.df[self.df[bin_col] == bin_value][score_col]
                    if not bin_data.empty:
                        groups.append(bin_data.values)
                
                # Skip if fewer than 2 groups
                if len(groups) < 2:
                    continue
                
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Store results
                anova_results[bin_name][score_name] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,  # Flag if statistically significant at α=0.05
                    "num_bins": len(groups)
                }
        
        return anova_results

    def _add_group_statistics(self, summary):
        """Add statistics grouped by dataset, model, and complexity level if available."""
        # Add dataset statistics if available
        if "dataset" in self.df.columns:
            summary["dataset_statistics"] = self._calculate_group_statistics("dataset")
        
        # Add model statistics if available
        if "model_used" in self.df.columns:
            summary["model_statistics"] = self._calculate_group_statistics("model_used")
        
        # Add complexity level statistics if available
        if "complexity_level" in self.df.columns:
            summary["complexity_level_statistics"] = self._calculate_group_statistics("complexity_level")

    def _calculate_group_statistics(self, group_column):
        """Calculate statistics for each value in the specified grouping column."""
        group_stats = {}
        
        for group_value in self.df[group_column].dropna().unique():
            # Convert to string for JSON compatibility if needed
            group_key = str(group_value)
            group_df = self.df[self.df[group_column] == group_value]
            
            group_stats[group_key] = {
                "count": len(group_df),
                "mean_raw": group_df[f"{self.score_name}_raw"].mean(),
                "mean_normalized": group_df[f"{self.score_name}_normalized"].mean(),
                "mean_percentile": group_df[f"{self.score_name}_percentile"].mean(),
                "discretized_distribution": group_df[f"{self.score_name}_discretized"].value_counts().sort_index().to_dict(),
                "percentile_discretized_distribution": group_df[f"{self.score_name}_percentile_discretized"].value_counts().sort_index().to_dict(),
                "quantile_distribution": group_df[f"{self.score_name}_quantile"].value_counts().sort_index().to_dict()
            }
        
        return group_stats
    
    def create_visualizations(self):
        """Create visualizations for the complexity scores."""
        logger.info("Creating visualizations for complexity scores")
        viz_paths = self.visualizer.create_all_visualizations(self.df, self.score_name)
        
        if not viz_paths:
            logger.warning("No visualizations were created")
            return False
        
        logger.info(f"Created {len(viz_paths)} visualizations")
        return True
    
    def run(self):
        """Execute the pipeline steps and return the processed dataframe."""
        logger.info("Starting Phase 5: Complexity Scoring")
        
        # Load data
        if not self.load_data():
            return pd.DataFrame()
        
        # Load model
        if not self.load_model():
            return pd.DataFrame()
        
        # Compute scores
        if not self.compute_scores():
            return pd.DataFrame()
        
        # Process scores
        if not self.process_scores():
            return pd.DataFrame()
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        logger.info("Phase 5 complete")
        return self.df


def main():
    """Command-line interface for running Phase 5."""
    parser = argparse.ArgumentParser(description="Phase 5: Complexity Scoring")
    parser.add_argument("--input-file", required=True, help="Path to the filtered CSV from Phase 4")
    parser.add_argument("--model-dir", required=True, help="Directory containing model pipeline and features")
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--score-name", default="complexity_score", help="Base name for score columns")
    parser.add_argument("--num-bins", type=int, default=21, help="Number of bins to divide the data into (default: 21)")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip generating visualizations")
    
    args = parser.parse_args()
    
    pipeline = Phase5Pipeline(
        input_file=args.input_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        score_name=args.score_name,
        num_bins=args.num_bins
    )
    
    result_df = pipeline.run()
    
    if result_df.empty:
        logger.error("Phase 5 encountered errors")
        return 1
    
    logger.info(f"Phase 5 completed successfully with {len(result_df)} processed samples")
    return 0


if __name__ == "__main__":
    main()