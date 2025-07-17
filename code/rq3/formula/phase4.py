import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils.helpers import setup_logging, save_json

# Initialize logging
logger = setup_logging()



class ScoreThresholder:
    """Applies raw complexity score thresholds to bin scores."""
    
    def __init__(self, thresholds_file: Path) -> None:
        """Initialize with thresholds file."""
        self.thresholds_file = Path(thresholds_file)
        self.thresholds = self._load_thresholds()
        
    def _load_thresholds(self) -> dict:
        """Load thresholds from JSON file."""
        from utils.helpers import load_json  # Only import here since we need it
        
        if not self.thresholds_file.exists():
            logger.error(f"Thresholds file not found: {self.thresholds_file}")
            return {}
        
        thresholds_data = load_json(self.thresholds_file)
        
        if "bin_thresholds" in thresholds_data:
            return thresholds_data["bin_thresholds"]
            
        if "quantile_scores" in thresholds_data and "bin_thresholds" in thresholds_data["quantile_scores"]:
            return thresholds_data["quantile_scores"]["bin_thresholds"]
            
        logger.error(f"No bin_thresholds found in {self.thresholds_file}")
        return {}
        
    def bin_score(self, score) -> int:
        """Convert a raw score to a binned score using thresholds."""
        if not self.thresholds:
            logger.error("No thresholds available for binning")
            return None
        
        for bin_value, threshold in self.thresholds.items():
            if threshold["raw_min"] <= score <= threshold["raw_max"]:
                return int(bin_value)
        
        logger.warning(f"Score {score} does not fall within any bin range")
        return None


class ComplexityVisualizer:
    """Creates visualizations comparing complexity scores."""
    
    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory."""
        self.base_output_dir = Path(output_dir)
        
        # Create visualization directories
        self.vis_dir = self.base_output_dir / "vis"
        self.vis_all_dir = self.vis_dir / "all"
        self.vis_combinations_dir = self.vis_dir / "combinations"
        self.vis_individual_dir = self.vis_dir / "individual"
        
        # Create directories
        for directory in [self.vis_dir, self.vis_all_dir, self.vis_combinations_dir, self.vis_individual_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
        
        # Standard model names mapping with variations
        self.model_names = {
            "baseline": "Llama 3.1 8B",
            "fewshot": "Llama 3.1 8B (Few-shot)",
            "few-shot": "Llama 3.1 8B (Few-shot)",  # Added hyphenated version
            "finetuned": "Llama 3.1 8B (Fine-tuned)",
            "finetuned-nl": "Llama 3.1 8B (Instruction-tuned)",
            "claude-api": "Claude 3.7 Sonnet"
        }
        
        # Define the order for models in the legend
        self.model_order = [
            "baseline", 
            "fewshot", 
            "few-shot",
            "finetuned", 
            "finetuned-nl", 
            "claude-api"
        ]
        
        # Map for ordering display names in the legend
        self.display_name_order = {self.model_names.get(model, model): i for i, model in enumerate(self.model_order)}
    
    def create_visualizations(self, df: pd.DataFrame) -> bool:
        """Create all visualizations for the dataset."""
        if df.empty:
            logger.error("Cannot create visualizations: DataFrame is empty")
            return False
        
        # Filter out rows where binned_score is None (couldn't be binned)
        valid_df = df.dropna(subset=["binned_score"])
        if len(valid_df) == 0:
            logger.error("No valid binned scores found, cannot create visualizations")
            return False
            
        if len(valid_df) < len(df):
            logger.warning(f"Filtered out {len(df) - len(valid_df)} rows with no valid binned score")
        
        # Create standard comparison plots for all data
        self._create_model_comparison_plot(valid_df, output_dir=self.vis_all_dir, filename="complexity_model_comparison.png")
        # Add new plot without confidence intervals
        self._create_model_comparison_plot_no_ci(valid_df, output_dir=self.vis_all_dir, filename="complexity_model_comparison_no_ci.png")
        self._create_distribution_plot(valid_df, output_dir=self.vis_all_dir, filename="raw_score_distribution.png")
        
        # Create dataset-specific visualizations if available
        if "dataset_name" in valid_df.columns:
            self._create_dataset_visualizations(valid_df)
            self._create_dataset_combination_visualizations(valid_df)
        
        logger.info("Created all visualizations")
        return True
    
    def _create_model_comparison_plot(self, df: pd.DataFrame, output_dir: Path, filename: str) -> bool:
        """Create lineplot comparing expected vs. actual binned complexity by model type."""
        # Prepare data for plotting
        required_columns = ["complexity_level", "model", "binned_score"]
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns for model comparison plot")
            return False
        
        plot_df = df[required_columns].copy()
        plot_df["model_name"] = plot_df["model"].map(lambda x: self.model_names.get(x, x))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort display names according to the defined order
        # First, get unique model names in the dataframe
        model_names_in_df = list(plot_df["model_name"].unique())
        
        # Sort them according to our predefined order
        sorted_model_names = sorted(
            model_names_in_df, 
            key=lambda x: self.display_name_order.get(x, float('inf'))
        )
        
        # Create lineplot for binned scores with ordered hue
        sns.lineplot(
            data=plot_df,
            x="complexity_level",
            y="binned_score",
            hue="model_name",
            style="model_name",
            markers=['o'] * len(sorted_model_names),  # Use same marker for all models
            markersize=6,
            dashes=False,
            ci="sd",
            ax=ax,
            hue_order=sorted_model_names,
            style_order=sorted_model_names
        )
        
        # Add reference line for perfect alignment
        ax.plot(
            [0, 100],
            [0, 100],
            linestyle="--",
            color="gray",
            alpha=0.7,
            label="Perfect alignment"
        )
        
        # Style the plot
        ax.set_xlabel("Target Complexity Level", fontsize=14)
        ax.set_ylabel("Generated Complexity Level", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Remove legend title
        handles, labels = ax.get_legend_handles_labels()
        # Remove "Perfect alignment" from sorting
        if "Perfect alignment" in labels:
            perf_index = labels.index("Perfect alignment")
            perfect_handle = handles.pop(perf_index)
            perfect_label = labels.pop(perf_index)
            # Sort remaining handles and labels
            sorted_indices = sorted(range(len(labels)), 
                                   key=lambda i: self.display_name_order.get(labels[i], float('inf')))
            handles = [handles[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
            # Add "Perfect alignment" back at the end
            handles.append(perfect_handle)
            labels.append(perfect_label)
        
        ax.legend(handles=handles, labels=labels, title=None)
        
        # Save the figure
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved model comparison plot to {filepath}")
        return True
    
    def _create_model_comparison_plot_no_ci(self, df: pd.DataFrame, output_dir: Path, filename: str) -> bool:
        """Create lineplot comparing expected vs. actual binned complexity by model type without confidence intervals."""
        # Prepare data for plotting
        required_columns = ["complexity_level", "model", "binned_score"]
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns for model comparison plot")
            return False
        
        plot_df = df[required_columns].copy()
        plot_df["model_name"] = plot_df["model"].map(lambda x: self.model_names.get(x, x))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort display names according to the defined order
        # First, get unique model names in the dataframe
        model_names_in_df = list(plot_df["model_name"].unique())
        
        # Sort them according to our predefined order
        sorted_model_names = sorted(
            model_names_in_df, 
            key=lambda x: self.display_name_order.get(x, float('inf'))
        )
        
        # Create lineplot for binned scores WITHOUT confidence intervals and with ordered hue
        sns.lineplot(
            data=plot_df,
            x="complexity_level",
            y="binned_score",
            hue="model_name",
            style="model_name",
            markers=['o'] * len(sorted_model_names),  # Use same marker for all models
            dashes=False,
            ci=None,  # No confidence intervals
            ax=ax,
            hue_order=sorted_model_names,
            style_order=sorted_model_names
        )
        
        # Add reference line for perfect alignment
        ax.plot(
            [0, 100],
            [0, 100],
            linestyle="--",
            color="gray",
            alpha=0.7,
            label="Perfect alignment"
        )
        
        # Style the plot
        ax.set_xlabel("Target Complexity Level", fontsize=14)
        ax.set_ylabel("Generated Complexity Level", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Remove legend title
        handles, labels = ax.get_legend_handles_labels()
        # Remove "Perfect alignment" from sorting
        if "Perfect alignment" in labels:
            perf_index = labels.index("Perfect alignment")
            perfect_handle = handles.pop(perf_index)
            perfect_label = labels.pop(perf_index)
            # Sort remaining handles and labels
            sorted_indices = sorted(range(len(labels)), 
                                   key=lambda i: self.display_name_order.get(labels[i], float('inf')))
            handles = [handles[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
            # Add "Perfect alignment" back at the end
            handles.append(perfect_handle)
            labels.append(perfect_label)
        
        ax.legend(handles=handles, labels=labels, title=None)
        
        # Save the figure
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved model comparison plot (no CI) to {filepath}")
        return True
    
    def _create_distribution_plot(self, df: pd.DataFrame, output_dir: Path, filename: str) -> bool:
        """Create distribution plot showing raw scores by model type."""
        # Check required columns
        required_columns = ["model", "raw_score", "binned_score"]
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns for distribution plot")
            return False
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort model types according to the defined order
        model_types = list(df["model"].unique())
        sorted_model_types = sorted(
            model_types,
            key=lambda x: self.model_order.index(x) if x in self.model_order else float('inf')
        )
        
        # Create distplot for each model type in the correct order
        for model_type in sorted_model_types:
            model_data = df[df["model"] == model_type]["raw_score"]
            display_name = self.model_names.get(model_type, model_type)
            
            sns.kdeplot(
                model_data,
                label=display_name,
                fill=True,
                alpha=0.5,
                ax=ax
            )
        
        # Add vertical lines for bin thresholds
        thresholds = self._get_bin_thresholds(df)
        
        ymin, ymax = ax.get_ylim()
        for bin_val, threshold in thresholds:
            ax.axvline(threshold, ymin=0, ymax=0.9, color='gray', linestyle='--', alpha=0.5)
            ax.text(threshold, ymax * 0.95, f'{bin_val}', ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Style the plot
        ax.set_xlabel("Raw Model Score", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Fix legend order
        handles, labels = ax.get_legend_handles_labels()
        # Create a mapping from labels to their intended order
        label_to_order = {self.model_names.get(model, model): i 
                         for i, model in enumerate(self.model_order) 
                         if self.model_names.get(model, model) in labels}
        # Sort the handles and labels
        sorted_indices = sorted(range(len(labels)), 
                               key=lambda i: label_to_order.get(labels[i], float('inf')))
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        ax.legend(handles=handles, labels=labels, title=None)  # No legend title
        
        # Save the figure
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved distribution plot to {filepath}")
        return True
    
    def _get_bin_thresholds(self, df: pd.DataFrame) -> list:
        """Extract bin thresholds from the data."""
        thresholds = []
        
        for bin_val in sorted(set(df["binned_score"].dropna())):
            bin_data = df[df["binned_score"] == bin_val]
            if len(bin_data) == 0:
                continue
                
            thresholds.append((bin_val, bin_data["raw_score"].min()))
        
        return thresholds
    
    def _create_dataset_visualizations(self, df: pd.DataFrame) -> bool:
        """Create visualizations for individual datasets."""
        if "dataset_name" not in df.columns:
            return False
            
        for dataset_name in df["dataset_name"].unique():
            dataset_df = df[df["dataset_name"] == dataset_name]
            if dataset_df.empty:
                continue
                
            # Create output directory for this dataset
            dataset_dir = self.vis_individual_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualizations
            self._create_model_comparison_plot(
                dataset_df, 
                output_dir=dataset_dir,
                filename=f"complexity_model_comparison.png"
            )
            
            # Add the new plot without confidence intervals
            self._create_model_comparison_plot_no_ci(
                dataset_df, 
                output_dir=dataset_dir,
                filename=f"complexity_model_comparison_no_ci.png"
            )
            
            self._create_distribution_plot(
                dataset_df, 
                output_dir=dataset_dir,
                filename=f"raw_score_distribution.png"
            )
        
        logger.info("Created individual dataset visualizations")
        return True
    
    def _create_dataset_combination_visualizations(self, df: pd.DataFrame) -> bool:
        """Create visualizations for combinations of datasets."""
        if "dataset_name" not in df.columns:
            return False
            
        import itertools
        
        dataset_names = list(df["dataset_name"].unique())
        if len(dataset_names) <= 1:
            return False
            
        # Generate all dataset combinations (except individual datasets and all datasets together)
        for k in range(2, len(dataset_names)):
            for combo in itertools.combinations(dataset_names, k):
                combo_name = "_vs_".join(sorted(combo))
                
                # Filter data for this combination
                combo_df = df[df["dataset_name"].isin(combo)]
                if combo_df.empty:
                    continue
                    
                # Create output directory for this combination
                combo_dir = self.vis_combinations_dir / combo_name
                combo_dir.mkdir(parents=True, exist_ok=True)
                
                # Create visualizations
                self._create_model_comparison_plot(
                    combo_df, 
                    output_dir=combo_dir,
                    filename=f"complexity_model_comparison.png"
                )
                
                # Add the new plot without confidence intervals
                self._create_model_comparison_plot_no_ci(
                    combo_df, 
                    output_dir=combo_dir,
                    filename=f"complexity_model_comparison_no_ci.png"
                )
                
                self._create_distribution_plot(
                    combo_df, 
                    output_dir=combo_dir,
                    filename=f"raw_score_distribution.png"
                )
        
        logger.info("Created dataset combination visualizations")
        return True
    
    def calculate_model_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate evaluation metrics (MAE, RMSE, correlation, R²) for each model."""
        metrics = {}
        
        # Check required columns
        required_columns = ["model", "complexity_level", "binned_score"]
        if not all(col in df.columns for col in required_columns):
            logger.warning("Missing required columns for metrics calculation")
            return metrics
        
        # Calculate metrics for each model
        for model_type in df["model"].unique():
            # Filter for this model and remove any rows with NaN values
            model_df = df[df["model"] == model_type].dropna(subset=["complexity_level", "binned_score"])
            
            if len(model_df) <= 1:
                logger.warning(f"Not enough valid data points for model {model_type}")
                continue
                
            # Get true and predicted values
            y_true = model_df["complexity_level"]
            y_pred = model_df["binned_score"]
            
            # Calculate metrics
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            # Calculate R-squared
            y_mean = np.mean(y_true)
            tss = np.sum((y_true - y_mean) ** 2)
            rss = np.sum((y_true - y_pred) ** 2)
            r2 = 1 - (rss / tss) if tss > 0 else 0
            
            # Add to metrics dictionary
            model_name = self.model_names.get(model_type, model_type)
            metrics[model_type] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "correlation": float(correlation),
                "r2": float(r2),
                "model_name": model_name,
                "sample_count": len(model_df)
            }
        
        return metrics


class Phase4Pipeline:
    """Processes data to compute, bin, and visualize complexity scores."""
    
    def __init__(self, in_file, out_dir, thresholds_file, model_dir, remove_outliers=False):
        """Initialize with input/output files and directories."""
        self.in_file = Path(in_file)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir)
        self.remove_outliers = remove_outliers
        
        # Initialize thresholder
        self.thresholder = ScoreThresholder(thresholds_file)
        
        # Data attributes
        self.df = None
        self.pipeline = None
        self.features = []
        
        # Initialize visualizer
        self.visualizer = ComplexityVisualizer(self.out_dir)
    
    def run(self):
        """Execute the pipeline steps."""
        logger.info(f"Running Phase 4 pipeline on {self.in_file}")
        
        # Load data - only support CSV
        if not self._load_csv_data():
            logger.error("Failed to load data")
            return {}
        
        # Verify that complexity_level column exists
        if "complexity_level" not in self.df.columns:
            logger.warning("No 'complexity_level' column found in dataset - metrics may be incomplete")
        else:
            logger.info(f"Found {self.df['complexity_level'].notna().sum()} samples with complexity_level data")
        
        # Load model
        if not self._load_model():
            logger.error("Failed to load model")
            return {}
        
        # Check and extract features from nested structure if needed
        if "metrics" in self.df.columns:
            self._extract_features_from_metrics()
        
        # Compute scores
        if not self._compute_scores():
            logger.error("Failed to compute scores")
            return {}
        
        # Bin scores
        if not self._bin_scores():
            logger.error("Failed to bin scores")
            return {}
        
        # Drop rows where binning failed
        if "binned_score" in self.df.columns:
            valid_count = self.df["binned_score"].notna().sum()
            original_count = len(self.df)
            
            if valid_count < original_count:
                logger.warning(f"Filtering out {original_count - valid_count} rows where binning failed")
                self.df = self.df.dropna(subset=["binned_score"])
                logger.info(f"Proceeding with {len(self.df)} valid rows")
        
        if self.remove_outliers:
            self._filter_outliers(max_diff=100)

        # Create visualizations
        self.visualizer.create_visualizations(self.df)
        
        # Save results
        self._save_results()
        
        logger.info("Phase 4 pipeline completed successfully")
        
        # Return results dictionary
        return {
            "scored_df": self.df,
            "summary": self._generate_summary()
        }
    
    def _load_csv_data(self) -> bool:
        """Load data from a CSV file."""
        if not self.in_file.exists():
            logger.error(f"Input file not found: {self.in_file}")
            return False
            
        # Check file extension
        if self.in_file.suffix.lower() != '.csv':
            logger.error(f"Input file must be a CSV file: {self.in_file}")
            return False
            
        # Load CSV file
        self.df = pd.read_csv(self.in_file)
        
        if self.df.empty:
            logger.error("CSV file is empty")
            return False
            
        logger.info(f"Loaded {len(self.df)} samples from CSV file {self.in_file}")
        return True
    
    def _load_model(self) -> bool:
        """Load the model pipeline and features."""
        # Load pipeline
        pipeline_path = self.model_dir / "complexity_pipeline.joblib"
        if not pipeline_path.exists():
            logger.error(f"Pipeline file not found: {pipeline_path}")
            return False
        
        self.pipeline = joblib.load(pipeline_path)
        if self.pipeline is None:
            logger.error("Failed to load pipeline")
            return False
        
        # Load features
        features_path = self.model_dir / "model_features.json"
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            return False
        
        from utils.helpers import load_json  # Only import here since we need it
        features_data = load_json(features_path)
        if features_data is None:
            logger.error("Failed to load features data")
            return False
        
        if "features" not in features_data:
            logger.error("No features list found in model_features.json")
            return False
            
        self.features = features_data["features"]
        
        if not self.features:
            logger.error("Features list is empty")
            return False
            
        logger.info(f"Loaded model with {len(self.features)} features")
        return True
    
    def _extract_features_from_metrics(self) -> None:
        """Extract features from the metrics dictionary in each row."""
        if "metrics" not in self.df.columns:
            return
            
        logger.info("Extracting features from metrics dictionary")
        
        for feature in self.features:
            self.df[feature] = self.df.apply(lambda row: self._get_feature_from_metrics(row["metrics"], feature), axis=1)
            
        logger.info(f"Extracted {len(self.features)} features from metrics")
    
    def _get_feature_from_metrics(self, metrics, feature_path):
        """Get a feature value from a nested metrics dictionary."""
        if not isinstance(metrics, dict):
            return None
            
        # Navigate through the metrics dictionary based on feature path
        parts = feature_path.split("/")
        value = metrics
        
        for part in parts:
            if not isinstance(value, dict):
                return None
                
            value = value.get(part)
            
        return value
    
    def _compute_scores(self) -> bool:
        """Apply the model to compute complexity scores."""
        if self.pipeline is None:
            logger.error("No model loaded, cannot compute scores")
            return False
        
        # Check for required features
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        
        # Create a copy of features for computation
        features_df = self.df[self.features].copy()
        
        # Handle NaN values
        self._handle_nan_features(features_df)
        
        # Compute raw scores
        logger.info("Computing raw complexity scores")
        self.df["raw_score"] = self.pipeline.decision_function(features_df)
        
        logger.info(f"Computed raw scores for {len(self.df)} samples")
        return True
    
    def _handle_nan_features(self, features_df) -> None:
        """Handle NaN values in features by imputing with column means."""
        nan_counts = features_df.isna().sum()
        features_with_nans = nan_counts[nan_counts > 0]
        
        if features_with_nans.empty:
            return
            
        logger.warning(f"Found {len(features_with_nans)} features with NaN values")
        
        for feature, count in features_with_nans.items():
            logger.warning(f"  {feature}: {count} NaNs")
            
            # Calculate mean value (or use 0 if all values are NaN)
            mean_value = features_df[feature].mean()
            if pd.isna(mean_value):
                mean_value = 0
                
            # Impute NaN values
            features_df[feature].fillna(mean_value, inplace=True)
            
        logger.info("Imputed all NaN values in features")
    
    def _filter_outliers(self, max_diff=10):
        """Filter out rows where the difference between target and actual complexity is > max_diff."""
        if "complexity_level" not in self.df.columns or "binned_score" not in self.df.columns:
            logger.warning("Cannot filter outliers: missing complexity_level or binned_score columns")
            return 0
        
        # Calculate absolute difference
        self.df["complexity_diff"] = abs(self.df["complexity_level"] - self.df["binned_score"])
        
        # Count and filter outliers
        outlier_mask = self.df["complexity_diff"] > max_diff
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.info(f"Removing {outlier_count} outliers with complexity difference > {max_diff}")
            
            # Log some information about the outliers being removed
            outliers_df = self.df[outlier_mask]
            if not outliers_df.empty and "model" in outliers_df.columns:
                model_counts = outliers_df["model"].value_counts()
                for model, count in model_counts.items():
                    logger.info(f"  Model '{model}': {count} outliers removed")
                
            # Remove the outliers
            self.df = self.df[~outlier_mask]
            logger.info(f"Proceeding with {len(self.df)} rows after outlier removal")
        
        return outlier_count

    def _bin_scores(self) -> bool:
        """Bin the raw scores using thresholds."""
        if "raw_score" not in self.df.columns:
            logger.error("No 'raw_score' column found, cannot bin scores")
            return False
        
        # Apply binning function
        logger.info("Binning scores using raw thresholds")
        self.df["binned_score"] = self.df["raw_score"].apply(self.thresholder.bin_score)
        
        # Check results
        binned_count = self.df["binned_score"].notna().sum()
        if binned_count == 0:
            logger.error("Failed to bin any scores")
            return False
            
        logger.info(f"Binned {binned_count} scores successfully")
        return True
    
    def _calculate_model_metrics(self, df=None) -> dict:
        """Calculate metrics (MAE, RMSE, correlation, R²) for each model."""
        if df is None:
            df = self.df
        
        # Use the visualizer's method to calculate metrics
        return self.visualizer.calculate_model_metrics(df)
    
    def _save_results(self) -> bool:
        """Save the processed DataFrame and summary statistics."""
        # Save DataFrame
        output_path = self.out_dir / "scores_df.csv"
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved scored dataframe to {output_path}")
        
        # Save summary statistics
        summary = self._generate_summary()
        summary_path = self.out_dir / "summary.json"
        save_json(summary, summary_path)
        logger.info(f"Saved score summary to {summary_path}")
        
        return True
    
    def _generate_summary(self) -> dict:
        """Generate summary statistics of the scores."""
        summary = {"count": len(self.df)}
        
        # Add score statistics
        summary.update(self._get_score_statistics())
        
        # Add model-specific statistics
        if "model" in self.df.columns:
            # Calculate model metrics
            model_metrics = self._calculate_model_metrics()
            if model_metrics:
                summary["model_metrics"] = model_metrics
        
        # Add correlation statistics (primarily between complexity_level and binned_score)
        correlations = self._calculate_correlations()
        if correlations:
            summary["correlations"] = correlations
        
        return summary
    
    def _get_score_statistics(self) -> dict:
        """Get statistics for different score types."""
        stats = {}
        
        # Raw scores
        if "raw_score" in self.df.columns:
            raw_scores = self.df["raw_score"].dropna()
            if not raw_scores.empty:
                stats["raw_scores"] = {
                    "min": float(raw_scores.min()),
                    "max": float(raw_scores.max()),
                    "mean": float(raw_scores.mean()),
                    "median": float(raw_scores.median()),
                    "std": float(raw_scores.std())
                }
        
        # Binned scores
        if "binned_score" in self.df.columns:
            binned_scores = self.df["binned_score"].dropna()
            if not binned_scores.empty:
                stats["binned_scores"] = {
                    "distribution": binned_scores.value_counts().sort_index().to_dict()
                }
        
        # Target complexity level
        if "complexity_level" in self.df.columns:
            complexity_levels = self.df["complexity_level"].dropna()
            if not complexity_levels.empty:
                stats["complexity_level"] = {
                    "min": float(complexity_levels.min()),
                    "max": float(complexity_levels.max()),
                    "mean": float(complexity_levels.mean()),
                    "median": float(complexity_levels.median()),
                    "std": float(complexity_levels.std()),
                    "distribution": complexity_levels.value_counts().sort_index().to_dict()
                }
        
        return stats
    
    def _calculate_correlations(self) -> dict:
        """Calculate correlations between different score types."""
        correlations = {}
        
        # Define pairs to correlate - focus primarily on complexity_level vs binned_score
        col_pairs = [
            ("complexity_level", "binned_score", "complexity_level_vs_binned"),
            ("complexity_level", "raw_score", "complexity_level_vs_raw")
        ]
        
        for col1, col2, name in col_pairs:
            if col1 not in self.df.columns or col2 not in self.df.columns:
                continue
                
            # Calculate correlation on non-NaN values
            valid_df = self.df[[col1, col2]].dropna()
            if len(valid_df) <= 1:  # Need at least 2 points for correlation
                continue
                
            correlation = float(valid_df[col1].corr(valid_df[col2]))
            correlations[name] = correlation
        
        return correlations


def main():
    """Command-line interface for running the Phase 4 pipeline."""
    parser = argparse.ArgumentParser(description="Phase 4 of complexity score analysis pipeline")
    parser.add_argument("--input-file", required=True, help="Path to the input CSV file")
    parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    parser.add_argument("--thresholds-file", required=True, help="Path to the JSON file containing bin thresholds")
    parser.add_argument("--model-dir", required=True, help="Directory containing model pipeline and features")
    parser.add_argument("--remove-outliers", action="store_true", 
                    help="Remove outliers where the difference between target and actual complexity is > 10")
       
    args = parser.parse_args()
    
    pipeline = Phase4Pipeline(
        in_file=args.input_file,
        out_dir=args.output_dir,
        thresholds_file=args.thresholds_file,
        model_dir=args.model_dir,
        remove_outliers=args.remove_outliers
    )
    
    results = pipeline.run()
    if not results:
        logger.error("Processing encountered errors")
        return 1
        
    logger.info("Processing completed successfully")
    return 0


if __name__ == "__main__":
    main()