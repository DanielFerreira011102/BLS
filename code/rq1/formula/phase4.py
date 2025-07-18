import argparse
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, uniform

# Import custom modules
from utils.helpers import load_json, save_json, setup_logging
from rq1.formula.impl.stats import (
    cohens_d, overlap_coefficient, kl_divergence,
    js_divergence, mutual_information, statistical_test
)

# Constants
SCALER_TYPES = {
    "z_score": StandardScaler,
    "min_max": MinMaxScaler,
    "robust": RobustScaler,
    "none": None
}

# Columns to exclude from feature processing
EXCLUDED_COLUMNS = ['id', 'text_id', 'document_id', 'text', 'dataset', 'sample_id']

# Initialize logging
logger = setup_logging()

class DatasetManager:
    """Manages datasets for training and evaluation."""
    
    def __init__(self, phase1_dir: Path):
        self.phase1_dir = Path(phase1_dir)
        self.datasets = {}
        self.train_data = pd.DataFrame()
        self.eval_datasets = {}
    
    def load_datasets(self, training_datasets: List[str], evaluation_datasets: List[str], 
                     test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
        """Load datasets for training and evaluation."""
        # Handle case with no evaluation datasets
        if not evaluation_datasets:
            train_dfs = self._load_training_datasets(training_datasets)
            if train_dfs:
                self.train_data = pd.concat(train_dfs, ignore_index=True)
                logger.info(f"Combined training data: {len(self.train_data)} rows")

            return self.train_data, {}
        
        # Process overlapping datasets (used for both training and evaluation)
        overlapping = set(training_datasets) & set(evaluation_datasets)
        logger.info(f"Overlapping datasets: {list(overlapping)}")
        
        # Load training datasets
        train_dfs = []
        for dataset in training_datasets:
            df = self._load_dataset(dataset)
            if df.empty:
                continue
                
            # Register dataset
            self.register_dataset(dataset, category="training")
                
            # For overlapping datasets, split into train/test
            if dataset in overlapping:
                train_df = self._split_overlapping_dataset(df, dataset, test_size, random_state)
                train_dfs.append(train_df)
            else:
                train_dfs.append(df)
        
        # Load evaluation-only datasets
        for dataset in evaluation_datasets:
            if dataset in overlapping:
                continue  # Already handled above
                
            df = self._load_dataset(dataset)
            if df.empty:
                continue
                
            self.register_dataset(dataset, category="evaluation")
            target = (df["text_type"] == "expert").astype(int)
            self.eval_datasets[dataset] = (df, target)
        
        # Combine training data
        if train_dfs:
            self.train_data = pd.concat(train_dfs, ignore_index=True)
            logger.info(f"Combined training data: {len(self.train_data)} rows")
        
        return self.train_data, self.eval_datasets
    
    def _load_training_datasets(self, training_datasets: List[str]) -> List[pd.DataFrame]:
        """Load and register all training datasets."""
        train_dfs = []
        
        for dataset in training_datasets:
            df = self._load_dataset(dataset)
            if df.empty:
                continue
                
            self.register_dataset(dataset, category="training")
            train_dfs.append(df)
        
        logger.info(f"Loaded {len(train_dfs)} training datasets")
        return train_dfs
    
    def _split_overlapping_dataset(self, df: pd.DataFrame, dataset: str, 
                                 test_size: float, random_state: int) -> pd.DataFrame:
        """Split a dataset for both training and evaluation."""
        # Create target variable (expert=1, simple=0)
        target = (df["text_type"] == "expert").astype(int)
        
        # Split with stratification
        train_df, test_df, train_target, test_target = train_test_split(
            df, target, test_size=test_size, random_state=random_state, stratify=target
        )
        
        # Create a test dataset with proper identifier
        test_dataset_name = f"{dataset}_test"
        test_df = test_df.copy()
        test_df["dataset"] = test_dataset_name
        test_df["original_dataset"] = dataset  # Store original dataset name
        
        # Register test split with reference to parent
        self.register_dataset(test_dataset_name, category="training_test", parent=dataset)
        self.eval_datasets[test_dataset_name] = (test_df, test_target)
        
        return train_df
    
    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset from CSV file."""
        path = self.phase1_dir / dataset_name / "metrics_df.csv"
        
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        
        if "text_type" not in df.columns:
            logger.warning(f"Missing required 'text_type' column in dataset: {dataset_name}")
            return pd.DataFrame()
        
        # Add dataset column if not present
        if "dataset" not in df.columns:
            df["dataset"] = dataset_name
        
        logger.info(f"Loaded dataset {dataset_name}: {len(df)} rows")    
        return df
    
    def register_dataset(self, dataset_name: str, category: str = "training", parent: Optional[str] = None):
        """Register dataset with metadata."""
        self.datasets[dataset_name] = {
            "display_name": parent or dataset_name,
            "category": category,
            "parent": parent
        }
    
    def get_sorted_datasets(self) -> List[str]:
        """Get datasets sorted by category for visualization."""
        def sort_key(dataset):
            info = self.datasets.get(dataset, {})
            category = info.get("category", "")
            parent = info.get("parent", "")
            
            # Order: training first, then training_test, then evaluation
            category_order = {"training": 0, "training_test": 1, "evaluation": 2}
            cat_value = category_order.get(category, 3)
            
            # For training_test datasets, sort with parent
            sort_name = parent if category == "training_test" else dataset
            
            return (cat_value, sort_name, dataset)
            
        return sorted(self.datasets.keys(), key=sort_key)


class ModelTrainer:
    """Trains text complexity model using logistic regression."""
    
    def __init__(self, random_state: int = 42, cv: int = 10, n_jobs: int = -1):
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs
        self.pipeline = None
        
    def train(self, X: pd.DataFrame, y: np.ndarray, features: List[str], 
            normalization: str = "z_score", penalty: str = "elasticnet",
            model_params: Optional[Dict] = None) -> Pipeline:
        """Train model with provided data and parameters."""
        logger.info(f"Training model with {normalization} normalization and {penalty} penalty")
        
        # Filter features that exist in X
        features = [f for f in features if f in X.columns]
        if not features:
            raise ValueError("No valid features for training")
        
        # Create pipeline steps
        steps = []
        
        # Add scaler if needed
        if normalization and normalization != 'none':
            scaler_class = SCALER_TYPES.get(normalization)
            if scaler_class:
                steps.append(("scaler", scaler_class()))
        
        # Determine solver based on penalty
        solver = "saga" if penalty in ["elasticnet", "l1"] else "lbfgs"
        
        # Train with specific parameters if provided
        if model_params:
            model = self._create_logistic_model(penalty, solver, model_params)
            steps.append(("model", model))
            
            pipeline = Pipeline(steps)
            pipeline.fit(X[features], y)
            self.pipeline = pipeline
            
            logger.info(f"Model trained with parameters: C={model_params.get('C')}, l1_ratio={model_params.get('l1_ratio', 'N/A')}")
            return pipeline
        
        # Otherwise, perform hyperparameter search
        model = LogisticRegression(
            penalty=penalty if penalty != "none" else None,
            solver=solver,
            max_iter=10000,
            random_state=self.random_state
        )
        steps.append(("model", model))
        
        # Setup and run hyperparameter search
        pipeline = self._hyperparameter_search(steps, features, X, y, penalty)
        self.pipeline = pipeline
        
        return pipeline
    
    def _create_logistic_model(self, penalty: str, solver: str, model_params: Dict) -> LogisticRegression:
        """Create a logistic regression model with specified parameters."""
        return LogisticRegression(
            penalty=penalty if penalty != "none" else None,
            solver=solver,
            C=model_params.get('C', 0.1),
            l1_ratio=model_params.get('l1_ratio', 0.5) if penalty == "elasticnet" else None,
            max_iter=10000,
            random_state=self.random_state
        )
    
    def _hyperparameter_search(self, steps: List, features: List[str], X: pd.DataFrame, 
                             y: np.ndarray, penalty: str) -> Pipeline:
        """Perform hyperparameter search to find optimal model."""
        # Create parameter distributions
        param_dist = {'model__C': loguniform(0.001, 10.0)}
        if penalty == "elasticnet":
            param_dist['model__l1_ratio'] = uniform(0.0, 1.0)
        
        # Create pipeline
        base_pipeline = Pipeline(steps)
        
        # Setup cross-validation
        cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Create and run randomized search
        search = RandomizedSearchCV(
            base_pipeline,
            param_dist,
            n_iter=30,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state
        )
        
        search.fit(X[features], y)
        
        # Log best parameters
        best_params = {
            param.replace('model__', ''): value 
            for param, value in search.best_params_.items()
        }
        logger.info(f"Best parameters from search: {best_params}")
        
        return search.best_estimator_
    
    def extract_formula(self, features: List[str], training_datasets: List[str], 
                      normalization: str) -> Dict:
        """Extract formula dictionary from trained model."""
        if not self.pipeline or "model" not in self.pipeline.named_steps:
            return {}
        
        model = self.pipeline.named_steps['model']
        
        # Extract model parameters
        model_params = {
            "C": float(model.C), 
            "penalty": model.penalty if hasattr(model, "penalty") else None
        }
        
        # Add l1_ratio if applicable
        if hasattr(model, 'l1_ratio') and model.l1_ratio is not None:
            model_params["l1_ratio"] = float(model.l1_ratio)
            
        # Create formula dictionary
        formula = {
            "intercept": float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0,
            "coefficients": {
                feature: float(coef) 
                for feature, coef in zip(features, model.coef_[0])
                if abs(coef) > 1e-5  # Only include non-zero coefficients
            },
            "features": features,
            "trained_on": training_datasets,
            "normalization": normalization,
            "hyperparameters": {
                "random_state": self.random_state,
                "model_params": model_params
            }
        }
        
        return formula


class ModelEvaluator:
    """Evaluates formula model performance on multiple datasets."""
    
    def evaluate(self, pipeline: Pipeline, features: List[str], 
            evaluation_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict:
        """Evaluate model on evaluation datasets."""
        logger.info(f"Evaluating model on {len(evaluation_data)} datasets")
        
        if not pipeline or not features:
            return {}
        
        # Results containers
        dataset_results = {}
        individual_metrics = {}
        combined_dfs = []
        combined_ys = []
        combined_scores = []
        
        # Process each dataset
        for dataset_name, (df, y) in evaluation_data.items():
            # Skip datasets with insufficient data
            if len(y) < 10 or len(np.unique(y)) < 2:
                logger.warning(f"Dataset {dataset_name} has insufficient data or classes")
                continue
            
            # Filter to features that exist in the dataset
            dataset_features = [f for f in features if f in df.columns]
            if not dataset_features:
                logger.warning(f"Dataset {dataset_name} has no matching features")
                continue
            
            # Compute scores
            scores = pipeline.decision_function(df[dataset_features])
            
            # Calculate statistics
            stats = self._compute_stats(scores, y)
            dataset_results[dataset_name] = stats
            
            # Store data for combined analysis
            combined_dfs.append(df)
            combined_ys.append(y)
            combined_scores.append(scores)
            
            logger.info(f"Dataset {dataset_name}: AUC={stats['auc']:.4f}, Cohen's d={stats['cohens_d']:.4f}")
        
        # Skip if no valid datasets
        if not combined_dfs:
            logger.warning("No valid datasets for evaluation")
            return {}
        
        # Compute combined statistics
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_y = np.concatenate(combined_ys)
        combined_scores = np.concatenate(combined_scores)
        
        # Calculate overall stats
        overall_stats = self._compute_stats(combined_scores, combined_y)
        
        # Calculate outlier statistics
        outlier_stats = self._analyze_outliers(combined_df, combined_scores)
        
        # Calculate individual feature metrics using the combined data (THIS IS THE FIX)
        combined_features = [f for f in features if f in combined_df.columns]
        individual_metrics = self._evaluate_features(combined_df, combined_y, combined_features)
        
        # Compile results
        results = {
            "overall_stats": overall_stats,
            "dataset_stats": dataset_results,
            "individual_metrics": individual_metrics,
            "outlier_stats": outlier_stats,
            "combined_df": combined_df,
            "combined_y": combined_y,
            "combined_scores": combined_scores
        }
        
        logger.info(f"Overall evaluation: AUC={overall_stats['auc']:.4f}, Cohen's d={overall_stats['cohens_d']:.4f}")
        return results
        
    def _compute_stats(self, scores: np.ndarray, y: np.ndarray) -> Dict:
        """Compute comprehensive statistics for evaluation."""
        if len(scores) != len(y) or len(scores) == 0:
            return {}
        
        # Split by class
        simple_scores = scores[y == 0]
        expert_scores = scores[y == 1]
        
        # Early return if either class is empty
        if len(simple_scores) == 0 or len(expert_scores) == 0:
            return {}
        
        # Calculate AUC
        auc = roc_auc_score(y, scores)
        
        # Calculate ROC curve and find optimal threshold
        fpr, tpr, thresholds = roc_curve(y, scores)
        j_scores = tpr - fpr  # Youden's J statistic
        best_idx = np.argmax(j_scores)
        
        # Statistical comparison between classes
        stat, pval, test_name = statistical_test(simple_scores, expert_scores)
        
        # Return comprehensive statistics
        return {
            "simple_mean": float(np.mean(simple_scores)),
            "expert_mean": float(np.mean(expert_scores)),
            "mean_difference": float(np.mean(expert_scores) - np.mean(simple_scores)),
            "simple_std": float(np.std(simple_scores)),
            "expert_std": float(np.std(expert_scores)),
            "cohens_d": float(cohens_d(simple_scores, expert_scores)),
            "auc": float(auc),
            "best_threshold": float(thresholds[best_idx]),
            "best_j": float(j_scores[best_idx]),
            "overlap_coefficient": float(overlap_coefficient(simple_scores, expert_scores)),
            "kl_simple_to_expert": float(kl_divergence(simple_scores, expert_scores)),
            "kl_expert_to_simple": float(kl_divergence(expert_scores, simple_scores)),
            "js_divergence": float(js_divergence(simple_scores, expert_scores)),
            "mutual_information": float(mutual_information(scores, y)),
            "n_simple": int(len(simple_scores)),
            "n_expert": int(len(expert_scores)),
            "test_used": test_name,
            "test_statistic": float(stat),
            "p_value": float(pval)
        }
    
    def _evaluate_features(self, df: pd.DataFrame, y: np.ndarray, features: List[str]) -> Dict:
        """Evaluate individual feature performance."""
        metrics = {}
        
        for feature in features:
            # Skip if feature not in dataframe
            if feature not in df.columns:
                continue
            
            # Get values for each class
            feature_values = df[feature].values
            simple_values = feature_values[y == 0]
            expert_values = feature_values[y == 1]
            
            # Skip if either class has no data
            if len(simple_values) == 0 or len(expert_values) == 0:
                continue
            
            # Calculate mean difference and determine if inversion needed
            mean_diff = np.mean(expert_values) - np.mean(simple_values)
            
            # Invert values if negative effect size (simple > expert)
            values_for_auc = feature_values if mean_diff > 0 else -feature_values
            
            # Calculate AUC
            auc = roc_auc_score(y, values_for_auc)
            
            # Store metrics
            metrics[feature] = {
                "mean_diff": float(mean_diff),
                "cohens_d": float(cohens_d(simple_values, expert_values)),
                "auc": float(auc),
                "scores": values_for_auc.tolist()
            }
        
        return metrics
    
    def _analyze_outliers(self, df: pd.DataFrame, scores: np.ndarray) -> Dict:
        """Analyze outliers in scores using IQR method."""
        # Calculate IQR boundaries
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        is_outlier = (scores < lower_bound) | (scores > upper_bound)
        
        # Calculate outlier statistics
        outlier_stats = {
            "thresholds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            },
            "total_outliers": int(np.sum(is_outlier)),
            "outlier_percentage": float(np.mean(is_outlier) * 100),
            "dataset_outliers": {}
        }
        
        # Calculate outliers per dataset
        for dataset in df["dataset"].unique():
            dataset_mask = df["dataset"] == dataset
            if not any(dataset_mask):
                continue
                
            dataset_outliers = is_outlier[dataset_mask]
            outlier_stats["dataset_outliers"][dataset] = {
                "count": int(np.sum(dataset_outliers)),
                "percentage": float(np.mean(dataset_outliers) * 100)
            }
        
        return outlier_stats


class Visualizer:
    """Creates visualizations for formula model results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            "formula": self.output_dir / "formula",
            "distributions": self.output_dir / "distributions",
            "roc": self.output_dir / "roc",
            "datasets": self.output_dir / "datasets"
        }
        
        # Create directories
        for path in self.subdirs.values():
            path.mkdir(exist_ok=True)
        
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    
    def create_all_visualizations(self, formula: Dict, results: Dict, 
                              dataset_manager: DatasetManager, show_title: bool = True,
                              feature_names: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Create all visualizations for model results."""
        if not formula or not results:
            return {}
        
        viz_paths = {}
        
        # Plot formula coefficients
        viz_paths["coefficients"] = self.plot_coefficients(formula, show_title=show_title, feature_names=feature_names)
        
        # Get combined data for plots
        combined_df = results.get("combined_df")
        combined_y = results.get("combined_y")
        combined_scores = results.get("combined_scores")
        
        if not self._has_valid_data(combined_df, combined_y, combined_scores):
            return viz_paths
            
        # Plot score distributions
        simple_scores = combined_scores[combined_y == 0]
        expert_scores = combined_scores[combined_y == 1]
        
        if len(simple_scores) > 0 and len(expert_scores) > 0:
            viz_paths["distributions"] = self.plot_distributions(
                simple_scores, expert_scores, results.get("overall_stats", {}), show_title=show_title
            )
        
        # Plot ROC curves
        individual_metrics = results.get("individual_metrics", {})
        viz_paths["roc"] = self.plot_roc_curves(
            combined_y, combined_scores, individual_metrics, show_title=show_title, feature_names=feature_names
        )
        
        # Plot dataset distributions
        dataset_stats = results.get("dataset_stats", {})
        viz_paths["datasets"] = self.plot_dataset_distributions(
            dataset_stats, combined_df, combined_scores, combined_y, dataset_manager, show_title=show_title
        )
        
        # Plot boxplot
        outlier_stats = results.get("outlier_stats", {})
        if outlier_stats and "thresholds" in outlier_stats:
            viz_paths["boxplot"] = self.plot_boxplot(
                combined_df, combined_scores, outlier_stats["thresholds"], dataset_manager, show_title=show_title
            )
        
        logger.info(f"Created {len(viz_paths)} visualizations")
        return viz_paths
    
    def _has_valid_data(self, df, y, scores):
        """Check if data is valid for visualization."""
        return (df is not None and len(df) > 0 and 
                y is not None and len(y) > 0 and
                scores is not None and len(scores) > 0)
    
    def plot_coefficients(self, formula: Dict, show_title: bool = True, 
                       feature_names: Optional[Dict[str, str]] = None) -> str:
        """Plot top formula coefficients as a bar chart."""
        if "coefficients" not in formula or not formula["coefficients"]:
            return ""
        
        # Sort coefficients by absolute value
        coeffs = formula["coefficients"]
        sorted_items = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 20 for readability
        top_items = sorted_items[:min(20, len(sorted_items))]
        
        # Prepare data for plotting
        features = [item[0] for item in top_items]
        values = [item[1] for item in top_items]
        
        # Apply feature name substitution if provided
        if feature_names:
            features = [feature_names.get(f, f) for f in features]
        
        # Create DataFrame with directions
        df = pd.DataFrame({
            'Feature': features,
            'Coefficient': values,
            'Direction': ['Positive' if v >= 0 else 'Negative' for v in values]
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(features) * 0.3)))
        
        # Plot bars
        sns.barplot(
            x='Coefficient',
            y='Feature',
            hue='Direction',
            data=df,
            palette={'Positive': 'green', 'Negative': 'red'},
            alpha=0.7,
            ax=ax
        )
        
        # Add value labels
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(
                width + 0.01 if width >= 0 else width - 0.01,
                p.get_y() + p.get_height()/2,
                f"{width:.4f}",
                va='center',
                ha='left' if width >= 0 else 'right',
                fontsize=8
            )
        
        # Add zero line
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Style plot
        if show_title:
            ax.set_title("Formula Coefficients", fontsize=16)
        ax.set_xlabel("Coefficient Value", fontsize=14)
        ax.set_ylabel("Feature", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Coefficient Sign", loc="lower right", fontsize=10)
        
        # Save plot
        filepath = self.subdirs['formula'] / "formula_coefficients.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        return str(filepath)
    
    def plot_distributions(self, simple_scores: np.ndarray, expert_scores: np.ndarray, 
                        stats: Dict, title: str = "Score Distribution", 
                        show_title: bool = True) -> str:
        """Plot histograms of simple and expert scores with statistics."""
        if len(simple_scores) == 0 or len(expert_scores) == 0:
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histograms
        sns.histplot(simple_scores, bins=20, alpha=0.5, color="green", 
                   edgecolor="none", label="Simple", ax=ax)
        sns.histplot(expert_scores, bins=20, alpha=0.5, color="red", 
                   edgecolor="none", label="Expert", ax=ax)
        
        # Add mean lines
        ax.axvline(np.mean(simple_scores), color="green", linestyle="--", 
                 linewidth=2, label="Simple Mean")
        ax.axvline(np.mean(expert_scores), color="red", linestyle="--", 
                 linewidth=2, label="Expert Mean")
        
        # Add stats annotation
        self._add_stats_annotation(ax, stats)
        
        # Style plot
        if show_title:
            ax.set_title(title, fontsize=16)
        ax.set_xlabel("Score", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        filepath = self.subdirs['distributions'] / "overall_distribution.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        return str(filepath)
    
    def plot_roc_curves(self, y: np.ndarray, scores: np.ndarray, 
                      individual_metrics: Dict, max_metrics: int = 5,
                      show_title: bool = True, 
                      feature_names: Optional[Dict[str, str]] = None) -> str:
        """Plot ROC curves for model and individual features."""
        if len(y) == 0 or len(scores) == 0:
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot random baseline
        ax.plot([0, 1], [0, 1], linestyle='--', color='k', alpha=0.8, 
            label="Random (AUC = 0.5)")
        
        # Plot formula ROC
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, color='blue', linewidth=2, 
            label=f"Formula (AUC = {auc:.3f})")
        
        # Plot individual metrics if provided
        if individual_metrics:
            # Sort by AUC and take top N
            sorted_metrics = sorted(
                individual_metrics.items(),
                key=lambda x: x[1]["auc"],
                reverse=True
            )[:max_metrics]
            
            # Plot each metric
            colors = sns.color_palette("husl", len(sorted_metrics))
            for i, (metric, data) in enumerate(sorted_metrics):
                metric_scores = np.array(data["scores"])
                if len(metric_scores) != len(y):
                    continue
                
                # Apply feature name substitution if provided
                display_name = feature_names.get(metric, metric) if feature_names else metric
                    
                metric_fpr, metric_tpr, _ = roc_curve(y, metric_scores)
                metric_auc = data["auc"]
                
                ax.plot(
                    metric_fpr, metric_tpr,
                    label=f"{display_name} (AUC = {metric_auc:.3f})",
                    color=colors[i],
                    linewidth=1.5,
                    alpha=0.7
                )
        
        # Style plot
        if show_title:
            ax.set_title("ROC Curves", fontsize=16)
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        filepath = self.subdirs['roc'] / "roc_curves.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        return str(filepath)
    
    def plot_dataset_distributions(self, dataset_stats: Dict, df: pd.DataFrame, 
                                    scores: np.ndarray, y: np.ndarray, 
                                    dataset_manager: DatasetManager,
                                    show_title: bool = True) -> str:
        """Plot score distributions for each dataset with a global legend."""
        if not dataset_stats or df.empty:
            return ""
        
        # Get all datasets in display order
        datasets = dataset_manager.get_sorted_datasets()
        valid_datasets = [d for d in datasets if d in dataset_stats]
        
        if not valid_datasets:
            return ""
        
        # Create subplots
        fig, axes = plt.subplots(
            len(valid_datasets), 1,
            figsize=(12, 4 * len(valid_datasets)),
            sharex=True
        )

        fig.supylabel('Frequency', fontsize=14, x=0.03) # x=0.02, y=0.52

        if len(valid_datasets) == 1:
            axes = [axes]

        # --- Get legend handles from dummy plot ---
        temp_fig, temp_ax = plt.subplots()
        sns.histplot([0], color="green", label="Simple", ax=temp_ax)
        sns.histplot([1], color="red", label="Complex", ax=temp_ax)
        temp_ax.axvline(0, color="green", linestyle="--", linewidth=2, label="Simple Mean")
        temp_ax.axvline(1, color="red", linestyle="--", linewidth=2, label="Complex Mean")
        handles, labels = temp_ax.get_legend_handles_labels()
        plt.close(temp_fig)

        # Plot each dataset
        for ax, dataset_id in zip(axes, valid_datasets):
            dataset_info = dataset_manager.datasets.get(dataset_id, {})
            display_name = dataset_info.get("display_name", dataset_id)
            category = dataset_info.get("category", "evaluation")
            
            mask = df["dataset"] == dataset_id
            dataset_y = y[mask]
            dataset_scores = scores[mask]
            
            if len(dataset_y) < 10 or len(np.unique(dataset_y)) < 2:
                ax.text(
                    0.5, 0.5,
                    f"Insufficient data for {display_name}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=12
                )
                continue
            
            simple_scores = dataset_scores[dataset_y == 0]
            expert_scores = dataset_scores[dataset_y == 1]
            
            if len(simple_scores) == 0 or len(expert_scores) == 0:
                continue
            
            # Plot histograms
            sns.histplot(simple_scores, bins=20, alpha=0.5, color="green",
                        edgecolor="none", ax=ax)
            sns.histplot(expert_scores, bins=20, alpha=0.5, color="red",
                        edgecolor="none", ax=ax)
            
            ax.axvline(np.mean(simple_scores), color="green", linestyle="--", linewidth=2)
            ax.axvline(np.mean(expert_scores), color="red", linestyle="--", linewidth=2)
            
            stats = dataset_stats[dataset_id]
            self._add_stats_annotation(ax, stats)
            
            is_training_related = category in ["training", "training_test"]
            title_color = "blue" if is_training_related else "black"

            # Change display name for specific models (hardcoded for now because I did not have time)
            display_name = {
                "claude": "HSQA-Claude",
                "cochrane": "Cochrane",
                "plaba-paragraph": "PLABA-para",
                "plaba-sentence": "PLABA-sent"
            }.get(display_name.lower(), display_name)

            ax.set_title(display_name, color=title_color, fontsize=16)

            # legend in this order: Simple, Complex, Simple Mean, Complex Mean
            # ax.legend(
            #     handles[2:] + handles[:2],
            #     labels[2:] + labels[:2],
            #     loc="upper right",
            #     fontsize=10
            # )

            ax.set_ylabel("")
            ax.grid(True, linestyle='--', alpha=0.7)

        axes[-1].set_xlabel("Score", fontsize=14)

        # --- Add global legend ---
        fig.legend(
            handles[2:] + handles[:2],  # Simple, Complex, Simple Mean, Complex Mean
            labels[2:] + labels[:2],
            loc="upper center",
            bbox_to_anchor=(0.54, 1.04),  # center above subplots, not whole figure
            ncol=4, fontsize=12
        )

        fig.tight_layout()

        # Save and return path
        filepath = self.subdirs['datasets'] / "dataset_distributions.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return str(filepath)
    
    def plot_boxplot(self, df: pd.DataFrame, scores: np.ndarray, 
                   thresholds: Dict, dataset_manager: DatasetManager,
                   show_title: bool = True, show_legend: bool = False) -> str:
        """Plot boxplot of scores by dataset with outlier thresholds."""
        if df.empty or len(scores) == 0:
            return ""
        
        # Prepare data for plotting
        plot_data = []
        
        # Process each dataset
        for dataset in df["dataset"].unique():
            # Get dataset metadata
            info = dataset_manager.datasets.get(dataset, {})
            display_name = info.get("display_name", dataset)
            category = info.get("category", "evaluation")
            
            # Get dataset scores
            mask = df["dataset"] == dataset
            dataset_scores = scores[mask]
            
            # Store data for plotting
            for score in dataset_scores:
                plot_data.append({
                    "dataset": display_name,
                    "score": score,
                    "category": category
                })
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Order datasets by median score and category
        medians = plot_df.groupby("dataset")["score"].median()
        
        # Then sort by category and median
        ordered_datasets = []
        
        # Add training datasets first
        training_categories = ["training", "training_test"]
        training_mask = plot_df["category"].isin(training_categories)
        training_datasets = plot_df[training_mask]["dataset"].unique()
        ordered_datasets.extend(sorted(training_datasets, key=lambda x: medians.get(x, 0)))
        
        # Add evaluation datasets next
        eval_datasets = plot_df[~training_mask]["dataset"].unique()
        ordered_datasets.extend(sorted(eval_datasets, key=lambda x: medians.get(x, 0)))
        
        # Plot boxplot
        boxplot = sns.boxplot(
            x="dataset",
            y="score",
            data=plot_df,
            ax=ax,
            palette="Set2",
            order=ordered_datasets
        )
        
        # Color the x-tick labels based on dataset category
        for i, dataset in enumerate(ordered_datasets):
            # Get all categories for this dataset (may have mixed rows)
            categories = plot_df[plot_df["dataset"] == dataset]["category"].unique()
            
            # Use blue if any entry is training-related
            is_training = any(cat in training_categories for cat in categories)
            color = "blue" if is_training else "black"
            
            plt.setp(ax.get_xticklabels()[i], color=color)
        
        # Add threshold lines
        if "upper" in thresholds:
            ax.axhline(
                thresholds["upper"],
                color="red",
                linestyle="--",
                label=f"Upper Threshold: {thresholds['upper']:.2f}"
            )
        if "lower" in thresholds:
            ax.axhline(
                thresholds["lower"],
                color="red",
                linestyle="--",
                label=f"Lower Threshold: {thresholds['lower']:.2f}"
            )
        
        # Style plot
        if show_title:
            ax.set_title("Score Distribution by Dataset", fontsize=16)
        ax.set_xlabel("Dataset", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        if show_legend:
            ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Save plot
        filepath = self.subdirs['datasets'] / "boxplot.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        return str(filepath)
    
    def _add_stats_annotation(self, ax, stats: Dict) -> None:
        """Add statistical annotation to plot."""
        # Format statistical test info
        test_info = f"p: {stats.get('p_value', 0):.3e}"
        
        # Create annotation text
        info_text = (
            f"Mean Diff: {stats.get('mean_difference', 0):.3f}\n"
            f"Cohen's d: {stats.get('cohens_d', 0):.3f}\n"
            f"Overlap: {stats.get('overlap_coefficient', 0):.3f}\n"
            f"AUC: {stats.get('auc', 0):.3f}\n"
            f"JS Div: {stats.get('js_divergence', 0):.3f}\n"
            f"{test_info}"
        )
        
        # Add to plot
        ax.annotate(
            info_text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            backgroundcolor="white",
            alpha=0.8,
            va="top"
        )


class Phase4Pipeline:
    """Main pipeline for text complexity formula modeling."""
    
    def __init__(self, 
                phase1_dir: str,
                output_dir: str = None,
                training_datasets: List[str] = ["claude"],
                evaluation_datasets: List[str] = None,
                features: List[str] = None,
                features_file: str = None,
                normalization: str = "z_score",
                test_size: float = 0.2,
                model_params: Dict = None,
                penalty: str = "elasticnet",
                cv: int = 10,
                n_jobs: int = -1,
                random_state: int = 42,
                experiment_id: str = None,
                show_plot_titles: bool = True,
                feature_name_map: Dict[str, str] = None):
        """Initialize text complexity formula modeling pipeline."""
        # Directories
        self.phase1_dir = Path(phase1_dir)
        self.output_dir = Path(output_dir or self.phase1_dir.parent / "phase4")
        
        # Add experiment subdirectory if provided
        if experiment_id:
            self.output_dir = self.output_dir / "experiments" / experiment_id
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['data', 'models', 'vis']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Configuration parameters
        self.training_datasets = training_datasets
        self.evaluation_datasets = evaluation_datasets or []
        self.features = features
        self.features_file = features_file
        self.normalization = normalization
        self.test_size = test_size
        self.model_params = model_params
        self.penalty = penalty
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Visualization parameters
        self.show_plot_titles = show_plot_titles
        self.feature_name_map = feature_name_map
        
        # Initialize components
        self.dataset_manager = DatasetManager(self.phase1_dir)
        self.trainer = ModelTrainer(random_state=random_state, cv=cv, n_jobs=n_jobs)
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer(self.output_dir / "vis")
        
        # Data containers
        self.train_data = pd.DataFrame()
        self.eval_datasets = {}
        self.X_train = pd.DataFrame()
        self.y_train = np.array([])
        self.X_test = pd.DataFrame()
        self.y_test = np.array([])
        
        # Results containers
        self.pipeline = None
        self.formula = {}
        self.evaluation_results = {}
    
    def run(self) -> Dict:
        """Execute the full pipeline and return results."""
        logger.info(f"Starting formula model pipeline with penalty={self.penalty}")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Prepare features
        self.prepare_features()
        
        # Step 3: Prepare training data
        self.prepare_training_data()
        
        # Step 4: Train model
        self.train_model()
        
        # Step 5: Extract formula
        self.extract_formula()
        
        # Step 6: Evaluate model
        self.evaluate_model()
        
        # Step 7: Visualize results
        self.visualize_results()
        
        # Step 8: Save results
        summary = self.save_results()
        
        logger.info("Pipeline completed successfully")
        return summary
    
    def load_data(self) -> None:
        """Load datasets for training and evaluation."""
        logger.info("Loading datasets")
        
        # Load all datasets
        self.train_data, self.eval_datasets = self.dataset_manager.load_datasets(
            self.training_datasets,
            self.evaluation_datasets,
            self.test_size,
            self.random_state
        )
        
        if self.train_data.empty:
            raise ValueError("No valid training data found")
            
        logger.info(f"Loaded {len(self.train_data)} training samples and "
                  f"{len(self.eval_datasets)} evaluation datasets")
    
    def prepare_features(self) -> None:
        """Prepare features for model training."""
        # If features already provided, use them
        if self.features:
            logger.info(f"Using {len(self.features)} provided features")
            return
            
        # Load features from file if provided
        if self.features_file:
            self.features = self._load_features_from_file()
            if self.features:
                logger.info(f"Loaded {len(self.features)} features from file")
                return
        
        # Otherwise, determine all available features
        self._determine_all_features()
    
    def _load_features_from_file(self) -> List[str]:
        """Load features from a file (JSON or CSV)."""
        file_path = Path(self.features_file)
        
        if not file_path.exists():
            logger.warning(f"Features file not found: {file_path}")
            return []
        
        # Handle based on file type
        suffix = file_path.suffix.lower()
        
        # JSON file
        if suffix == '.json':
            data = load_json(file_path)
            
            # Try different possible structures
            if isinstance(data, list):
                return data
            
            if isinstance(data, dict):
                if 'features' in data:
                    return data['features']
                if 'selected_features' in data:
                    return data['selected_features']
        
        # CSV file
        elif suffix == '.csv':
            df = pd.read_csv(file_path)
            
            # Try different column names
            if 'feature' in df.columns:
                return df['feature'].tolist()
            
            if not df.empty:
                return df.iloc[:, 0].tolist()
        
        logger.warning(f"Could not extract features from {file_path}")
        return []
    
    def _determine_all_features(self) -> None:
        """Determine all available numeric features."""
        if self.train_data.empty:
            self.features = []
            return
            
        # Get all numeric columns
        numeric_cols = self.train_data.select_dtypes(include=['number']).columns.tolist()
        
        # Remove excluded columns
        self.features = [col for col in numeric_cols if col not in EXCLUDED_COLUMNS]
        
        logger.info(f"Using all {len(self.features)} available numeric features")
    
    def prepare_training_data(self) -> None:
        """Prepare training data for model fitting."""
        if self.train_data.empty or not self.features:
            raise ValueError("No valid training data or features")
        
        # Create target variable
        y = (self.train_data["text_type"] == "expert").astype(int)
        
        # Validate features
        valid_features = [f for f in self.features if f in self.train_data.columns]
        if not valid_features:
            raise ValueError("No valid features found in training data")
        
        # Select feature columns
        X = self.train_data[valid_features]
        
        # Use all data for training if no evaluation needed
        if not self.evaluation_datasets:
            self.X_train = X
            self.y_train = y
            self.features = valid_features
            return
        
        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Create proper test split with original dataset name
        original_dataset = self.training_datasets[0]  # Use first dataset if multiple
        test_dataset_name = f"{original_dataset}_test"
        
        # Add test data to evaluation datasets
        test_df = self.train_data.iloc[self.X_test.index].copy()
        test_df["dataset"] = test_dataset_name
        test_df["original_dataset"] = original_dataset  # Store original name
        
        # Register dataset - critical for sorting and coloring
        self.dataset_manager.register_dataset(
            test_dataset_name, 
            category="training_test",  
            parent=original_dataset  
        )
        
        # Add to evaluation datasets
        self.eval_datasets[test_dataset_name] = (test_df, self.y_test)
        
        logger.info(f"Prepared training data: {len(self.X_train)} train, {len(self.X_test)} test")
        
        # Update features to valid ones only
        self.features = valid_features
    
    def train_model(self) -> None:
        """Train the model using logistic regression."""
        self.pipeline = self.trainer.train(
            self.X_train,
            self.y_train,
            self.features,
            self.normalization,
            self.penalty,
            self.model_params
        )
        
        logger.info("Model training completed")
    
    def extract_formula(self) -> None:
        """Extract formula from trained model."""
        self.formula = self.trainer.extract_formula(
            self.features,
            self.training_datasets,
            self.normalization
        )
        
        logger.info(f"Extracted formula with {len(self.formula.get('coefficients', {}))} non-zero coefficients")
    
    def evaluate_model(self) -> None:
        """Evaluate model on evaluation datasets."""
        if not self.eval_datasets:
            logger.info("Skipping evaluation: no evaluation datasets provided")
            self.evaluation_results = {}
            return
        
        self.evaluation_results = self.evaluator.evaluate(
            self.pipeline,
            self.features,
            self.eval_datasets
        )
        
        logger.info("Model evaluation completed")
    
    def visualize_results(self) -> None:
        """Create visualizations for model results."""
        if not self.evaluation_results:
            logger.info("Skipping visualization: no evaluation results available")
            return
        
        viz_paths = self.visualizer.create_all_visualizations(
            self.formula,
            self.evaluation_results,
            self.dataset_manager,
            show_title=self.show_plot_titles,
            feature_names=self.feature_name_map
        )
        
        # Store visualization paths in evaluation results
        if viz_paths:
            self.evaluation_results["visualization_paths"] = viz_paths
    
    def save_results(self) -> Dict:
        """Save model, formula, and evaluation results."""
        # Create directories
        models_dir = self.output_dir / "models"
        data_dir = self.output_dir / "data"
        
        # Save model pipeline
        if self.pipeline:
            model_path = models_dir / "complexity_pipeline.joblib"
            joblib.dump(self.pipeline, model_path)
            logger.info(f"Saved model to {model_path}")
        
        # Save features
        if self.features:
            features_path = models_dir / "model_features.json"
            save_json({"features": self.features}, features_path)
            logger.info(f"Saved features to {features_path}")
        
        # Save formula
        if self.formula:
            formula_path = self.output_dir / "formula.json"
            save_json(self.formula, formula_path)
            logger.info(f"Saved formula to {formula_path}")
            
            # Save coefficients as CSV for easier analysis
            if "coefficients" in self.formula:
                coeffs = {k: v for k, v in self.formula["coefficients"].items() if abs(v) > 1e-5}
                coef_df = pd.DataFrame({
                    "feature": list(coeffs.keys()),
                    "coefficient": list(coeffs.values()),
                    "importance": [abs(v) for v in coeffs.values()]
                }).sort_values("importance", ascending=False)
                
                coef_path = data_dir / "formula_coefficients.csv"
                coef_df.to_csv(coef_path, index=False)
                logger.info(f"Saved coefficients to {coef_path}")
        
        # Save evaluation results
        if self.evaluation_results:
            # Save plot data separately for later replotting
            plot_data = {
                "combined_df": self.evaluation_results.get("combined_df"),
                "combined_y": self.evaluation_results.get("combined_y"),
                "combined_scores": self.evaluation_results.get("combined_scores"),
                "individual_metrics": self.evaluation_results.get("individual_metrics", {}),
                "dataset_stats": self.evaluation_results.get("dataset_stats", {}),
                "overall_stats": self.evaluation_results.get("overall_stats", {}),
                "outlier_stats": self.evaluation_results.get("outlier_stats", {})
            }
            
            plot_data_path = data_dir / "plot_data.joblib"
            joblib.dump(plot_data, plot_data_path, compress=3)
            logger.info(f"Saved plot data to {plot_data_path}")
            
            # Save serializable results (without large dataframes)
            serializable_results = {
                k: v for k, v in self.evaluation_results.items() 
                if k not in ["combined_df", "combined_y", "combined_scores"]
            }
            
            results_path = data_dir / "evaluation_results.json"
            save_json(serializable_results, results_path)
            logger.info(f"Saved evaluation results to {results_path}")
        
        # Create summary
        summary = {
            "formula_info": {
                "num_features": len(self.features),
                "num_non_zero": len(self.formula.get("coefficients", {})),
                "normalization": self.normalization,
                "trained_on": self.training_datasets,
                "penalty": self.penalty
            },
            "performance": self.evaluation_results.get("overall_stats", {}),
            "dataset_performance": self.evaluation_results.get("dataset_stats", {}),
            "visualizations": self.evaluation_results.get("visualization_paths", {}),
            "files": {
                "formula": "formula.json",
                "model": "models/complexity_pipeline.joblib",
                "features": "models/model_features.json",
                "results": "data/evaluation_results.json",
                "plot_data": "data/plot_data.joblib"
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        save_json(summary, summary_path)
        logger.info(f"Saved summary to {summary_path}")
        
        return summary


def compare_penalties(results: Dict[str, Dict]) -> Dict:
    """Compare results from different penalty types."""
    if not results:
        return {}
    
    # Metrics to compare
    comparison = {
        "auc": {},
        "cohens_d": {},
        "mean_difference": {},
        "overlap_coefficient": {},
        "js_divergence": {},
        "num_features": {},
        "non_zero_coeffs": {}
    }
    
    # Extract metrics for each penalty
    for penalty, result in results.items():
        formula_info = result.get("formula_info", {})
        performance = result.get("performance", {})
        
        comparison["auc"][penalty] = performance.get("auc", 0)
        comparison["cohens_d"][penalty] = performance.get("cohens_d", 0)
        comparison["mean_difference"][penalty] = performance.get("mean_difference", 0)
        comparison["overlap_coefficient"][penalty] = performance.get("overlap_coefficient", 0)
        comparison["js_divergence"][penalty] = performance.get("js_divergence", 0)
        comparison["num_features"][penalty] = formula_info.get("num_features", 0)
        comparison["non_zero_coeffs"][penalty] = formula_info.get("num_non_zero", 0)
    
    # Find best penalties for each metric
    best_penalties = {}
    
    # For metrics where higher is better
    for metric in ["auc", "cohens_d", "mean_difference", "js_divergence"]:
        if comparison[metric]:
            best_penalties[metric] = max(comparison[metric].items(), key=lambda x: x[1])[0]
    
    # For metrics where lower is better
    for metric in ["overlap_coefficient"]:
        if comparison[metric]:
            best_penalties[metric] = min(comparison[metric].items(), key=lambda x: x[1])[0]
    
    # Create a balanced score (performance vs. sparsity)
    if comparison["auc"] and comparison["non_zero_coeffs"]:
        # Normalize AUC scores
        auc_values = list(comparison["auc"].values())
        min_auc = min(auc_values)
        max_auc = max(auc_values)
        auc_range = max_auc - min_auc if max_auc > min_auc else 1.0
        
        # Normalize non-zero coefficients (lower is better)
        non_zero_values = list(comparison["non_zero_coeffs"].values())
        max_non_zero = max(non_zero_values)
        
        # Calculate balanced score (70% performance, 30% sparsity)
        balance_scores = {}
        for penalty in comparison["auc"]:
            normalized_auc = (comparison["auc"][penalty] - min_auc) / auc_range
            normalized_sparsity = 1.0 - (comparison["non_zero_coeffs"][penalty] / max_non_zero)
            balance_scores[penalty] = 0.7 * normalized_auc + 0.3 * normalized_sparsity
        
        best_penalties["balance"] = max(balance_scores.items(), key=lambda x: x[1])[0]
    
    # Get recommended penalty
    recommended = best_penalties.get("balance", best_penalties.get("auc", "elasticnet"))
    
    return {
        "metrics": comparison,
        "best_penalties": best_penalties,
        "recommended": recommended
    }


def print_formula_summary(formula: Dict, results: Dict) -> None:
    """Print formatted summary of formula and results."""
    if not formula:
        print("\nNo formula was produced.")
        return
    
    print("\n=== Formula Summary ===")
    
    # Print equation
    print(f"Complexity Score = {formula.get('intercept', 0):.6f}")
    
    # Get non-zero coefficients
    coeffs = formula.get("coefficients", {})
    sorted_features = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Print top coefficients (max 20)
    top_features = sorted_features[:min(20, len(sorted_features))]
    for metric, weight in top_features:
        op = "+" if weight >= 0 else "-"
        print(f"  {op} {abs(weight):.6f} × {metric}")
    
    # Print basic info
    print(f"\nFeatures Used: {len(formula.get('features', []))}")
    print(f"Non-zero Features: {len(coeffs)}")
    print(f"Normalization: {formula.get('normalization')}")
    print(f"Trained on Datasets: {', '.join(formula.get('trained_on', []))}")
    
    # Print hyperparameters
    hyperparams = formula.get("hyperparameters", {})
    model_params = hyperparams.get("model_params", {})
    if model_params:
        print(f"Model Parameters: C={model_params.get('C')}, "
              f"l1_ratio={model_params.get('l1_ratio')}, "
              f"penalty={model_params.get('penalty')}")
    
    # Print performance
    overall_stats = results.get("overall_stats", {})
    if overall_stats:
        print("\nOverall Performance:")
        print(f"  AUC: {overall_stats.get('auc', 0):.4f}")
        print(f"  Cohen's d: {overall_stats.get('cohens_d', 0):.4f}")
        print(f"  Mean Difference: {overall_stats.get('mean_difference', 0):.4f}")
        print(f"  Overlap Coefficient: {overall_stats.get('overlap_coefficient', 0):.4f}")
        print(f"  JS Divergence: {overall_stats.get('js_divergence', 0):.4f}")
        
        # Print statistical test
        test_used = overall_stats.get('test_used', '')
        if test_used:
            p_value = overall_stats.get('p_value', 0)
            print(f"  Statistical Test: {test_used}, p-value: {p_value:.4e}")
        
        # Print sample counts
        n_simple = overall_stats.get('n_simple', 0)
        n_expert = overall_stats.get('n_expert', 0)
        print(f"  Samples: {n_simple + n_expert} total")
        print(f"           {n_simple} simple, {n_expert} expert")
    
    # Print dataset performance
    dataset_stats = results.get("dataset_stats", {})
    if dataset_stats:
        print("\nDataset-Specific Performance:")
        for dataset, stats in dataset_stats.items():
            print(f"  {dataset}:")
            print(f"    AUC: {stats.get('auc', 0):.4f}, Cohen's d: {stats.get('cohens_d', 0):.4f}")
            print(f"    Mean Diff: {stats.get('mean_difference', 0):.4f}, Overlap: {stats.get('overlap_coefficient', 0):.4f}")
            print(f"    Samples: {stats.get('n_simple', 0)} simple, {stats.get('n_expert', 0)} expert")


def print_comparison_summary(comparison: Dict) -> None:
    """Print formatted comparison of different penalty types."""
    if not comparison:
        print("\nNo comparison data available.")
        return
    
    print("\n=== Penalty Comparison Summary ===")
    
    # Get metrics and penalties
    metrics = comparison.get("metrics", {})
    if not metrics:
        print("No metrics data available.")
        return
    
    # Get all penalties
    penalties = set()
    for metric_dict in metrics.values():
        penalties.update(metric_dict.keys())
    
    # Print table header
    headers = ["Penalty", "AUC", "Cohen's d", "Mean Diff", "Overlap", "JS Div", "Features", "Non-Zero"]
    print(f"{headers[0]:<10}", end="")
    for h in headers[1:]:
        print(f"{h:>12}", end="")
    print()
    print("-" * 95)
    
    # Print metrics for each penalty
    for penalty in sorted(penalties):
        print(f"{penalty:<10}", end="")
        
        print(f"{metrics.get('auc', {}).get(penalty, 0):.4f}".rjust(12), end="")
        print(f"{metrics.get('cohens_d', {}).get(penalty, 0):.4f}".rjust(12), end="")
        print(f"{metrics.get('mean_difference', {}).get(penalty, 0):.4f}".rjust(12), end="")
        print(f"{metrics.get('overlap_coefficient', {}).get(penalty, 0):.4f}".rjust(12), end="")
        print(f"{metrics.get('js_divergence', {}).get(penalty, 0):.4f}".rjust(12), end="")
        print(f"{metrics.get('num_features', {}).get(penalty, 0)}".rjust(12), end="")
        print(f"{metrics.get('non_zero_coeffs', {}).get(penalty, 0)}".rjust(12))
    
    print("-" * 95)
    
    # Print best penalties
    best_penalties = comparison.get("best_penalties", {})
    if best_penalties:
        print("\nBest Penalty by Metric:")
        for metric, penalty in best_penalties.items():
            if metric != "balance":
                print(f"  {metric:<20}: {penalty}")
    
    # Print recommendation
    recommended = comparison.get("recommended")
    if recommended:
        print(f"\nRecommended Penalty: {recommended} (best balance of performance and sparsity)")


def main() -> None:
    """Main function for running the formula modeling pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Formula-based text complexity modeling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--phase1-dir", required=True, type=str,
                      help="Directory with Phase 1 results")
    
    # Feature options (mutually exclusive)
    feature_group = parser.add_mutually_exclusive_group(required=False)
    feature_group.add_argument("--features", type=str, nargs="+",
                             help="List of feature names to use")
    feature_group.add_argument("--features-file", type=str,
                             help="Path to JSON or CSV file with features")
    
    # Optional arguments
    parser.add_argument("--output-dir", type=str,
                      help="Directory for outputs")
    parser.add_argument("--training-datasets", type=str, nargs="+", default=["claude"],
                      help="List of training dataset names")
    parser.add_argument("--evaluation-datasets", type=str, nargs="+",
                      help="List of evaluation dataset names")
    parser.add_argument("--normalization", type=str, default="z_score",
                      choices=["z_score", "min_max", "robust", "none"],
                      help="Normalization method")
    parser.add_argument("--test-size", type=float, default=0.2,
                      help="Proportion of data for testing")
    parser.add_argument("--model-c", type=float,
                      help="C parameter for LogisticRegression")
    parser.add_argument("--model-l1-ratio", type=float,
                      help="l1_ratio parameter for LogisticRegression")
    parser.add_argument("--penalty", type=str, nargs="+",
                      default=["elasticnet", "l1", "l2", "none"],
                      choices=["elasticnet", "l1", "l2", "none"],
                      help="Penalty type(s) for LogisticRegression")
    parser.add_argument("--cv", type=int, default=10,
                      help="Number of cross-validation folds")
    parser.add_argument("--n-jobs", type=int, default=-1,
                      help="Number of parallel jobs")
    parser.add_argument("--random-state", type=int, default=42,
                      help="Random state for reproducibility")
    # New visualization arguments
    parser.add_argument("--no-plot-titles", action="store_true",
                      help="Hide titles on plots")
    parser.add_argument("--feature-name-map", type=str,
                      help="Path to JSON file with feature name mappings")
    
    args = parser.parse_args()
    
    # Set model parameters if provided
    model_params = None
    if args.model_c is not None:
        model_params = {'C': args.model_c}
        if args.model_l1_ratio is not None:
            model_params['l1_ratio'] = args.model_l1_ratio
    
    # Load feature name map if provided
    feature_name_map = None
    if args.feature_name_map:
        feature_name_map = load_json(args.feature_name_map)
        logger.info(f"Loaded feature name mappings for {len(feature_name_map)} features")

    # Run for each penalty
    all_results = {}
    
    for penalty in args.penalty:
        print(f"\n=== Running formula pipeline with penalty: {penalty} ===")
        
        # Create experiment ID based on penalty
        experiment_id = f"penalty_{penalty}"
        
        # Run pipeline
        pipeline = Phase4Pipeline(
            phase1_dir=args.phase1_dir,
            output_dir=args.output_dir,
            training_datasets=args.training_datasets,
            evaluation_datasets=args.evaluation_datasets,
            features=args.features,
            features_file=args.features_file,
            normalization=args.normalization,
            test_size=args.test_size,
            model_params=model_params,
            penalty=penalty,
            cv=args.cv,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            experiment_id=experiment_id,
            show_plot_titles=not args.no_plot_titles,
            feature_name_map=feature_name_map
        )
        
        # Run pipeline
        results = pipeline.run()
        all_results[penalty] = results
        
        # Print summary
        print_formula_summary(pipeline.formula, pipeline.evaluation_results)


if __name__ == "__main__":
    main()