import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import loguniform
from tqdm import tqdm
import plotly.express as px

from utils.helpers import save_json, setup_logging
from rq1.formula.impl.stats import statistical_test

# Initialize logging
logger = setup_logging()

# Define columns to exclude from feature processing
EXCLUDED_COLUMNS = ['sample_id', 'id', 'text_id', 'document_id', 'text', 'dataset']

class FeatureProcessor:
    """Handles feature processing tasks like AUC calculation and correlation filtering."""

    def __init__(self, random_state: int = 42, cv: int = 10, n_jobs: int = -1):
        # Set random state, cross-validation folds, and parallel job count
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs

    def calculate_feature_aucs(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate AUC for each feature using cross-validation."""
        logger.info(f"Calculating AUC for {X.shape[1]} features")
        # Initialize dictionary to store AUC scores for each feature
        feature_aucs = {}
        # Create stratified k-fold splitter for cross-validation
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Iterate through each feature
        for feature in X.columns:
            # Select single feature for AUC calculation
            X_single = X[[feature]]
            # Define pipeline with scaling and logistic regression
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, random_state=self.random_state))
            ])
            # Compute cross-validated AUC scores
            cv_scores = cross_val_score(pipeline, X_single, y, cv=cv_splitter, scoring='roc_auc', n_jobs=self.n_jobs)
            # Store mean AUC score for the feature
            feature_aucs[feature] = np.mean(cv_scores)

        logger.info("AUC calculation complete")
        return feature_aucs

    def filter_correlated_features(self, X: pd.DataFrame, feature_aucs: Dict[str, float], threshold: float = 0.7) -> Tuple[pd.DataFrame, Dict]:
        """Filter out highly correlated features - keep highest AUC from each pair."""
        logger.info(f"Filtering correlated features with threshold {threshold}")
        
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = set()
        
        for col in upper_tri.columns:
            # Skip if current column is already marked for dropping
            if col in to_drop:
                continue
                
            # Find features correlated with this column
            correlated = upper_tri.index[upper_tri[col] > threshold]
            
            for corr_feature in correlated:
                # Skip if correlated feature is already marked for dropping
                if corr_feature in to_drop:
                    continue
                    
                # Compare AUCs of the current column and the correlated feature
                auc1 = feature_aucs.get(col, 0)
                auc2 = feature_aucs.get(corr_feature, 0)

                # If the current column has a lower AUC than the correlated feature, drop it
                if auc1 < auc2:
                    to_drop.add(col)
                    break  # No need to check more pairs with this column
                
                # If the correlated feature has a lower AUC, drop it
                to_drop.add(corr_feature)
        
        # Filter dataset
        X_filtered = X.drop(columns=list(to_drop))
        
        filter_info = {
            "threshold": threshold,
            "dropped": list(to_drop),
            "original_count": X.shape[1],
            "remaining_count": X_filtered.shape[1]
        }
        
        logger.info(f"Reduced features from {X.shape[1]} to {X_filtered.shape[1]}")
        return X_filtered, filter_info
    

class BootstrapLassoSelector:
    """Performs feature selection using Lasso with bootstrap sampling."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, lasso_C: float, bootstrap_samples: int = 1000, 
                max_features: int = None, random_state: int = 42, n_jobs: int = -1):
        # Initialize input data and parameters
        self.X = X  # Fixed typo from 'self.Task' to 'self.X'
        self.y = y
        self.lasso_C = lasso_C
        self.bootstrap_samples = bootstrap_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        # Initialize storage for results and statistics
        self.features = []
        self.importances = {}
        self.filtering_history = {}
        self.bootstrap_stats = {}
        self.after_ci_count = None
        self.after_cv_count = None

    def run_bootstrap(self):
        """Run bootstrap iterations to collect coefficient statistics."""
        logger.info(f"Running {self.bootstrap_samples} Lasso bootstrap iterations")
        # Fit scaler to training data
        scaler = StandardScaler().fit(self.X)
        # Initialize dictionary to store coefficients for each feature
        all_coeffs = {feature: [] for feature in self.X.columns}

        # Perform bootstrap iterations
        for i in tqdm(range(self.bootstrap_samples), desc="Bootstrap iterations"):
            # Set random seed for reproducibility
            np.random.seed(self.random_state + i)
            # Sample indices with replacement
            indices = resample(np.arange(len(self.X)), replace=True)
            # Create bootstrap sample
            X_boot, y_boot = self.X.iloc[indices], self.y.iloc[indices]
            # Scale bootstrap features
            X_boot_scaled = scaler.transform(X_boot)

            # Train Lasso logistic regression model
            model = LogisticRegression(penalty='l1', solver="liblinear", C=self.lasso_C, max_iter=10000, random_state=self.random_state + i)
            model.fit(X_boot_scaled, y_boot)

            # Collect coefficients for each feature
            for j, feature in enumerate(self.X.columns):
                all_coeffs[feature].append(model.coef_[0][j])

        # Calculate statistics from bootstrap coefficients
        self.bootstrap_stats = self._calculate_stats(all_coeffs)

    def _calculate_stats(self, all_coeffs: Dict[str, List[float]]) -> Dict:
        """Calculate statistics from bootstrap coefficients."""
        stats = {}
        # Iterate through features and their coefficients
        for feature, coeffs in all_coeffs.items():
            coeffs_array = np.array(coeffs)
            # Calculate mean and standard error
            mean = np.mean(coeffs_array)
            std_error = np.std(coeffs_array, ddof=1)
            # Calculate 95% confidence interval
            lower_ci = np.percentile(coeffs_array, 2.5)
            upper_ci = np.percentile(coeffs_array, 97.5)
            # Calculate coefficient of variation
            cv = std_error / abs(mean) if abs(mean) > 1e-10 else float('inf')
            # Store statistics for the feature
            stats[feature] = type('Stats', (), {
                'mean': mean, 'lower_ci': lower_ci, 'upper_ci': upper_ci, 'cv': cv, 'ci_contains_zero': (lower_ci <= 0 <= upper_ci)
            })()
        return stats

    def select_features(self) -> Tuple[List[str], Dict[str, float]]:
        """Select features based on bootstrap statistics and filtering criteria."""
        # Run bootstrap if stats are not yet computed
        if not self.bootstrap_stats:
            self.run_bootstrap()

        # Filter features where CI does not include zero
        remaining = [f for f, stats in self.bootstrap_stats.items() if not stats.ci_contains_zero]
        self.after_ci_count = len(remaining)
        # Store filtered features due to CI
        filtered_by_ci = {f: {'lower_ci': stats.lower_ci, 'upper_ci': stats.upper_ci, 'mean': stats.mean} 
                          for f, stats in self.bootstrap_stats.items() if stats.ci_contains_zero}

        # Get CV values for remaining features
        cv_values = [self.bootstrap_stats[f].cv for f in remaining if self.bootstrap_stats[f].cv < float('inf')]
        # Handle case where no valid CV values exist
        if not cv_values:
            self.features = []
            self.importances = {}
            return self.features, self.importances

        # Calculate IQR-based upper fence for CV filtering
        q1, q3 = np.percentile(cv_values, [25, 75])
        upper_fence = q3 + 1.5 * (q3 - q1)
        # Filter features by CV threshold
        selected = [f for f in remaining if self.bootstrap_stats[f].cv <= upper_fence]
        self.after_cv_count = len(selected)
        # Store filtered features due to CV
        filtered_by_cv = {f: self.bootstrap_stats[f].cv for f in remaining if self.bootstrap_stats[f].cv > upper_fence}

        # Calculate feature importances based on absolute mean coefficients
        importances = {f: abs(self.bootstrap_stats[f].mean) for f in selected}
        
        # Sort features by importance
        sorted_features = sorted(selected, key=lambda x: importances[x], reverse=True)

        # Limit to maximum number of features if specified
        final_features = sorted_features[:self.max_features] if self.max_features else sorted_features
        
        # Update instance variables
        self.features = final_features
        self.importances = {f: importances[f] for f in final_features}
        self.filtering_history = {
            'ci': {'alpha': 0.05, 'filtered': filtered_by_ci, 'count': len(filtered_by_ci)},
            'cv': {'threshold': upper_fence, 'filtered': filtered_by_cv, 'count': len(filtered_by_cv)}
        }

        logger.info(f"Selected {len(self.features)} features after filtering")
        return self.features, self.importances

class Phase3Pipeline:
    """Orchestrates the feature selection pipeline."""

    def __init__(self, phase1_dir: str, output_dir: str, training_datasets: List[str], 
                 bootstrap_samples: int = 1000, max_features: int = None,
                 correlation_threshold: float = 0.7, random_state: int = 42, 
                 cv: int = 10, n_jobs: int = -1):
        # Initialize directories and parameters
        self.phase1_dir = Path(phase1_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for data, filtering, and visualizations
        self.subdirs = {
            'data': self.output_dir / 'data',
            'visualizations': self.output_dir / 'vis'
        }
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # Store configuration parameters
        self.training_datasets = training_datasets
        self.bootstrap_samples = bootstrap_samples
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs

        # Initialize storage for data and results
        self.X_train = None
        self.y_train = None
        self.feature_aucs = {}
        self.correlation_info = {}
        self.selected_features = []
        self.feature_importances = {}
        self.filtering_history = {}
        self.initial_feature_count = None
        self.after_correlation_count = None
        self.selector = None

    def run(self):
        """Execute the full feature selection pipeline."""
        # Step 1: Load data
        self.load_data()

        # Step 2: Filter correlated features
        self.filter_correlated_features()

        # Step 3: Run feature selection with bootstrap Lasso
        self.run_feature_selection()

        # Step 4: Save results
        self.save_results()

        # Step 5: Generate funnel chart
        self.plot_funnel_chart()

    def load_data(self):
        """Load and prepare training data from specified datasets."""
        logger.info("Loading training datasets")
        # Load CSV files from training datasets
        train_dfs = [pd.read_csv(self.phase1_dir / dataset / "metrics_df.csv") 
                     for dataset in self.training_datasets 
                     if (self.phase1_dir / dataset / "metrics_df.csv").exists() 
                     and "text_type" in pd.read_csv(self.phase1_dir / dataset / "metrics_df.csv").columns]

        # Check if any valid datasets were found
        if not train_dfs:
            raise ValueError("No valid training datasets found")

        # Concatenate datasets
        self.df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"Loaded {len(self.df)} samples")

        # Create binary labels (1 for expert, 0 for non-expert)
        self.y_train = (self.df["text_type"] == "expert").astype(int)
        # Select numeric columns, excluding non-feature columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in EXCLUDED_COLUMNS]
        self.X_train = self.df[feature_cols]
        # Store initial feature count
        self.initial_feature_count = len(feature_cols)

    def filter_correlated_features(self):
        """Filter out correlated features from the training data."""
        # Initialize feature processor
        processor = FeatureProcessor(random_state=self.random_state, cv=self.cv, n_jobs=self.n_jobs)
        # Calculate AUC for each feature
        self.feature_aucs = processor.calculate_feature_aucs(self.X_train, self.y_train)
        # Filter correlated features
        self.X_train, self.correlation_info = processor.filter_correlated_features(self.X_train, self.feature_aucs, self.correlation_threshold)
        # Store feature count after correlation filtering
        self.after_correlation_count = len(self.X_train.columns)

    def run_feature_selection(self):
        """Run the bootstrap Lasso feature selection process."""
        logger.info("Running feature selection with Lasso")
        # Define Lasso pipeline with scaling
        lasso = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty='l1', solver="liblinear", max_iter=10000, random_state=self.random_state))
        ])
        # Define hyperparameter grid for Lasso C
        param_grid = {'model__C': loguniform(0.001, 1.0)}
        # Perform randomized search to find best C
        search = RandomizedSearchCV(lasso, param_grid, n_iter=30, cv=self.cv, scoring='roc_auc', n_jobs=self.n_jobs, random_state=self.random_state)
        search.fit(self.X_train, self.y_train)
        best_C = search.best_params_['model__C']

        # Initialize and run bootstrap Lasso selector
        self.selector = BootstrapLassoSelector(self.X_train, self.y_train, lasso_C=best_C, 
                                               bootstrap_samples=self.bootstrap_samples, 
                                               max_features=self.max_features,
                                               random_state=self.random_state, n_jobs=self.n_jobs)
        self.selector.select_features()
        
        # Store selected features and their importances
        self.selected_features = self.selector.features
        self.feature_importances = self.selector.importances
        self.filtering_history = self.selector.filtering_history

    def save_results(self):
        """Save the selected features and filtering history to files."""
        logger.info("Saving results")
        # Create DataFrame for selected features and their importances
        features_df = pd.DataFrame({
            "feature": self.selected_features,
            "importance": [self.feature_importances.get(f, 0) for f in self.selected_features]
        })
        # Save features to CSV
        features_df.to_csv(self.subdirs["data"] / "selected_features.csv", index=False)

        # Combine correlation and bootstrap filtering info
        filtering_history = {"correlation": self.correlation_info, "bootstrap": self.filtering_history}
        # Save filtering history to JSON
        save_json(filtering_history, self.subdirs["data"] / "filtering_history.json")

    def plot_funnel_chart(self):
        """Generate and save a funnel chart visualizing the feature selection process."""
        # Define stages for funnel chart
        stages = [
            "Initial Features",
            "After Correlation Filtering",
            "After CI Filtering",
            "After CV Filtering",
            "Final Selected Features"
        ]
        # Collect feature counts at each stage
        counts = [
            self.initial_feature_count,
            self.after_correlation_count,
            self.selector.after_ci_count,
            self.selector.after_cv_count,
            len(self.selected_features)
        ]

        # Skip if any counts are missing
        if any(count is None for count in counts):
            logger.warning("Some feature counts are missing, skipping funnel chart")
            return

        # Generate and save funnel chart
        logger.info("Generating funnel chart")
        fig = px.funnel(x=counts, y=stages)
        fig.write_html(self.subdirs['visualizations'] / 'funnel_chart.html')
        logger.info("Funnel chart saved to visualizations/funnel_chart.html")

def main():
    """Parse arguments and run the Phase 3 pipeline."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Simplified Phase 3 Feature Selection Pipeline", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phase1-dir", type=str, required=True, help="Path to Phase 1 output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--training-datasets", type=str, nargs="+", default=["claude"], help="List of training dataset names")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--max-features", type=int, default=None, help="Maximum number of features to select")
    parser.add_argument("--correlation-threshold", type=float, default=0.7, help="Correlation threshold")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")

    # Parse command-line arguments
    args = parser.parse_args()

    # Initialize and run the pipeline
    pipeline = Phase3Pipeline(
        phase1_dir=args.phase1_dir,
        output_dir=args.output_dir,
        training_datasets=args.training_datasets,
        bootstrap_samples=args.bootstrap_samples,
        max_features=args.max_features,
        correlation_threshold=args.correlation_threshold,
        random_state=args.random_state,
        cv=args.cv,
        n_jobs=args.n_jobs
    )
    pipeline.run()

if __name__ == "__main__":
    main()