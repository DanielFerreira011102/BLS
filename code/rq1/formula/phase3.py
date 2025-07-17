import argparse
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from scipy.stats import uniform, loguniform
from scipy.stats import t as ttest_ind
from tqdm import tqdm

# Import from utils and stats modules
from utils.helpers import save_json, load_json, setup_logging
from rq1.formula.impl.stats import (
    cohens_d, overlap_coefficient, kl_divergence, js_divergence,
    mutual_information, statistical_test,
)

# Initialize logging
logger = setup_logging()

# Constants
EXCLUDED_COLUMNS = ['sample_id', 'id', 'text_id', 'document_id', 'text', 'dataset']
OUTPUT_SUBDIRS = ['data', 'vis', 'filtering', 'models']
METRICS = ['auc', 'cohens_d', 'js_divergence', 'overlap_coefficient', 'mutual_information', 'mean_difference']
ORDERING_TYPES = ['coef', 'auc']
REGULARIZATION_TYPES = ['lasso', 'elasticnet']
BOOTSTRAP_VARIANTS = ['upper_fence', 'q1']
SELECTION_METHODS = ['ci_005', 'ci_000']

# Get the appropriate scaler class based on normalization type
def get_scaler_class(normalization: str):
    """Get scaler class based on normalization type."""
    scalers = {
        "z_score": StandardScaler,
        "min_max": MinMaxScaler,
        "robust": RobustScaler,
        "none": None
    }
    return scalers.get(normalization, StandardScaler)


class FeatureStats:
    """Statistics for a feature from bootstrap analysis."""
    
    def __init__(self, mean: float, std_error: float, lower_ci: float, upper_ci: float,
                ci_contains_zero: bool, cv: float, p_value: float, t_stat: float,
                all_coeffs: np.ndarray):
        """Initialize with statistics."""
        self.mean = mean
        self.std_error = std_error
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci
        self.ci_contains_zero = ci_contains_zero
        self.cv = cv
        self.p_value = p_value
        self.t_stat = t_stat
        self.all_coeffs = all_coeffs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mean': self.mean,
            'std_error': self.std_error,
            'lower_ci': self.lower_ci,
            'upper_ci': self.upper_ci,
            'ci_contains_zero': self.ci_contains_zero,
            'cv': self.cv,
            'p_value': self.p_value,
            't_stat': self.t_stat,
            'all_coeffs': self.all_coeffs.tolist() if isinstance(self.all_coeffs, np.ndarray) else self.all_coeffs
        }


class FeatureProcessor:
    """Processes features for selection and evaluation."""
    
    def __init__(self, random_state=42, cv=10, n_jobs=-1):
        """Initialize processor with parameters."""
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs
        
    def calculate_feature_aucs(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate AUC for each feature individually."""
        logger.info(f"Calculating AUC for {X.shape[1]} features")
        
        feature_aucs = {}
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        for feature in X.columns:
            X_single = X[[feature]]
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty=None,
                    solver='lbfgs',
                    max_iter=10000,
                    random_state=self.random_state
                ))
            ])
            
            cv_scores = cross_val_score(
                pipeline, X_single, y,
                cv=cv_splitter,
                scoring='roc_auc',
                n_jobs=self.n_jobs
            )
            
            feature_aucs[feature] = np.mean(cv_scores)
        
        logger.info(f"AUC calculation complete")
        return feature_aucs
    
    def filter_correlated_features(self, X: pd.DataFrame, feature_aucs: Dict[str, float], 
                                 threshold: float = 0.7) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Filter highly correlated features."""
        logger.info(f"Filtering correlated features (threshold: {threshold})")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle only
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = set()
        filtered_pairs = []
        
        for col in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[col] > threshold]
            
            for corr_feature in correlated_features:
                corr_value = upper_tri.loc[corr_feature, col]
                filtered_pairs.append({
                    "col1": col,
                    "col2": corr_feature,
                    "corr": corr_value
                })
                
                # Keep feature with higher AUC
                if feature_aucs.get(col, 0) >= feature_aucs.get(corr_feature, 0):
                    to_drop.add(corr_feature)
                else:
                    to_drop.add(col)
        
        # Create filtered dataset
        drop_list = list(to_drop)
        X_filtered = X.drop(columns=drop_list)
        
        # Create info dictionary
        filter_info = {
            "threshold": threshold,
            "pairs": filtered_pairs,
            "dropped": drop_list,
            "count": len(drop_list),
            "original_count": X.shape[1],
            "remaining_count": X_filtered.shape[1]
        }
        
        logger.info(f"Dropped {len(drop_list)} correlated features")
        logger.info(f"Features reduced from {X.shape[1]} to {X_filtered.shape[1]}")
        
        return X_filtered, filter_info


class FeatureSelector:
    """Base class for all feature selection methods."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, name: str, 
                max_features: Optional[int] = None, 
                random_state: int = 42, cv: int = 10, 
                n_jobs: int = -1, normalization: str = "z_score"):
        """Initialize with data and parameters."""
        self.X = X
        self.y = y
        self.name = name
        self.max_features = max_features
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs
        self.normalization = normalization
        
        # Results
        self.features = []
        self.importances = {}
        self.filtering_history = {}
        self.params = {}
    
    def select_features(self) -> Tuple[List[str], Dict[str, float]]:
        """Select features using the method."""
        raise NotImplementedError("Subclasses must implement select_features")
    
    def limit_features(self, features: List[str], importances: Dict[str, float]) -> List[str]:
        """Limit the number of features if max_features is specified."""
        if not self.max_features or len(features) <= self.max_features:
            return features
        
        # Sort by importance and take top features
        sorted_features = sorted(features, key=lambda x: importances[x], reverse=True)
        return sorted_features[:self.max_features]
    
    def get_display_name(self) -> str:
        """Get a display name for the method."""
        return self.name.upper()
    
    def get_path(self) -> Path:
        """Get a path for storing results."""
        return Path(self.name)
    
    def get_color(self) -> str:
        """Get a color for plotting."""
        return "gray"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "features": self.features,
            "importances": self.importances,
            "filtering_history": self.filtering_history,
            "params": self.params
        }


class LassoSelector(FeatureSelector):
    """Feature selection using LASSO regression."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, threshold: float = 1e-3, **kwargs):
        """Initialize LASSO method."""
        super().__init__(X, y, name="lasso", **kwargs)
        self.threshold = threshold
    
    def find_best_parameters(self) -> Dict[str, float]:
        """Find optimal LASSO parameters."""
        logger.info("Finding optimal LASSO parameters")
        
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scaler_cls = get_scaler_class(self.normalization)
        
        pipeline = Pipeline([
            ("scaler", scaler_cls()),
            ("model", LogisticRegression(
                penalty='l1',
                solver="liblinear",
                max_iter=10000,
                random_state=self.random_state
            ))
        ])
        
        param_grid = {'model__C': loguniform(0.001, 1.0)}
        
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=30, cv=cv_splitter,
            scoring="roc_auc", n_jobs=self.n_jobs, random_state=self.random_state
        )
        
        search.fit(self.X, self.y)
        
        self.params = {'C': search.best_params_['model__C']}
        logger.info(f"LASSO best C: {self.params['C']:.6f}, score: {search.best_score_:.4f}")
        
        return self.params
    
    def select_features(self) -> Tuple[List[str], Dict[str, float]]:
        """Select features using LASSO."""
        logger.info("Running LASSO feature selection")
        
        # Find optimal parameters if not already done
        if not self.params:
            self.find_best_parameters()
        
        # Create and fit model
        scaler_cls = get_scaler_class(self.normalization)
        pipeline = Pipeline([
            ("scaler", scaler_cls()),
            ("model", LogisticRegression(
                penalty='l1',
                solver="liblinear",
                C=self.params['C'],
                max_iter=10000,
                random_state=self.random_state
            ))
        ])
        
        pipeline.fit(self.X, self.y)
        
        # Extract coefficients
        model = pipeline.named_steps['model']
        importances = {col: abs(model.coef_[0][i]) for i, col in enumerate(self.X.columns)}
        
        # Apply threshold
        selected = [col for col, coef in importances.items() if coef >= self.threshold]
        
        # Track filtered features
        filtered = {col: coef for col, coef in importances.items() if coef < self.threshold}
        self.filtering_history['threshold'] = {
            'value': self.threshold,
            'filtered': filtered,
            'count': len(filtered)
        }
        
        # Apply feature limit if needed
        original_count = len(selected)
        selected = self.limit_features(selected, importances)
        
        # Track limited features if any were removed
        if len(selected) < original_count:
            limited = {f: importances[f] for f in set(self.X.columns) - set(selected)}
            self.filtering_history['max_features'] = {
                'limit': self.max_features,
                'filtered': limited,
                'count': len(limited)
            }
        
        # Sort by importance
        self.features = sorted(selected, key=lambda x: importances[x], reverse=True)
        self.importances = importances
        
        logger.info(f"LASSO selected {len(self.features)} features")
        return self.features, self.importances
    
    def get_color(self) -> str:
        """Get color for LASSO."""
        return "#1f77b4"  # Blue


class ElasticNetSelector(FeatureSelector):
    """Feature selection using ElasticNet regression."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, threshold: float = 1e-3, **kwargs):
        """Initialize ElasticNet method."""
        super().__init__(X, y, name="elasticnet", **kwargs)
        self.threshold = threshold
    
    def find_best_parameters(self) -> Dict[str, float]:
        """Find optimal ElasticNet parameters."""
        logger.info("Finding optimal ElasticNet parameters")
        
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scaler_cls = get_scaler_class(self.normalization)
        
        pipeline = Pipeline([
            ("scaler", scaler_cls()),
            ("model", LogisticRegression(
                penalty='elasticnet',
                solver="saga",
                max_iter=10000,
                random_state=self.random_state
            ))
        ])
        
        param_grid = {
            'model__C': loguniform(0.001, 1.0),
            'model__l1_ratio': uniform(0.5, 0.5)  # 0.5 to 1.0
        }
        
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=30, cv=cv_splitter,
            scoring="roc_auc", n_jobs=self.n_jobs, random_state=self.random_state
        )
        
        search.fit(self.X, self.y)
        
        self.params = {
            'C': search.best_params_['model__C'],
            'l1_ratio': search.best_params_['model__l1_ratio']
        }
        
        param_str = ", ".join(f"{k}={v:.6f}" for k, v in self.params.items())
        logger.info(f"ElasticNet best params: {param_str}, score: {search.best_score_:.4f}")
        
        return self.params
    
    def select_features(self) -> Tuple[List[str], Dict[str, float]]:
        """Select features using ElasticNet."""
        logger.info("Running ElasticNet feature selection")
        
        # Find optimal parameters
        if not self.params:
            self.find_best_parameters()
        
        # Create and fit model
        scaler_cls = get_scaler_class(self.normalization)
        pipeline = Pipeline([
            ("scaler", scaler_cls()),
            ("model", LogisticRegression(
                penalty='elasticnet',
                solver="saga",
                C=self.params['C'],
                l1_ratio=self.params['l1_ratio'],
                max_iter=10000,
                random_state=self.random_state
            ))
        ])
        
        pipeline.fit(self.X, self.y)
        
        # Extract coefficients
        model = pipeline.named_steps['model']
        importances = {col: abs(model.coef_[0][i]) for i, col in enumerate(self.X.columns)}
        
        # Apply threshold
        selected = [col for col, coef in importances.items() if coef >= self.threshold]
        
        # Track filtered features
        filtered = {col: coef for col, coef in importances.items() if coef < self.threshold}
        self.filtering_history['threshold'] = {
            'value': self.threshold,
            'filtered': filtered,
            'count': len(filtered)
        }
        
        # Apply feature limit if needed
        original_count = len(selected)
        selected = self.limit_features(selected, importances)
        
        # Track limited features
        if len(selected) < original_count:
            limited = {f: importances[f] for f in set(self.X.columns) - set(selected)}
            self.filtering_history['max_features'] = {
                'limit': self.max_features,
                'filtered': limited,
                'count': len(limited)
            }
        
        # Sort by importance
        self.features = sorted(selected, key=lambda x: importances[x], reverse=True)
        self.importances = importances
        
        logger.info(f"ElasticNet selected {len(self.features)} features")
        return self.features, self.importances
    
    def get_display_name(self) -> str:
        """Get display name for ElasticNet."""
        return "ElasticNet"
    
    def get_color(self) -> str:
        """Get color for ElasticNet."""
        return "#ff7f0e"  # Orange


class BootstrapSelector(FeatureSelector):
    """Feature selection using bootstrap sampling."""
    
    # Color mapping for different variants
    COLOR_MAP = {
        'upper_fence': '#d62728',  # Red
        'q1': '#9467bd'            # Purple
    }
    
    # Display name mappings
    VARIANT_NAMES = {
        'upper_fence': 'Upper Fence',
        'q1': 'Q1'
    }
    
    SELECTION_NAMES = {
        'ci_005': 'CI (α=0.05)',
        'ci_000': 'CI (α=0.00)'
    }
    
    REG_NAMES = {
        'lasso': 'LASSO',
        'elasticnet': 'ElasticNet'
    }
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, regularization: str,
                bootstrap_samples: int = 1000, variant: str = None, 
                selection: str = None, **kwargs):
        """Initialize bootstrap method with regularization."""
        name = f"bootstrap_{regularization}"
        if variant and selection:
            name = f"{name}_{variant}_{selection}"
            
        super().__init__(X, y, name=name, **kwargs)
        
        self.regularization = regularization
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_stats = {}
        self.standalone_params = {}
        
        self.variant = variant
        self.selection = selection
        self.threshold_info = {}
    
    def set_regularization_params(self, lasso_params: Optional[Dict[str, float]] = None, 
                                elasticnet_params: Optional[Dict[str, float]] = None) -> None:
        """Set parameters from standalone methods."""
        self.standalone_params = {
            'lasso': lasso_params or {},
            'elasticnet': elasticnet_params or {}
        }
    
    def run_bootstrap_iterations(self) -> Dict[str, FeatureStats]:
        """Run bootstrap iterations to collect coefficient statistics."""
        logger.info(f"Running {self.bootstrap_samples} bootstrap samples with {self.regularization}")
        start_time = time.time()
        
        # Get params for regularization
        reg_params = self.standalone_params.get(self.regularization, {})
        
        # Create and fit scaler
        scaler_cls = get_scaler_class(self.normalization)
        scaler = scaler_cls().fit(self.X)
        
        # Store coefficients for each feature
        all_coeffs = {feature: [] for feature in self.X.columns}

        # Run bootstrap iterations
        for i in tqdm(range(self.bootstrap_samples), desc="Bootstrap iterations"):
            # Create bootstrap sample with unique seed
            np.random.seed(self.random_state + i)
            indices = resample(np.arange(len(self.X)), replace=True)
            X_boot, y_boot = self.X.iloc[indices], self.y.iloc[indices]
            
            # Scale the data
            X_boot_scaled = scaler.transform(X_boot)
            
            # Create appropriate model based on regularization
            model = self._create_model_for_bootstrap(i, reg_params)
            
            # Fit model
            model.fit(X_boot_scaled, y_boot)
            
            # Store coefficients
            for j, feature in enumerate(self.X.columns):
                all_coeffs[feature].append(model.coef_[0][j])
        
        # Calculate statistics
        self.bootstrap_stats = self._calculate_bootstrap_stats(all_coeffs)
        
        duration = time.time() - start_time
        logger.info(f"Bootstrap completed in {duration:.2f} seconds")
        
        return self.bootstrap_stats
    
    def _create_model_for_bootstrap(self, iteration: int, reg_params: Dict[str, float]) -> LogisticRegression:
        """Create a model for bootstrap iteration."""
        if self.regularization == 'lasso':
            return LogisticRegression(
                penalty='l1',
                solver="liblinear",
                C=reg_params.get('C', 0.1),
                max_iter=10000,
                random_state=self.random_state + iteration
            )
        
        # Default to elasticnet
        return LogisticRegression(
            penalty='elasticnet',
            solver="saga",
            C=reg_params.get('C', 0.1),
            l1_ratio=reg_params.get('l1_ratio', 0.5),
            max_iter=10000,
            random_state=self.random_state + iteration
        )
    
    def _calculate_bootstrap_stats(self, all_coeffs: Dict[str, List[float]]) -> Dict[str, FeatureStats]:
        """Calculate statistics from bootstrap coefficients."""
        stats = {}
        for feature, coeffs in all_coeffs.items():
            coeffs_array = np.array(coeffs)
            mean = np.mean(coeffs_array)
            std_error = np.std(coeffs_array, ddof=1)
            
            # Confidence intervals
            lower_ci = np.percentile(coeffs_array, 2.5)
            upper_ci = np.percentile(coeffs_array, 97.5)
            
            # CV and p-value
            cv = std_error / abs(mean) if abs(mean) > 1e-10 else float('inf')
            t_stat = mean / (std_error / np.sqrt(len(coeffs_array))) if std_error > 0 else float('inf')
            p_value = 2 * (1 - ttest_ind.cdf(abs(t_stat), df=len(coeffs_array)-1))
            
            # Store stats
            stats[feature] = FeatureStats(
                mean=mean,
                std_error=std_error,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                ci_contains_zero=(lower_ci <= 0 <= upper_ci),
                cv=cv,
                p_value=p_value,
                t_stat=t_stat,
                all_coeffs=coeffs_array
            )
        
        return stats
    
    def select_features(self) -> Tuple[List[str], Dict[str, float]]:
        """Select features using bootstrap statistics."""
        # Run bootstrap if needed
        if not self.bootstrap_stats:
            self.bootstrap_stats = self.run_bootstrap_iterations()
        
        # If no variant specified, return empty
        if not self.variant or not self.selection:
            return [], {}
        
        # Apply post-processing
        return self._apply_feature_selection()
    
    def _apply_feature_selection(self) -> Tuple[List[str], Dict[str, float]]:
        """Apply post-processing with specified variant and selection."""
        if not self.variant or not self.selection:
            logger.error("Cannot select features: variant or selection not specified")
            return [], {}
        
        logger.info(f"Applying bootstrap selection: {self.variant} + {self.selection} + {self.regularization}")
        
        # Determine CV threshold based on variant
        cv_threshold = self._determine_cv_threshold()
        
        # Apply feature selection
        selected, filtered_by_cv, filtered_by_ci = self._apply_selection_criteria(cv_threshold)
        
        # Calculate importances
        importances = {f: abs(self.bootstrap_stats[f].mean) for f in selected}
        
        # Track filtering history
        self.filtering_history['cv'] = {
            'threshold': cv_threshold,
            'filtered': filtered_by_cv,
            'count': len(filtered_by_cv)
        }
        
        alpha = 0.05 if self.selection == 'ci_005' else 0.0
        self.filtering_history['ci'] = {
            'alpha': alpha,
            'filtered': filtered_by_ci,
            'count': len(filtered_by_ci)
        }
        
        # Apply feature limit
        original_count = len(selected)
        selected = self.limit_features(selected, importances)
        
        # Track limited features if any were removed
        if len(selected) < original_count:
            limited = {f: importances[f] for f in set(selected) - set(selected[:self.max_features])}
            self.filtering_history['max_features'] = {
                'limit': self.max_features,
                'filtered': limited,
                'count': len(limited)
            }
        
        # Sort features
        self.features = sorted(selected, key=lambda x: importances[x], reverse=True)
        self.importances = importances
        
        # Store threshold info
        self.threshold_info = {
            'value': cv_threshold,
            'variant': self.variant,
            'selection': self.selection,
            'alpha': alpha,
            'regularization': self.regularization
        }
        
        logger.info(f"Bootstrap selected {len(self.features)} features")
        return self.features, self.importances
    
    def _apply_selection_criteria(self, cv_threshold: float) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Apply CV and CI selection criteria."""
        selected = []
        filtered_by_cv = {}
        filtered_by_ci = {}
        
        # Apply selection criteria - alpha determines confidence interval threshold
        alpha = 0.05 if self.selection == 'ci_005' else 0.0
        
        for feature, stats in self.bootstrap_stats.items():
            # Skip features with CV above threshold
            if stats.cv > cv_threshold:
                filtered_by_cv[feature] = stats.cv
                continue
            
            # Check if CI contains zero
            contains_zero = False
            if alpha == 0.0:
                # Strict check - any coefficient is zero
                coeffs = stats.all_coeffs
                contains_zero = min(coeffs) <= 0 <= max(coeffs)
            else:
                # Standard CI check
                contains_zero = stats.ci_contains_zero
            
            if contains_zero:
                filtered_by_ci[feature] = {
                    'lower_ci': stats.lower_ci,
                    'upper_ci': stats.upper_ci,
                    'mean': stats.mean
                }
                continue
            
            # Feature passed all filters
            selected.append(feature)
            
        return selected, filtered_by_cv, filtered_by_ci
    
    def _determine_cv_threshold(self) -> float:
        """Determine CV threshold based on variant."""
        # Get valid CV values (exclude infinity)
        cv_values = [s.cv for s in self.bootstrap_stats.values() if s.cv < float('inf')]
        
        if not cv_values:
            return float('inf')
        
        # Calculate threshold based on variant
        if self.variant == 'upper_fence':
            q1, q3 = np.percentile(cv_values, [25, 75])
            threshold = q3 + 1.5 * (q3 - q1)
            logger.info(f"CV threshold (upper fence): {threshold:.4f}")
            
        elif self.variant == 'q1':
            threshold = np.percentile(cv_values, 25)
            logger.info(f"CV threshold (Q1): {threshold:.4f}")
            
        else:
            threshold = float('inf')
            logger.warning(f"Unknown variant '{self.variant}', using infinite threshold")
        
        return threshold
    
    def get_display_name(self) -> str:
        """Get display name for bootstrap method."""
        if not self.variant or not self.selection:
            return f"Bootstrap ({self.regularization})"
            
        variant = self.VARIANT_NAMES.get(self.variant, self.variant)
        selection = self.SELECTION_NAMES.get(self.selection, self.selection)
        reg = self.REG_NAMES.get(self.regularization, self.regularization)
        
        return f"{variant} + {selection} + {reg}"
    
    def get_path(self) -> Path:
        """Get path for storing results."""
        if not self.variant or not self.selection:
            return Path(self.regularization)
        return Path(self.regularization) / self.variant / self.selection
    
    def get_color(self) -> str:
        """Get color for bootstrap method."""
        if not self.variant:
            return "gray"
        return self.COLOR_MAP.get(self.variant, "gray")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "regularization": self.regularization,
            "variant": self.variant,
            "selection": self.selection,
            "threshold_info": self.threshold_info
        })
        return result


class FeatureOrderer:
    """Orders features based on different strategies."""
    
    def __init__(self, feature_aucs: Optional[Dict[str, float]] = None):
        """Initialize with feature AUCs."""
        self.feature_aucs = feature_aucs or {}
    
    def order_by_coefficient(self, features: List[str], importances: Dict[str, float]) -> List[str]:
        """Order features by coefficient magnitude."""
        return sorted(features, key=lambda x: importances.get(x, 0), reverse=True)
    
    def order_by_auc(self, features: List[str]) -> List[str]:
        """Order features by individual AUC."""
        return sorted(features, key=lambda x: self.feature_aucs.get(x, 0), reverse=True)
    
    def create_ordered_variants(self, method_id: str, method: FeatureSelector) -> Dict[str, Dict[str, Any]]:
        """Create ordered variants of a method."""
        variants = {}
        
        # Coefficient ordering
        coef_id = f"{method_id}_coef"
        coef_ordered = self.order_by_coefficient(method.features, method.importances)
        variants[coef_id] = {
            "features": coef_ordered,
            "importances": method.importances,
            "ordering": "coef"
        }
        
        # AUC ordering
        auc_id = f"{method_id}_auc"
        auc_ordered = self.order_by_auc(method.features)
        variants[auc_id] = {
            "features": auc_ordered,
            "importances": self.feature_aucs,
            "ordering": "auc"
        }
        
        return variants


class ModelEvaluator:
    """Evaluates feature sets using cross-validation."""
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series, 
                external_datasets: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]] = None,
                cv: int = 10, random_state: int = 42, 
                normalization: str = "z_score", n_jobs: int = -1):
        """Initialize with datasets and parameters."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.external_datasets = external_datasets or {}
        self.cv = cv
        self.random_state = random_state
        self.normalization = normalization
        self.n_jobs = n_jobs
        
        # Results storage
        self.results = {}
        self.optimal_counts = {}
    
    def evaluate_sequential(self, method_id: str, features: List[str], 
                          base_method: Optional[FeatureSelector] = None) -> List[Dict[str, Any]]:
        """Evaluate features by adding them sequentially."""
        logger.info(f"Evaluating {method_id} with {len(features)} features")
        
        results = []
        current_features = []
        
        for i, feature in enumerate(features):
            # Add next feature
            current_features.append(feature)
            feature_set = current_features.copy()
            
            logger.info(f"  Step {i+1}/{len(features)}: Adding '{feature}'")
            
            # Cross-validate with current features
            cv_results = self._cross_validate(feature_set)
            
            # Train final model with best parameters
            pipeline = self._train_model(feature_set, cv_results.get('best_params', {}))
            
            # Evaluate on test set
            test_results = self._evaluate_on_dataset(pipeline, feature_set, self.X_test, self.y_test, "test")
            
            # Evaluate on external datasets
            dataset_results = {}
            for name, (X, y) in self.external_datasets.items():
                if self._check_features(X, feature_set):
                    dataset_results[name] = self._evaluate_on_dataset(pipeline, feature_set, X, y, name)
            
            # Evaluate on combined datasets
            overall_results = self._evaluate_combined(pipeline, feature_set)
            
            # Save coefficients
            model_info = {}
            if pipeline:
                model = pipeline.named_steps['model']
                model_info = {
                    "intercept": model.intercept_[0],
                    "coefficients": {feature_set[j]: model.coef_[0][j] for j in range(len(feature_set))}
                }
            
            # Store result
            result = {
                "method": method_id,
                "num_features": i + 1,
                "added_feature": feature,
                "feature_set": feature_set,
                "cv_results": cv_results,
                "test_results": test_results,
                "dataset_results": dataset_results,
                "overall_results": overall_results,
                "model_info": model_info
            }
            
            results.append(result)
            
            # Log key metrics
            logger.info(f"    CV AUC: {cv_results.get('mean_auc', 0):.4f}")
            if test_results:
                logger.info(f"    Test AUC: {test_results.get('auc', 0):.4f}")
        
        self.results[method_id] = results
        return results
    
    def _cross_validate(self, features: List[str]) -> Dict[str, Any]:
        """Perform cross-validation for a feature set."""
        if not features:
            return {"mean_auc": 0, "std_auc": 0}
        
        X_subset = self.X_train[features]
        
        # Create CV splitter
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Create pipeline
        scaler_cls = get_scaler_class(self.normalization)
        pipeline = Pipeline([
            ("scaler", scaler_cls()),
            ("model", LogisticRegression(
                penalty="elasticnet", 
                solver="saga", 
                max_iter=10000, 
                random_state=self.random_state
            ))
        ])
        
        # Create parameter grid
        param_grid = {
            'model__C': loguniform(0.001, 1.0),
            'model__l1_ratio': uniform(0.5, 0.5)
        }
        
        # Run grid search
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=30, 
            cv=cv_splitter, scoring="roc_auc",
            n_jobs=self.n_jobs, random_state=self.random_state
        )
        
        search.fit(X_subset, self.y_train)
        
        # Extract scores
        fold_scores = [search.cv_results_[f"split{i}_test_score"][search.best_index_] 
                      for i in range(self.cv)]
        
        return {
            "mean_auc": search.best_score_,
            "std_auc": np.std(fold_scores),
            "fold_scores": fold_scores,
            "best_params": search.best_params_
        }
    
    def _train_model(self, features: List[str], params: Dict[str, Any]) -> Optional[Pipeline]:
        """Train a model with the given features and parameters."""
        if not features:
            return None
        
        X_subset = self.X_train[features]
        
        # Create pipeline
        scaler_cls = get_scaler_class(self.normalization)
        pipeline = Pipeline([
            ("scaler", scaler_cls()),
            ("model", LogisticRegression(
                penalty="elasticnet", 
                solver="saga",
                C=params.get('model__C', 0.1),
                l1_ratio=params.get('model__l1_ratio', 0.5),
                max_iter=10000, 
                random_state=self.random_state
            ))
        ])
        
        pipeline.fit(X_subset, self.y_train)
        return pipeline
    
    def _evaluate_on_dataset(self, pipeline: Optional[Pipeline], features: List[str], 
                          X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        # Skip evaluation if there's no pipeline, no features, or too few samples
        if not pipeline or not features or len(y) < 10 or y.nunique() < 2:
            return {}
        
        X_subset = X[features]
        
        # Get decision scores
        scores = pipeline.decision_function(X_subset)
        
        # Split by class
        simple_scores = scores[y == 0]
        expert_scores = scores[y == 1]
        
        if len(simple_scores) < 2 or len(expert_scores) < 2:
            return {}
        
        # Calculate statistics
        simple_mean = np.mean(simple_scores)
        expert_mean = np.mean(expert_scores)
        
        # Adjust scores for ROC if needed
        scores_for_roc = -scores if simple_mean > expert_mean else scores
        
        # Calculate AUC and ROC curve
        auc = roc_auc_score(y, scores_for_roc)
        fpr, tpr, thresholds = roc_curve(y, scores_for_roc)
        best_idx = np.argmax(tpr - fpr)
        
        # Statistical test
        stat, pval, test_name = statistical_test(simple_scores, expert_scores)
        
        # Calculate all metrics
        all_scores = np.concatenate([simple_scores, expert_scores])
        all_labels = np.concatenate([np.zeros(len(simple_scores)), np.ones(len(expert_scores))])
        
        return {
            "auc": auc,
            "mean_difference": expert_mean - simple_mean,
            "simple_mean": simple_mean,
            "expert_mean": expert_mean,
            "simple_std": np.std(simple_scores),
            "expert_std": np.std(expert_scores),
            "cohens_d": cohens_d(simple_scores, expert_scores),
            "overlap_coefficient": overlap_coefficient(simple_scores, expert_scores),
            "js_divergence": js_divergence(simple_scores, expert_scores),
            "mutual_information": mutual_information(all_scores, all_labels),
            "n_simple": len(simple_scores),
            "n_expert": len(expert_scores),
            "p_value": pval
        }
    
    def _evaluate_combined(self, pipeline: Optional[Pipeline], features: List[str]) -> Dict[str, Any]:
        """Evaluate on combined external datasets."""
        if not self.external_datasets:
            return {}
        
        all_X = []
        all_y = []
        
        for name, (X, y) in self.external_datasets.items():
            if self._check_features(X, features):
                all_X.append(X[features])
                all_y.append(y)
        
        if not all_X:
            return {}
        
        # Combine datasets
        X_combined = pd.concat(all_X, axis=0, ignore_index=True)
        y_combined = pd.concat(all_y, axis=0, ignore_index=True)
        
        return self._evaluate_on_dataset(pipeline, features, X_combined, y_combined, "overall")
    
    def _check_features(self, X: pd.DataFrame, features: List[str]) -> bool:
        """Check if dataset has all required features."""
        return all(feature in X.columns for feature in features)
    
    def find_optimal_feature_counts(self) -> None:
        """Find optimal feature counts for each method and metric."""
        for method_id, evaluations in self.results.items():
            self.optimal_counts[method_id] = {
                "cv": {},
                "test": {},
                "overall": {},
                "datasets": {}
            }
            
            # Find optimal for CV AUC
            self._find_optimal_for_cv(method_id, evaluations)
            
            # Find optimal for test metrics
            self._find_optimal_for_dataset(method_id, evaluations, "test")
            
            # Find optimal for overall metrics
            self._find_optimal_for_dataset(method_id, evaluations, "overall")
            
            # Find optimal for each external dataset
            if evaluations and "dataset_results" in evaluations[0]:
                for dataset in evaluations[0]["dataset_results"].keys():
                    self._find_optimal_for_dataset(method_id, evaluations, dataset, is_nested=True)
    
    def _find_optimal_for_cv(self, method_id: str, evaluations: List[Dict[str, Any]]) -> None:
        """Find optimal feature count for CV AUC."""
        cv_data = [(e["num_features"], e["cv_results"]["mean_auc"]) 
                   for e in evaluations if "cv_results" in e]
        
        if not cv_data:
            return
        
        # Find best AUC
        best_idx = max(range(len(cv_data)), key=lambda i: cv_data[i][1])
        best_count, best_auc = cv_data[best_idx]
        
        self.optimal_counts[method_id]["cv"] = {
            "count": best_count,
            "features": evaluations[best_idx]["feature_set"],
            "mean_auc": best_auc,
            "std_auc": evaluations[best_idx]["cv_results"].get("std_auc", 0)
        }
    
    def _find_optimal_for_dataset(self, method_id: str, evaluations: List[Dict[str, Any]], 
                               dataset: str, is_nested: bool = False) -> None:
        """Find optimal feature counts for different metrics."""
        for metric in METRICS:
            self._find_optimal_for_metric(method_id, evaluations, dataset, metric, is_nested)
    
    def _find_optimal_for_metric(self, method_id: str, evaluations: List[Dict[str, Any]],
                              dataset: str, metric: str, is_nested: bool) -> None:
        """Find optimal feature count for a specific metric."""
        data_points = self._extract_metric_data_points(evaluations, dataset, metric, is_nested)
        
        if not data_points:
            return
        
        # For some metrics, maximize absolute value
        if metric in ["mean_difference"]:
            best_idx = max(range(len(data_points)), key=lambda i: abs(data_points[i][1]))
        else:
            best_idx = max(range(len(data_points)), key=lambda i: data_points[i][1])
        
        best_count, best_value = data_points[best_idx]
        
        # Store optimal
        if is_nested:
            if dataset not in self.optimal_counts[method_id]["datasets"]:
                self.optimal_counts[method_id]["datasets"][dataset] = {}
            
            self.optimal_counts[method_id]["datasets"][dataset][f"{metric}_count"] = best_count
            self.optimal_counts[method_id]["datasets"][dataset][metric] = best_value
        else:
            self.optimal_counts[method_id][dataset][f"{metric}_count"] = best_count
            self.optimal_counts[method_id][dataset][metric] = best_value
    
    def _extract_metric_data_points(self, evaluations: List[Dict[str, Any]], 
                                 dataset: str, metric: str, 
                                 is_nested: bool) -> List[Tuple[int, float]]:
        """Extract data points for a specific metric."""
        data_points = []
        
        for e in evaluations:
            if is_nested:
                if ("dataset_results" in e and dataset in e["dataset_results"] and 
                    metric in e["dataset_results"][dataset]):
                    data_points.append((
                        e["num_features"], 
                        e["dataset_results"][dataset][metric]
                    ))
            else:
                result_key = f"{dataset}_results"
                if result_key in e and metric in e[result_key]:
                    data_points.append((
                        e["num_features"], 
                        e[result_key][metric]
                    ))
        
        return data_points


class ResultsVisualizer:
    """Creates visualizations for feature selection results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        
        # Create visualization directories
        self.dirs = {
            'learning_curves': self.output_dir / "learning_curves",
            'auc_curves': self.output_dir / "auc_curves",
            'feature_importances': self.output_dir / "feature_importances",
            'filtering': self.output_dir / "filtering",
            'metrics': self.output_dir / "metrics"
        }
        
        # Create all directories
        for path in self.dirs.values():
            path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        for subdir in ['test', 'overall', 'datasets']:
            (self.dirs['auc_curves'] / subdir).mkdir(exist_ok=True)
            (self.dirs['metrics'] / subdir).mkdir(exist_ok=True)
        
        # Set visualization style
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    
    def plot_all(self, methods: Dict[str, FeatureSelector], 
                results: Dict[str, List[Dict[str, Any]]], 
                optimal_counts: Dict[str, Dict[str, Any]]) -> None:
        """Generate all visualizations."""
        logger.info("Generating visualizations")
        
        # Learning curves
        self.plot_learning_curves(methods, results, optimal_counts)
        
        # AUC curves
        self.plot_metric_curves_by_dataset(methods, results, optimal_counts, 
                                          "auc", "test")
        self.plot_metric_curves_by_dataset(methods, results, optimal_counts, 
                                          "auc", "overall")
        self.plot_dataset_metric_curves(methods, results, optimal_counts, "auc")
        
        # Feature importances
        self.plot_feature_importances(methods)
        
        # Bootstrap analysis (only for bootstrap methods)
        for method_id, method in methods.items():
            if isinstance(method, BootstrapSelector) and "bootstrap" in method_id:
                if hasattr(method, 'bootstrap_stats') and method.bootstrap_stats:
                    self.plot_cv_distribution(method, method.bootstrap_stats)
                    self.plot_ci_analysis(method, method.bootstrap_stats)
                    self.plot_p_value_distribution(method, method.bootstrap_stats)
        
        # Filtering analysis
        self.plot_filtering_analysis(methods)
        
        # Metric curves for other metrics
        for metric in [m for m in METRICS if m != "auc"]:
            self.plot_metric_curves_by_dataset(methods, results, optimal_counts, metric, "test")
            self.plot_metric_curves_by_dataset(methods, results, optimal_counts, metric, "overall")
            self.plot_dataset_metric_curves(methods, results, optimal_counts, metric)
        
        logger.info("Visualization complete")
    
    def plot_learning_curves(self, methods: Dict[str, FeatureSelector], 
                           results: Dict[str, List[Dict[str, Any]]], 
                           optimal_counts: Dict[str, Dict[str, Any]]) -> None:
        """Plot learning curves showing CV performance."""
        logger.info("Plotting learning curves")
        
        # Plot individual curves
        for method_id, evaluations in results.items():
            self._plot_individual_learning_curve(method_id, methods, evaluations)
        
        # Plot combined curves
        self._plot_combined_learning_curves(methods, results, optimal_counts)
    
    def _plot_individual_learning_curve(self, method_id: str, 
                                     methods: Dict[str, FeatureSelector], 
                                     evaluations: List[Dict[str, Any]]) -> None:
        """Plot learning curve for an individual method."""
        # Get original method
        method = self._get_base_method(method_id, methods)
        
        if not method or not evaluations:
            return
        
        # Extract learning curve data
        nums = [e["num_features"] for e in evaluations]
        means = [e["cv_results"]["mean_auc"] for e in evaluations]
        stds = [e["cv_results"]["std_auc"] for e in evaluations]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot curve
        ax.plot(nums, means, marker='o', color=method.get_color(), 
               label=f"{method.get_display_name()} (Mean AUC)", linewidth=2)
        
        # Plot error band
        ax.fill_between(
            nums,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2, color=method.get_color(), label="±1 Std Dev"
        )
        
        # Mark optimal point
        best_idx = np.argmax(means)
        ax.scatter(nums[best_idx], means[best_idx], s=150, color=method.get_color(), 
                  marker='*', edgecolor='black', zorder=10,
                  label=f"Optimal: {nums[best_idx]} features")
        
        # Style plot
        ax.set_title(f"Cross-Validation Learning Curve - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("Number of Features", fontsize=12)
        ax.set_ylabel("AUC (Cross-Validation)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # Save
        method_dir = self.dirs['learning_curves'] / method.get_path()
        method_dir.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(method_dir / f"{method_id}_learning_curve.png", dpi=300)
        plt.close(fig)
    
    def _plot_combined_learning_curves(self, methods: Dict[str, FeatureSelector],
                                    results: Dict[str, List[Dict[str, Any]]],
                                    optimal_counts: Dict[str, Dict[str, Any]]) -> None:
        """Plot combined learning curves for all methods."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method_id, evaluations in results.items():
            # Only use coefficient ordered variants
            if not method_id.endswith("_coef"):
                continue
                
            method = self._get_base_method(method_id, methods)
            
            if not method or not evaluations:
                continue
            
            # Extract data
            nums = [e["num_features"] for e in evaluations]
            means = [e["cv_results"]["mean_auc"] for e in evaluations]
            
            # Plot curve
            ax.plot(nums, means, marker='o', color=method.get_color(), 
                   label=method.get_display_name(), linewidth=2)
            
            # Mark optimal
            optimal = optimal_counts.get(method_id, {}).get("cv", {})
            if "count" in optimal:
                ax.scatter(optimal["count"], optimal["mean_auc"], s=150, color=method.get_color(),
                          marker='*', edgecolor='black', zorder=10)
        
        # Style plot
        ax.set_title("Cross-Validation Learning Curves - All Methods", fontsize=16)
        ax.set_xlabel("Number of Features", fontsize=14)
        ax.set_ylabel("AUC (Cross-Validation Mean)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12, loc='lower right')
        
        # Save
        plt.tight_layout()
        plt.savefig(self.dirs['learning_curves'] / "all_methods.png", dpi=300)
        plt.close(fig)
    
    def plot_metric_curves_by_dataset(self, methods: Dict[str, FeatureSelector], 
                                    results: Dict[str, List[Dict[str, Any]]],
                                    optimal_counts: Dict[str, Dict[str, Any]],
                                    metric: str, dataset: str) -> None:
        """Plot metric curves for a specific dataset."""
        logger.info(f"Plotting {metric} curves for {dataset}")
        
        dataset_dir = self.dirs['metrics'] if metric != "auc" else self.dirs['auc_curves']
        dataset_dir = dataset_dir / dataset
        
        # Get metric display name
        metric_display = self._get_metric_display_name(metric)
        
        # Plot individual curves
        for method_id, evaluations in results.items():
            self._plot_individual_metric_curve(
                method_id, methods, evaluations, optimal_counts,
                metric, metric_display, dataset, dataset_dir, is_nested=False
            )
        
        # Plot combined curves
        self._plot_combined_metric_curves(
            methods, results, metric, metric_display, dataset, dataset_dir
        )
    
    def plot_dataset_metric_curves(self, methods: Dict[str, FeatureSelector],
                                 results: Dict[str, List[Dict[str, Any]]],
                                 optimal_counts: Dict[str, Dict[str, Any]],
                                 metric: str) -> None:
        """Plot metric curves for all external datasets."""
        logger.info(f"Plotting dataset {metric} curves")
        
        # Find all available datasets
        datasets = set()
        for evaluations in results.values():
            for e in evaluations:
                if "dataset_results" in e:
                    datasets.update(e["dataset_results"].keys())
        
        # Get metric display name
        metric_display = self._get_metric_display_name(metric)
        
        # Plot curves for each dataset
        for dataset in datasets:
            dataset_dir = self.dirs['metrics'] if metric != "auc" else self.dirs['auc_curves']
            dataset_dir = dataset_dir / "datasets" / dataset
            dataset_dir.mkdir(exist_ok=True, parents=True)
            
            # Plot individual method curves
            for method_id, evaluations in results.items():
                self._plot_individual_metric_curve(
                    method_id, methods, evaluations, optimal_counts,
                    metric, metric_display, dataset, dataset_dir, is_nested=True
                )
            
            # Plot combined curve for this dataset
            self._plot_combined_metric_curves(
                methods, results, metric, metric_display, dataset, dataset_dir, is_nested=True
            )
    
    def _plot_individual_metric_curve(self, method_id: str, 
                                   methods: Dict[str, FeatureSelector],
                                   evaluations: List[Dict[str, Any]],
                                   optimal_counts: Dict[str, Dict[str, Any]],
                                   metric: str, metric_display: str,
                                   dataset: str, output_dir: Path,
                                   is_nested: bool = False) -> None:
        """Plot metric curve for an individual method."""
        method = self._get_base_method(method_id, methods)
        
        if not method:
            return
        
        # Extract data
        data_points = self._extract_metric_data_points(
            evaluations, dataset, metric, is_nested
        )
        
        if not data_points:
            return
            
        nums, values = zip(*data_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot curve
        ax.plot(nums, values, marker='o', color=method.get_color(), 
               label=method.get_display_name(), linewidth=2)
        
        # Mark optimal point
        self._mark_optimal_point(
            ax, method_id, optimal_counts, method,
            dataset, metric, nums, values, is_nested
        )
        
        # Style plot
        ax.set_title(f"{dataset} {metric_display} - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("Number of Features", fontsize=12)
        ax.set_ylabel(metric_display, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # Save
        method_dir = output_dir / method.get_path()
        method_dir.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(method_dir / f"{method_id}_{dataset}_{metric}.png", dpi=300)
        plt.close(fig)
    
    def _mark_optimal_point(self, ax: plt.Axes, method_id: str, 
                          optimal_counts: Dict[str, Dict[str, Any]],
                          method: FeatureSelector, dataset: str, 
                          metric: str, nums: List[int], values: List[float],
                          is_nested: bool) -> None:
        """Mark the optimal point on the plot."""
        count = self._get_optimal_feature_count(
            method_id, optimal_counts, dataset, metric, is_nested
        )
        
        if count is None:
            return
            
        best_idx = min(range(len(nums)), key=lambda i: abs(nums[i] - count))
        ax.scatter(nums[best_idx], values[best_idx], s=150, color=method.get_color(),
                  marker='*', edgecolor='black', zorder=10,
                  label=f"Optimal: {count} features")
    
    def _get_optimal_feature_count(self, method_id: str, 
                                optimal_counts: Dict[str, Dict[str, Any]],
                                dataset: str, metric: str, 
                                is_nested: bool) -> Optional[int]:
        """Get the optimal feature count for a method and metric."""
        if method_id not in optimal_counts:
            return None
            
        count_key = f"{metric}_count"
        
        if is_nested:
            # For dataset in datasets
            datasets_dict = optimal_counts[method_id].get("datasets", {})
            if dataset not in datasets_dict:
                return None
                
            return datasets_dict[dataset].get(count_key)
        
        # For test or overall
        if dataset not in optimal_counts[method_id]:
            return None
            
        return optimal_counts[method_id][dataset].get(count_key)
    
    def _plot_combined_metric_curves(self, methods: Dict[str, FeatureSelector],
                                  results: Dict[str, List[Dict[str, Any]]],
                                  metric: str, metric_display: str,
                                  dataset: str, output_dir: Path,
                                  is_nested: bool = False) -> None:
        """Plot combined metric curves for all methods."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method_id, evaluations in results.items():
            # Only use coefficient ordered variants
            if not method_id.endswith("_coef"):
                continue
                
            method = self._get_base_method(method_id, methods)
            
            if not method:
                continue
            
            # Extract data
            data_points = self._extract_metric_data_points(
                evaluations, dataset, metric, is_nested
            )
            
            if not data_points:
                continue
                
            nums, values = zip(*data_points)
            
            # Plot curve
            ax.plot(nums, values, marker='o', color=method.get_color(), 
                   label=method.get_display_name(), linewidth=2)
        
        # Style plot
        ax.set_title(f"{dataset} {metric_display} - All Methods", fontsize=16)
        ax.set_xlabel("Number of Features", fontsize=14)
        ax.set_ylabel(metric_display, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12, loc='lower right')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_{metric}_all_methods.png", dpi=300)
        plt.close(fig)
    
    def plot_feature_importances(self, methods: Dict[str, FeatureSelector], n_features: int = 20) -> None:
        """Plot feature importances for each method."""
        logger.info("Plotting feature importances")
        
        for method_id, method in methods.items():
            # Skip ordered variants
            if method_id.endswith("_coef") or method_id.endswith("_auc"):
                continue
                
            if not method.features or not method.importances:
                continue
            
            # Limit number of features to display
            features = method.features[:min(n_features, len(method.features))]
            if not features:
                continue
            
            # Create DataFrame for plotting
            importances = [method.importances.get(f, 0) for f in features]
            directions = ['Positive' if method.importances.get(f, 0) > 0 else 'Negative' 
                         for f in features]
            
            df = pd.DataFrame({
                'Feature': features,
                'Importance': [abs(method.importances.get(f, 0)) for f in features],
                'Direction': directions
            })
            
            # Sort by importance
            df = df.sort_values('Importance', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.3)))
            
            # Plot horizontal bar chart
            sns.barplot(x='Importance', y='Feature', hue='Direction', data=df,
                       palette={'Positive': 'blue', 'Negative': 'red'}, alpha=0.7, ax=ax)
            
            # Add value labels
            for i, p in enumerate(ax.patches):
                ax.text(p.get_width() + 0.01, p.get_y() + p.get_height()/2, 
                       f"{p.get_width():.4f}", va='center', fontsize=8)
            
            # Style plot
            ax.set_title(f"Top {len(features)} Feature Importances - {method.get_display_name()}", 
                        fontsize=14)
            ax.set_xlabel("Absolute Coefficient Value", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # Save
            method_dir = self.dirs['feature_importances'] / method.get_path()
            method_dir.mkdir(exist_ok=True, parents=True)
            plt.tight_layout()
            plt.savefig(method_dir / f"importances_top{len(features)}.png", dpi=300)
            plt.close(fig)
    
    def plot_filtering_analysis(self, methods: Dict[str, FeatureSelector]) -> None:
        """Plot analysis of feature filtering across methods."""
        logger.info("Plotting filtering analysis")
        
        filtering_dir = self.dirs['filtering']
        
        # Get filtering stats for each method
        method_filtering = self._get_filtering_stats(methods)
        
        if not method_filtering:
            logger.warning("No filtering data to visualize")
            return
        
        # Create filtering summary
        self._plot_filtering_summary(method_filtering, filtering_dir)
        
        # Create stage-by-stage filtering visualization
        self._plot_filtering_stages(method_filtering, filtering_dir)
    
    def _get_filtering_stats(self, methods: Dict[str, FeatureSelector]) -> List[Dict[str, Any]]:
        """Extract filtering statistics from methods."""
        method_filtering = []
        
        for method_id, method in methods.items():
            # Skip ordered variants
            if method_id.endswith("_coef") or method_id.endswith("_auc"):
                continue
                
            if not method.filtering_history:
                continue
            
            # Count initial features (sum of filtered + selected)
            filtered_count = sum(info.get('count', 0) for info in method.filtering_history.values())
            total_count = filtered_count + len(method.features)
            
            # Get filtering by stage
            stages = {}
            for stage, info in method.filtering_history.items():
                count = info.get('count', 0)
                if count > 0:
                    stages[stage] = count
            
            # Add to data
            method_filtering.append({
                'method_id': method_id,
                'method_name': method.get_display_name(),
                'total_features': total_count,
                'filtered_count': filtered_count,
                'selected_count': len(method.features),
                'stages': stages
            })
        
        return method_filtering
    
    def _plot_filtering_summary(self, method_filtering: List[Dict[str, Any]], 
                             output_dir: Path) -> None:
        """Plot summary of feature filtering across methods."""
        # Create DataFrame for plotting
        rows = []
        for mf in method_filtering:
            rows.append({
                'Method': mf['method_name'],
                'Selected': mf['selected_count'],
                'Filtered': mf['filtered_count'],
                'Total': mf['total_features']
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by selected count
        df = df.sort_values('Selected', ascending=False)
        
        # Calculate percentages
        df['Selected %'] = 100 * df['Selected'] / df['Total']
        df['Filtered %'] = 100 * df['Filtered'] / df['Total']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 2))
        
        # Plot horizontal stacked bar chart
        ax.barh(df['Method'], df['Selected %'], color='#4CAF50', label='Selected')
        ax.barh(df['Method'], df['Filtered %'], left=df['Selected %'], color='#F44336', label='Filtered')
        
        # Add percentage labels
        for i, row in df.iterrows():
            if row['Selected %'] > 5:
                ax.text(row['Selected %'] / 2, i, f"{row['Selected %']:.1f}%", 
                       ha='center', va='center', color='white', fontweight='bold')
            
            if row['Filtered %'] > 5:
                ax.text(row['Selected %'] + row['Filtered %'] / 2, i, f"{row['Filtered %']:.1f}%", 
                       ha='center', va='center', color='white', fontweight='bold')
        
        # Style plot
        ax.set_title("Feature Selection Rate by Method", fontsize=14)
        ax.set_xlabel("Percentage of Features", fontsize=12)
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
        
        # Save
        plt.tight_layout()
        plt.savefig(output_dir / "feature_selection_rate.png", dpi=300)
        plt.close(fig)
    
    def _plot_filtering_stages(self, method_filtering: List[Dict[str, Any]], 
                            output_dir: Path) -> None:
        """Plot filtering by stage for each method."""
        # Get all possible stages
        all_stages = set()
        for mf in method_filtering:
            all_stages.update(mf['stages'].keys())
        
        # Create stage order
        stage_order = ['correlation', 'cv', 'ci', 'threshold', 'max_features']
        
        # Add any missing stages
        for stage in all_stages:
            if stage not in stage_order:
                stage_order.append(stage)
        
        # Create DataFrame for plotting
        rows = []
        for mf in method_filtering:
            row = {'Method': mf['method_name']}
            
            # Add each stage
            for stage in stage_order:
                row[stage] = mf['stages'].get(stage, 0)
            
            # Add selected count
            row['Selected'] = mf['selected_count']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by selected count
        df = df.sort_values('Selected', ascending=False)
        
        # Create color mapping
        colors = {
            'correlation': '#66c2a5',
            'cv': '#fc8d62',
            'ci': '#8da0cb',
            'threshold': '#e78ac3',
            'max_features': '#a6d854',
            'Selected': '#000000'
        }
        
        # Fill in any missing stages with gray
        for stage in stage_order:
            if stage not in colors:
                colors[stage] = '#cccccc'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(df) * 0.8 + 2))
        
        # Plot stacked bar chart
        prev = np.zeros(len(df))
        
        for stage in stage_order:
            if stage in df.columns:
                ax.barh(df['Method'], df[stage], left=prev, color=colors[stage], 
                       label=stage.replace('_', ' ').title())
                prev += df[stage]
        
        # Add selected bars
        ax.barh(df['Method'], df['Selected'], left=prev, color=colors['Selected'], 
               label='Selected', alpha=0.6)
        
        # Style plot
        ax.set_title("Feature Filtering by Stage", fontsize=14)
        ax.set_xlabel("Number of Features", fontsize=12)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # Save
        plt.tight_layout()
        plt.savefig(output_dir / "feature_filtering_by_stage.png", dpi=300)
        plt.close(fig)
    
    def plot_cv_distribution(self, method: BootstrapSelector, 
                           bootstrap_stats: Dict[str, FeatureStats]) -> None:
        """Plot distribution of coefficient of variation (CV) values."""
        logger.info(f"Plotting CV distribution for {method.get_display_name()}")
        
        # Get CV values (excluding infinity)
        cvs = [stats.cv for stats in bootstrap_stats.values() if stats.cv < float('inf')]
        features = [f for f, stats in bootstrap_stats.items() if stats.cv < float('inf')]
        
        if not cvs:
            logger.warning(f"No valid CV values for {method.get_display_name()}")
            return
        
        # Create directory
        cv_dir = self.dirs['filtering'] / "cv" / method.get_path()
        cv_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(cvs, bins=30, kde=True, ax=ax, color='skyblue',
                    edgecolor='black', alpha=0.7)
        
        # Add threshold line if available
        if method.threshold_info and 'value' in method.threshold_info:
            threshold = method.threshold_info['value']
            ax.axvline(threshold, color='red', linestyle='--',
                      linewidth=2, label=f"Selected threshold: {threshold:.4f}")
        
        # Add reference lines
        q1, q3 = np.percentile(cvs, [25, 75])
        iqr = q3 - q1
        
        # Upper fence
        upper_fence = q3 + 1.5 * iqr
        ax.axvline(upper_fence, color='#d62728', linestyle='--',
                  label=f"Upper fence: {upper_fence:.4f}")
        
        # Q1
        ax.axvline(q1, color='#9467bd', linestyle='--',
                  label=f"Q1: {q1:.4f}")
        
        # Style plot
        ax.set_title(f"Coefficient of Variation Distribution - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("Coefficient of Variation", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save
        plt.tight_layout()
        plt.savefig(cv_dir / "cv_distribution.png", dpi=300)
        plt.close(fig)
        
        # Plot CV vs Mean coefficient scatter
        self._plot_cv_vs_mean(method, cv_dir, features, cvs, bootstrap_stats, q1, upper_fence)
    
    def _plot_cv_vs_mean(self, method: BootstrapSelector, cv_dir: Path, 
                       features: List[str], cvs: List[float],
                       bootstrap_stats: Dict[str, FeatureStats], 
                       q1: float, upper_fence: float) -> None:
        """Plot CV vs Mean coefficient scatter."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        mean_values = [abs(bootstrap_stats[f].mean) for f in features]
        df = pd.DataFrame({
            'Feature': features,
            'Mean Coefficient': mean_values,
            'CV': cvs
        })
        
        # Plot scatter
        sns.scatterplot(x='Mean Coefficient', y='CV', data=df, alpha=0.7, ax=ax)
        
        # Add threshold line if available
        if method.threshold_info and 'value' in method.threshold_info:
            threshold = method.threshold_info['value']
            ax.axhline(threshold, color='red', linestyle='--',
                      linewidth=2, label=f"Selected threshold: {threshold:.4f}")
        
        # Add reference lines
        ax.axhline(upper_fence, color='#d62728', linestyle='--',
                  label=f"Upper fence: {upper_fence:.4f}")
        ax.axhline(q1, color='#9467bd', linestyle='--',
                  label=f"Q1: {q1:.4f}")
        
        # Label important points
        mean_threshold = np.percentile(mean_values, 90)
        cv_threshold = np.percentile(cvs, 10)
        
        for _, row in df.iterrows():
            if row['Mean Coefficient'] > mean_threshold or row['CV'] < cv_threshold:
                ax.text(row['Mean Coefficient'], row['CV'], row['Feature'],
                       fontsize=8, ha='left', va='bottom')
        
        # Style plot
        ax.set_title(f"CV vs Mean Coefficient - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("Absolute Mean Coefficient", fontsize=12)
        ax.set_ylabel("Coefficient of Variation", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save
        plt.tight_layout()
        plt.savefig(cv_dir / "cv_vs_mean.png", dpi=300)
        plt.close(fig)
    
    def plot_ci_analysis(self, method: BootstrapSelector, 
                       bootstrap_stats: Dict[str, FeatureStats]) -> None:
        """Plot confidence interval analysis."""
        logger.info(f"Plotting CI analysis for {method.get_display_name()}")
        
        # Create directory
        ci_dir = self.dirs['filtering'] / "ci" / method.get_path()
        ci_dir.mkdir(exist_ok=True, parents=True)
        
        # Get selected and filtered features
        selected = set(method.features)
        filtered = method.filtering_history.get('ci', {}).get('filtered', {})
        
        # Combine some selected and filtered features for visualization
        vis_features = list(selected)
        # Add some filtered features (up to 20 total)
        if len(vis_features) < 20:
            vis_features.extend(list(filtered.keys())[:20-len(vis_features)])
        
        # Create data for plotting
        data = []
        
        for feature in vis_features:
            if feature not in bootstrap_stats:
                continue
                
            stats = bootstrap_stats[feature]
            data.append({
                'feature': feature,
                'mean': stats.mean,
                'lower_ci': stats.lower_ci,
                'upper_ci': stats.upper_ci,
                'contains_zero': stats.ci_contains_zero,
                'selected': feature in selected
            })
        
        if not data:
            logger.warning(f"No CI data for {method.get_display_name()}")
            return
        
        # Convert to DataFrame and sort by mean
        df = pd.DataFrame(data)
        df = df.sort_values('mean')
        
        # Plot confidence intervals
        fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.3)))
        
        # Plot CIs
        for i, row in enumerate(df.itertuples()):
            color = '#2ecc71' if row.selected else '#e74c3c'
            ax.plot([row.lower_ci, row.upper_ci], [i, i], color=color, linewidth=2)
            ax.scatter(row.mean, i, color=color, s=50, zorder=5)
        
        # Add zero line
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        # Style plot
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'])
        ax.set_title(f"Feature Confidence Intervals - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("Coefficient Value", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2ecc71', marker='o', lw=2, label='Selected'),
            Line2D([0], [0], color='#e74c3c', marker='o', lw=2, label='Filtered')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Save
        plt.tight_layout()
        plt.savefig(ci_dir / "confidence_intervals.png", dpi=300)
        plt.close(fig)
    
    def plot_p_value_distribution(self, method: BootstrapSelector, 
                                bootstrap_stats: Dict[str, FeatureStats]) -> None:
        """Plot p-value distribution from bootstrap sampling."""
        logger.info(f"Plotting p-value distribution for {method.get_display_name()}")
        
        # Get p-values
        p_values = [stats.p_value for stats in bootstrap_stats.values()]
        features = list(bootstrap_stats.keys())
        
        if not p_values:
            logger.warning(f"No p-values for {method.get_display_name()}")
            return
        
        # Create directory
        pval_dir = self.dirs['filtering'] / "pvalue" / method.get_path()
        pval_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(p_values, bins=30, kde=True, ax=ax, color='skyblue',
                    edgecolor='black', alpha=0.7)
        
        # Add reference line at alpha=0.05
        ax.axvline(0.05, color='red', linestyle='--',
                  linewidth=2, label=f"α = 0.05")
        
        # Count significant features
        sig_count = sum(p < 0.05 for p in p_values)
        ax.text(0.5, 0.9, f"{sig_count} features ({sig_count/len(p_values)*100:.1f}%) have p < 0.05",
               transform=ax.transAxes, ha='center', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Style plot
        ax.set_title(f"P-Value Distribution - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("P-Value", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save
        plt.tight_layout()
        plt.savefig(pval_dir / "pvalue_distribution.png", dpi=300)
        plt.close(fig)
        
        # Plot p-value vs mean coefficient
        self._plot_pvalue_vs_mean(method, pval_dir, features, p_values, bootstrap_stats)
    
    def _plot_pvalue_vs_mean(self, method: BootstrapSelector, pval_dir: Path,
                          features: List[str], p_values: List[float],
                          bootstrap_stats: Dict[str, FeatureStats]) -> None:
        """Plot p-value vs mean coefficient."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        mean_values = [abs(bootstrap_stats[f].mean) for f in features]
        df = pd.DataFrame({
            'Feature': features,
            'Mean Coefficient': mean_values,
            'P-Value': p_values,
            'Significant': [p < 0.05 for p in p_values]
        })
        
        # Plot scatter
        sns.scatterplot(x='Mean Coefficient', y='P-Value', hue='Significant',
                    data=df, palette={True: 'green', False: 'gray'}, alpha=0.7, ax=ax)
        
        # Add reference line
        ax.axhline(0.05, color='red', linestyle='--',
                  linewidth=2, label=f"α = 0.05")
        
        # Style plot
        ax.set_title(f"P-Value vs Mean Coefficient - {method.get_display_name()}", fontsize=14)
        ax.set_xlabel("Absolute Mean Coefficient", fontsize=12)
        ax.set_ylabel("P-Value", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, min(1.0, max(p_values) * 1.1))
        
        # Save
        plt.tight_layout()
        plt.savefig(pval_dir / "pvalue_vs_mean.png", dpi=300)
        plt.close(fig)
    
    def _get_base_method(self, method_id: str, methods: Dict[str, FeatureSelector]) -> Optional[FeatureSelector]:
        """Get the base method for a method ID."""
        # For variants with ordering suffix (e.g. "_coef", "_auc")
        for suffix in ["_coef", "_auc"]:
            if method_id.endswith(suffix):
                base_id = method_id[:-len(suffix)]
                return methods.get(base_id)
        
        # For regular methods
        return methods.get(method_id)
    
    def _extract_metric_data_points(self, evaluations: List[Dict[str, Any]],
                                  dataset: str, metric: str,
                                  is_nested: bool) -> List[Tuple[int, float]]:
        """Extract data points for a specific metric."""
        data_points = []
        
        for e in evaluations:
            if is_nested:
                if ("dataset_results" in e and dataset in e["dataset_results"] and 
                    metric in e["dataset_results"][dataset]):
                    data_points.append((
                        e["num_features"], 
                        e["dataset_results"][dataset][metric]
                    ))
            else:
                result_key = f"{dataset}_results"
                if result_key in e and metric in e[result_key]:
                    data_points.append((
                        e["num_features"], 
                        e[result_key][metric]
                    ))
        
        return data_points
    
    def _get_metric_display_name(self, metric: str) -> str:
        """Get display name for a metric."""
        metric_display = {
            "auc": "AUC",
            "cohens_d": "Cohen's d",
            "js_divergence": "Jensen-Shannon Divergence",
            "overlap_coefficient": "Overlap Coefficient",
            "mutual_information": "Mutual Information",
            "mean_difference": "Mean Difference"
        }
        return metric_display.get(metric, metric)


class Phase3Pipeline:
    """Main pipeline for Phase 3 feature selection and evaluation."""
    
    def __init__(self, 
                phase1_dir: str,
                output_dir: str,
                training_datasets: List[str] = ["claude"],
                evaluation_datasets: List[str] = None,
                test_size: float = 0.2,
                normalization: str = "z_score",
                cv: int = 10,
                max_features: Optional[int] = None,
                elasticnet_threshold: float = 1e-3,
                bootstrap_samples: int = 1000,
                random_state: int = 42,
                n_jobs: int = -1,
                correlation_threshold: float = 0.7,
                use_all_data_for_correlation: bool = False,
                filter_correlation_by_group: bool = False):
        """Initialize pipeline with configuration."""
        # Directories
        self.phase1_dir = Path(phase1_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {subdir: self.output_dir / subdir for subdir in OUTPUT_SUBDIRS}
        for path in self.subdirs.values():
            path.mkdir(exist_ok=True)
        
        # Configuration parameters
        self.training_datasets = training_datasets
        self.evaluation_datasets = evaluation_datasets or []
        self.test_size = test_size
        self.normalization = normalization
        self.cv = cv
        self.max_features = max_features
        self.elasticnet_threshold = elasticnet_threshold
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.correlation_threshold = correlation_threshold
        self.use_all_data_for_correlation = use_all_data_for_correlation
        self.filter_correlation_by_group = filter_correlation_by_group
        
        # Data containers
        self.df = None
        self.X_all = None  # For correlation analysis
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.external_datasets = {}
        
        # Results
        self.feature_aucs = {}
        self.correlation_info = {}
        self.methods = {}
        self.bootstrap_stats = {}
        self.ordered_variants = {}
        self.evaluation_results = {}
        self.optimal_counts = {}
        self.best_methods = {}
    
    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline."""
        logger.info("Starting Phase 3 pipeline")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Calculate feature AUCs
        self.calculate_feature_aucs()
        
        # Step 3: Filter correlated features
        self.filter_correlated_features()
        
        # Step 4: Run feature selection methods
        self.run_feature_selection()
        
        # Step 5: Create ordered variants
        self.create_ordered_variants()
        
        # Step 6: Evaluate feature sets
        self.evaluate_feature_sets()
        
        # Step 7: Find best methods
        self.find_best_methods()
        
        # Step 8: Visualize results
        self.visualize_results()
        
        # Step 9: Save results
        self.save_results()
        
        logger.info("Pipeline completed successfully")
        return self.create_summary()
    
    def load_data(self) -> None:
        """Load data from phase 1 outputs."""
        logger.info("Starting data loading")
        
        # Find overlapping datasets
        overlapping = set(self.training_datasets) & set(self.evaluation_datasets)
        logger.info(f"Overlapping datasets: {list(overlapping)}")
        
        # Load training datasets
        train_dfs = []
        all_dfs = []
        
        for dataset in self.training_datasets:
            df = self._load_dataset(dataset, for_training=True, is_overlapping=(dataset in overlapping))
            if df is not None:
                train_dfs.append(df)
                if self.use_all_data_for_correlation:
                    all_dfs.append(df)
        
        # Load additional datasets for correlation analysis if requested
        if self.use_all_data_for_correlation:
            for dataset in self.evaluation_datasets:
                if dataset not in overlapping:
                    df = self._load_dataset(dataset, for_correlation=True)
                    if df is not None:
                        all_dfs.append(df)
        
        # Load evaluation datasets
        for dataset in self.evaluation_datasets:
            self._load_dataset(dataset, for_training=False, is_overlapping=(dataset in overlapping))
        
        # Validate data
        if not train_dfs:
            raise ValueError("No valid training datasets found")
            
        # Concatenate training data
        self.df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"Loaded training data: {len(self.df)} samples")
        
        # Prepare training data
        self._prepare_training_data()
        
        # Create combined dataset for correlation if requested
        if self.use_all_data_for_correlation and all_dfs:
            self._prepare_correlation_data(all_dfs)
        
        logger.info(f"Loaded {len(self.external_datasets)} external datasets")
    
    def _load_dataset(self, dataset_name: str, for_training: bool = False, 
                    is_overlapping: bool = False, for_correlation: bool = False) -> Optional[pd.DataFrame]:
        """Load a single dataset, handling overlaps appropriately."""
        path = self.phase1_dir / dataset_name / "metrics_df.csv"
        
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            return None
        
        df = pd.read_csv(path)
        
        if "text_type" not in df.columns:
            logger.warning(f"Missing 'text_type' column in {dataset_name}")
            return None
        
        # For overlapping datasets, split into train/test portions
        if is_overlapping:
            return self._handle_overlapping_dataset(df, dataset_name, for_training)
        
        # For non-overlapping datasets
        return self._handle_non_overlapping_dataset(df, dataset_name, for_training, for_correlation)
    
    def _handle_overlapping_dataset(self, df: pd.DataFrame, dataset_name: str, 
                                  for_training: bool) -> Optional[pd.DataFrame]:
        """Handle overlapping datasets by splitting into train/test."""
        X_train, X_test, y_train, y_test = train_test_split(
            df,
            df["text_type"] == "expert",
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["text_type"]
        )
        
        if for_training:
            return X_train
        
        # Store test portion for evaluation
        y_test = y_test.astype(int)
        self.external_datasets[dataset_name] = (X_test, y_test)
        return None
    
    def _handle_non_overlapping_dataset(self, df: pd.DataFrame, dataset_name: str,
                                      for_training: bool, for_correlation: bool) -> Optional[pd.DataFrame]:
        """Handle non-overlapping datasets."""
        if for_training or for_correlation:
            return df
        
        # For evaluation datasets
        y = (df["text_type"] == "expert").astype(int)
        self.external_datasets[dataset_name] = (df, y)
        return None
    
    def _prepare_training_data(self) -> None:
        """Prepare training data for feature selection."""
        # Create target variable
        y = (self.df["text_type"] == "expert").astype(int)
        
        # Get features excluding specific columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in EXCLUDED_COLUMNS]
        logger.info(f"Using {len(feature_cols)} numeric features")
        
        # Extract feature matrix
        X = self.df[feature_cols]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Data split: {len(self.X_train)} training, {len(self.X_test)} test samples")
    
    def _prepare_correlation_data(self, all_dfs: List[pd.DataFrame]) -> None:
        """Prepare data for correlation analysis."""
        all_data_df = pd.concat(all_dfs, ignore_index=True)
        
        # Get numeric features that are also in training data
        numeric_cols = all_data_df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in EXCLUDED_COLUMNS]
        common_cols = [col for col in feature_cols if col in self.X_train.columns]
        
        self.X_all = all_data_df[common_cols]
        logger.info(f"Created combined dataset for correlation: {len(self.X_all)} samples, {len(common_cols)} features")
    
    def calculate_feature_aucs(self) -> None:
        """Calculate AUC scores for individual features."""
        processor = FeatureProcessor(
            random_state=self.random_state,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        
        self.feature_aucs = processor.calculate_feature_aucs(self.X_train, self.y_train)
    
    def filter_correlated_features(self) -> None:
        """Filter highly correlated features."""
        processor = FeatureProcessor(
            random_state=self.random_state,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        
        # Choose dataset for correlation analysis
        X_for_correlation = self.X_all if self.use_all_data_for_correlation else self.X_train
        
        if self.use_all_data_for_correlation and self.X_all is not None:
            logger.info(f"Using combined dataset ({len(X_for_correlation)}) samples for correlation filtering")
        
        # Select filtering method based on configuration
        if self.filter_correlation_by_group:
            self._filter_correlated_by_group(processor, X_for_correlation)
        else:
            self._filter_correlated_standard(processor, X_for_correlation)
        
        # Update datasets to remove filtered features
        self._update_datasets_after_filtering()
    
    def _filter_correlated_by_group(self, processor: FeatureProcessor, 
                                  X_for_correlation: pd.DataFrame) -> None:
        """Filter correlated features by group."""
        feature_groups = self._group_features_by_prefix(X_for_correlation)
        logger.info(f"Filtering correlations within {len(feature_groups)} feature groups")
        
        all_filtered_pairs = []
        all_dropped_features = []
        original_count = X_for_correlation.shape[1]
        
        # Process each group separately
        for group_name, group_cols in feature_groups.items():
            if len(group_cols) <= 1:
                continue  # Skip groups with only one feature
                
            logger.info(f"Processing group '{group_name}' with {len(group_cols)} features")
            
            # Filter within this group
            X_group = X_for_correlation[group_cols]
            group_aucs = {col: self.feature_aucs.get(col, 0) for col in group_cols}
            
            _, group_info = processor.filter_correlated_features(
                X_group, group_aucs, self.correlation_threshold
            )
            
            # Collect filtered pairs and dropped features
            all_filtered_pairs.extend(group_info.get('pairs', []))
            all_dropped_features.extend(group_info.get('dropped', []))
        
        # Create combined info dictionary
        self.correlation_info = {
            "threshold": self.correlation_threshold,
            "pairs": all_filtered_pairs,
            "dropped": all_dropped_features,
            "count": len(all_dropped_features),
            "original_count": original_count,
            "remaining_count": original_count - len(all_dropped_features),
            "by_group": True
        }
    
    def _filter_correlated_standard(self, processor: FeatureProcessor, 
                                  X_for_correlation: pd.DataFrame) -> None:
        """Filter correlated features across all features."""
        _, self.correlation_info = processor.filter_correlated_features(
            X_for_correlation, self.feature_aucs, self.correlation_threshold
        )
    
    def _group_features_by_prefix(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by their prefix (text before first /)."""
        groups = defaultdict(list)
        
        for col in X.columns:
            # Extract prefix (text before first /) or use 'no_prefix' if no '/' exists
            prefix = col.split("/")[0] if "/" in col else "no_prefix"
            groups[prefix].append(col)
        
        return dict(groups)  # Convert back to regular dict before returning
    
    def _update_datasets_after_filtering(self) -> None:
        """Update datasets to remove filtered features."""
        all_dropped_features = self.correlation_info.get('dropped', [])
        
        # Update training and test data
        columns_to_keep = [col for col in self.X_train.columns if col not in all_dropped_features]
        self.X_train = self.X_train[columns_to_keep]
        self.X_test = self.X_test[columns_to_keep]
        
        # Update external datasets
        self._update_external_datasets(columns_to_keep)
        
        logger.info(f"Features reduced from {self.correlation_info['original_count']} to {self.correlation_info['remaining_count']}")
    
    def _update_external_datasets(self, columns_to_keep: List[str]) -> None:
        """Update external datasets with filtered feature set."""
        for dataset, (X, y) in list(self.external_datasets.items()):
            common_cols = [col for col in columns_to_keep if col in X.columns]
            if len(common_cols) < 10:
                logger.warning(f"Dataset {dataset} has too few common features, removing")
                del self.external_datasets[dataset]
                continue
                
            self.external_datasets[dataset] = (X[common_cols], y)
    
    def run_feature_selection(self) -> None:
        """Run all feature selection methods."""
        logger.info("Running feature selection methods")
        
        # Run standalone methods
        self._run_lasso()
        self._run_elasticnet()
        
        # Run bootstrap methods
        self._run_bootstrap_methods()
    
    def _run_lasso(self) -> None:
        """Run LASSO feature selection."""
        lasso = LassoSelector(
            X=self.X_train,
            y=self.y_train,
            threshold=self.elasticnet_threshold,
            max_features=self.max_features,
            random_state=self.random_state,
            cv=self.cv,
            n_jobs=self.n_jobs,
            normalization=self.normalization
        )
        
        lasso.select_features()
        self.methods["lasso"] = lasso
    
    def _run_elasticnet(self) -> None:
        """Run ElasticNet feature selection."""
        elasticnet = ElasticNetSelector(
            X=self.X_train,
            y=self.y_train,
            threshold=self.elasticnet_threshold,
            max_features=self.max_features,
            random_state=self.random_state,
            cv=self.cv,
            n_jobs=self.n_jobs,
            normalization=self.normalization
        )
        
        elasticnet.select_features()
        self.methods["elasticnet"] = elasticnet
    
    def _run_bootstrap_methods(self) -> None:
        """Run bootstrap feature selection methods with variants."""
        # Parameters from standalone methods
        lasso_params = self.methods["lasso"].params if "lasso" in self.methods else {}
        elasticnet_params = self.methods["elasticnet"].params if "elasticnet" in self.methods else {}
        
        # Process each regularization
        for regularization in REGULARIZATION_TYPES:
            logger.info(f"Creating bootstrap method for {regularization}")
            
            # Create bootstrap base method
            bootstrap = BootstrapSelector(
                X=self.X_train,
                y=self.y_train,
                regularization=regularization,
                bootstrap_samples=self.bootstrap_samples,
                max_features=self.max_features,
                random_state=self.random_state,
                cv=self.cv,
                n_jobs=self.n_jobs,
                normalization=self.normalization
            )
            
            # Set parameters from standalone methods
            bootstrap.set_regularization_params(lasso_params, elasticnet_params)
            
            # Run bootstrap iterations once per regularization
            bootstrap.run_bootstrap_iterations()
            
            # Store bootstrap stats
            self.bootstrap_stats[regularization] = bootstrap.bootstrap_stats
            
            # Create all variants
            for variant in BOOTSTRAP_VARIANTS:
                for selection in SELECTION_METHODS:
                    key = f"bootstrap_{regularization}_{variant}_{selection}"
                    logger.info(f"Creating variant {key}")
                    
                    variant_method = BootstrapSelector(
                        X=self.X_train,
                        y=self.y_train,
                        regularization=regularization,
                        bootstrap_samples=self.bootstrap_samples,
                        max_features=self.max_features,
                        random_state=self.random_state,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                        normalization=self.normalization,
                        variant=variant,
                        selection=selection
                    )
                    
                    # Copy bootstrap stats
                    variant_method.bootstrap_stats = bootstrap.bootstrap_stats
                    variant_method.standalone_params = bootstrap.standalone_params
                    
                    # Select features for this variant
                    variant_method.select_features()
                    
                    # Store method
                    self.methods[key] = variant_method
    
    def create_ordered_variants(self) -> None:
        """Create feature set variants with different orderings."""
        logger.info("Creating ordered variants")
        
        orderer = FeatureOrderer(self.feature_aucs)
        
        for method_id, method in self.methods.items():
            self.ordered_variants[method_id] = orderer.create_ordered_variants(method_id, method)
    
    def evaluate_feature_sets(self) -> None:
        """Evaluate all feature sets."""
        logger.info("Evaluating feature sets")
        
        # Create evaluator
        evaluator = ModelEvaluator(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            external_datasets=self.external_datasets,
            cv=self.cv,
            random_state=self.random_state,
            normalization=self.normalization,
            n_jobs=self.n_jobs
        )
        
        # Evaluate ordered variants for each method
        for method_id, method in self.methods.items():
            logger.info(f"Evaluating {method.get_display_name()}")
            
            for ordering_type in ORDERING_TYPES:
                ordering_id = f"{method_id}_{ordering_type}"
                if ordering_id in self.ordered_variants[method_id]:
                    ordered_features = self.ordered_variants[method_id][ordering_id]["features"]
                    evaluator.evaluate_sequential(ordering_id, ordered_features, method)
        
        # Find optimal feature counts
        evaluator.find_optimal_feature_counts()
        
        # Store results
        self.evaluation_results = evaluator.results
        self.optimal_counts = evaluator.optimal_counts
    
    def find_best_methods(self) -> None:
        """Find best method for each metric."""
        logger.info("Finding best methods for each metric")
        
        self.best_methods = {
            "test": {metric: {} for metric in METRICS},
            "overall": {metric: {} for metric in METRICS},
            "datasets": {}
        }
        
        # Process test metrics
        self._find_best_for_dataset("test")
        
        # Process overall metrics
        self._find_best_for_dataset("overall")
        
        # Process external datasets
        for dataset in self.external_datasets:
            if dataset not in self.best_methods["datasets"]:
                self.best_methods["datasets"][dataset] = {metric: {} for metric in METRICS}
            
            self._find_best_for_dataset(dataset, is_external=True)
    
    def _find_best_for_dataset(self, dataset_name: str, is_external: bool = False) -> None:
        """Find best method for each metric in a dataset."""
        target = self.best_methods["datasets"][dataset_name] if is_external else self.best_methods[dataset_name]
        
        for metric in METRICS:
            self._find_best_for_metric(dataset_name, metric, target, is_external)
    
    def _find_best_for_metric(self, dataset_name: str, metric: str, 
                            target: Dict[str, Dict[str, Any]], is_external: bool) -> None:
        """Find best method for a specific metric in a dataset."""
        # Find best method
        best_method = None
        best_value = -float('inf')
        best_count = 0
        
        for method_id, optima in self.optimal_counts.items():
            # Get metric value and count
            value, count = self._get_metric_value_and_count(
                method_id, optima, dataset_name, metric, is_external
            )
            
            if value is None or count is None:
                continue
            
            # For some metrics, we want maximum absolute value
            value_to_compare = abs(value) if metric == "mean_difference" else value
            
            if value_to_compare > best_value:
                best_value = value_to_compare
                best_method = method_id
                best_count = count
        
        # Store best method
        if best_method:
            method_obj = self._get_base_method_for_id(best_method)
            if method_obj:
                # Extract ordering type if present
                ordering = self._extract_ordering_type(best_method)
                
                target[metric] = {
                    "method_id": best_method,
                    "method_name": method_obj.get_display_name(),
                    "feature_count": best_count,
                    "value": best_value,
                    "ordering": ordering
                }
    
    def _get_metric_value_and_count(self, method_id: str, optima: Dict[str, Any],
                                  dataset_name: str, metric: str, 
                                  is_external: bool) -> Tuple[Optional[float], Optional[int]]:
        """Get metric value and feature count for a method."""
        count_key = f"{metric}_count"
        
        if is_external:
            # Check for external dataset data
            if "datasets" not in optima or dataset_name not in optima["datasets"]:
                return None, None
            
            value = optima["datasets"][dataset_name].get(metric)
            count = optima["datasets"][dataset_name].get(count_key)
        else:
            # Check for test/overall dataset data
            if dataset_name not in optima:
                return None, None
            
            value = optima[dataset_name].get(metric)
            count = optima[dataset_name].get(count_key)
        
        return value, count
    
    def _get_base_method_for_id(self, method_id: str) -> Optional[FeatureSelector]:
        """Get the base method object for a method ID."""
        # Strip ordering suffix if present
        base_id = method_id
        for suffix in ["_coef", "_auc"]:
            if method_id.endswith(suffix):
                base_id = method_id[:-len(suffix)]
                break
                
        return self.methods.get(base_id)
    
    def _extract_ordering_type(self, method_id: str) -> Optional[str]:
        """Extract ordering type from method ID if present."""
        for suffix in ["_coef", "_auc"]:
            if method_id.endswith(suffix):
                return suffix[1:]  # Remove leading underscore
        return None
    
    def visualize_results(self) -> None:
        """Create visualizations of results."""
        visualizer = ResultsVisualizer(self.subdirs["vis"])
        visualizer.plot_all(self.methods, self.evaluation_results, self.optimal_counts)
    
    def save_results(self) -> None:
        """Save all results to files."""
        logger.info("Saving results")
        
        # Create methods directory
        methods_dir = self.subdirs["data"] / "methods"
        methods_dir.mkdir(exist_ok=True)
        
        # Save each method's details
        for method_id, method in self.methods.items():
            self._save_method_data(method_id, method, methods_dir)
        
        # Save other results
        save_json(self.optimal_counts, self.subdirs["data"] / "optimal_counts.json")
        save_json(self.best_methods, self.subdirs["data"] / "best_methods.json")
        save_json(self.correlation_info, self.subdirs["filtering"] / "correlation_info.json")
        
        # Save serializable bootstrap stats
        bootstrap_stats_serializable = self._prepare_bootstrap_stats_for_serialization()
        save_json(bootstrap_stats_serializable, self.subdirs["data"] / "bootstrap_stats.json")
        
        # Save plot data separately for replotting
        plot_data = {
            "methods": {method_id: method.to_dict() for method_id, method in self.methods.items()},
            "ordered_variants": self.ordered_variants,
            "evaluation_results": self.evaluation_results,
            "optimal_counts": self.optimal_counts,
            "correlation_info": self.correlation_info
        }
        
        plot_data_path = self.subdirs["data"] / "plot_data.joblib"
        joblib.dump(plot_data, plot_data_path, compress=3)
        logger.info(f"Saved plot data to {plot_data_path}")
    
    def _save_method_data(self, method_id: str, method: FeatureSelector, methods_dir: Path) -> None:
        """Save data for a single method."""
        # Create method directory
        method_dir = methods_dir / method.get_path()
        method_dir.mkdir(exist_ok=True, parents=True)
        
        # Save method details
        method_data = method.to_dict()
        save_json(method_data, method_dir / "details.json")
        
        # Save features
        self._save_feature_list(method, method_dir)
        
        # Save evaluation results if available
        self._save_evaluation_results(method_id, method, method_dir)
    
    def _save_feature_list(self, method: FeatureSelector, method_dir: Path) -> None:
        """Save list of features with importances."""
        pd.DataFrame({
            "feature": method.features,
            "importance": [method.importances.get(f, 0) for f in method.features]
        }).to_csv(method_dir / "features.csv", index=False)
    
    def _save_evaluation_results(self, method_id: str, method: FeatureSelector, method_dir: Path) -> None:
        """Save evaluation results for a method."""
        for variant_id, variant in self.ordered_variants.get(method_id, {}).items():
            if variant_id not in self.evaluation_results:
                continue
                
            # Convert evaluation data to DataFrame
            df = self._create_evaluation_dataframe(self.evaluation_results[variant_id])
            
            # Save to CSV
            variant_dir = method_dir / "variants"
            variant_dir.mkdir(exist_ok=True)
            df.to_csv(variant_dir / f"{variant_id}.csv", index=False)
    
    def _create_evaluation_dataframe(self, evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame from evaluation results."""
        rows = []
        for r in evaluations:
            row = {
                "num_features": r["num_features"],
                "added_feature": r["added_feature"],
                "cv_auc": r["cv_results"]["mean_auc"],
                "cv_std": r["cv_results"]["std_auc"]
            }
            
            # Add test metrics
            if "test_results" in r:
                for metric, value in r["test_results"].items():
                    row[f"test_{metric}"] = value
            
            # Add overall metrics
            if "overall_results" in r:
                for metric, value in r["overall_results"].items():
                    row[f"overall_{metric}"] = value
            
            # Add dataset metrics
            if "dataset_results" in r:
                for dataset, results in r["dataset_results"].items():
                    for metric, value in results.items():
                        row[f"{dataset}_{metric}"] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _prepare_bootstrap_stats_for_serialization(self) -> Dict[str, Any]:
        """Prepare bootstrap stats for JSON serialization."""
        bootstrap_stats_serializable = {}
        for reg, stats in self.bootstrap_stats.items():
            bootstrap_stats_serializable[reg] = {}
            for feature, feature_stats in stats.items():
                bootstrap_stats_serializable[reg][feature] = feature_stats.to_dict()
        
        return bootstrap_stats_serializable
    
    def create_summary(self) -> Dict[str, Any]:
        """Create a summary of pipeline results."""
        summary = {
            "datasets": {
                "training": self.training_datasets,
                "evaluation": list(self.external_datasets.keys()),
                "samples": {
                    "train": len(self.X_train),
                    "test": len(self.X_test)
                },
                "features": {
                    "original": self.correlation_info.get("original_count", 0),
                    "after_correlation": self.correlation_info.get("remaining_count", 0)
                }
            },
            "methods": {
                method_id: {
                    "name": method.get_display_name(),
                    "feature_count": len(method.features),
                    "type": method.__class__.__name__
                }
                for method_id, method in self.methods.items()
            },
            "best_methods": {
                "test": {
                    metric: info.get("method_name", "")
                    for metric, info in self.best_methods["test"].items() if info
                },
                "overall": {
                    metric: info.get("method_name", "")
                    for metric, info in self.best_methods["overall"].items() if info
                }
            },
            "parameters": {
                "correlation_threshold": self.correlation_threshold,
                "elasticnet_threshold": self.elasticnet_threshold,
                "max_features": self.max_features,
                "bootstrap_samples": self.bootstrap_samples,
                "cv": self.cv,
                "random_state": self.random_state,
                "use_all_data_for_correlation": self.use_all_data_for_correlation,
                "filter_correlation_by_group": self.filter_correlation_by_group
            }
        }
        
        # Save summary
        save_json(summary, self.subdirs["data"] / "summary.json")
        
        return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a summary of pipeline results."""
    print("\n===== Pipeline Summary =====")
    print(f"Training datasets: {', '.join(summary['datasets']['training'])}")
    print(f"Evaluation datasets: {', '.join(summary['datasets']['evaluation'])}")
    print(f"Samples: {summary['datasets']['samples']['train']} train, {summary['datasets']['samples']['test']} test")
    print(f"Features: {summary['datasets']['features']['original']} original, {summary['datasets']['features']['after_correlation']} after correlation filtering")
    
    print("\nMethods:")
    for method_id, info in summary["methods"].items():
        if "_coef" not in method_id and "_auc" not in method_id:
            print(f"  {info['name']}: {info['feature_count']} features")
    
    print("\nBest methods by metric:")
    print("  Test set:")
    for metric, method in summary["best_methods"]["test"].items():
        print(f"    {metric}: {method}")
    
    print("  Overall (combined datasets):")
    for metric, method in summary["best_methods"]["overall"].items():
        print(f"    {metric}: {method}")


def main() -> None:
    """Run the Phase 3 pipeline with command-line arguments."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Phase 3: Feature Selection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--phase1-dir",
        type=str,
        required=True,
        help="Path to Phase 1 output directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--training-datasets",
        type=str,
        nargs="+",
        default=["claude"],
        help="List of training dataset names"
    )
    
    parser.add_argument(
        "--evaluation-datasets",
        type=str,
        nargs="+",
        default=[],
        help="List of evaluation dataset names"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    
    parser.add_argument(
        "--normalization",
        type=str,
        default="z_score",
        choices=["z_score", "min_max", "robust", "none"],
        help="Normalization method"
    )
    
    parser.add_argument(
        "--cv",
        type=int,
        default=10,
        help="Number of cross-validation folds"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to select"
    )
    
    parser.add_argument(
        "--elasticnet-threshold",
        type=float,
        default=1e-3,
        help="Threshold for coefficient magnitude"
    )
    
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.7,
        help="Threshold for filtering correlated features"
    )
    
    parser.add_argument(
        "--use-all-data-for-correlation",
        action="store_true",
        help="Use all data (training and evaluation) for correlation filtering"
    )
    
    parser.add_argument(
        "--filter-correlation-by-group",
        action="store_true",
        help="Filter correlated features within prefix groups (e.g., 'gispy/', 'textstat/')"
    )

    args = parser.parse_args()
    
    # Run pipeline
    pipeline = Phase3Pipeline(
        phase1_dir=args.phase1_dir,
        output_dir=args.output_dir,
        training_datasets=args.training_datasets,
        evaluation_datasets=args.evaluation_datasets,
        test_size=args.test_size,
        normalization=args.normalization,
        cv=args.cv,
        max_features=args.max_features,
        elasticnet_threshold=args.elasticnet_threshold,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        correlation_threshold=args.correlation_threshold,
        use_all_data_for_correlation=args.use_all_data_for_correlation,
        filter_correlation_by_group=args.filter_correlation_by_group
    )
    
    # Run pipeline
    summary = pipeline.run()
    
    # Print summary
    print_summary(summary)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()