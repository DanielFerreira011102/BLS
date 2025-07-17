import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

from utils.helpers import load_json, setup_logging

# Initialize logging
logger = setup_logging()

# Define the list of features to keep
REQUIRED_FEATURES = [
    "jargon/roberta-large/abbr_general_density",
    "llm/mse_1.0+kl_0.0+ce_0.0+soft_0.0/dimension_scores/vocabulary_complexity",
    "textstat/dale_chall",
    "textstat/type_token_ratio",
    "commonlit/albert",
    "jargon/roberta-large/jargon_count",
    "scigispy/PCREF_chunk_1p",
    "scigispy/WRDIC",
    "syntax/en_core_web_trf/embedding_depth",
    "syntax/en_core_web_trf/verb_ratio",
    "syntax/en_core_web_trf/function_ratio",
    "cluster/medreadme",
    "masked_prob/random/Bio_ClinicalBERT",
    "syntax/en_core_web_trf/avg_dependency_distance",
    "umls/en_core_web_trf/avg_cui_score",
    "gispy/CoREF"
]


class DataValidator:
    """Validates and processes metric data from text samples."""

    def __init__(self, data: Dict) -> None:
        """Initializes the validator with input data."""
        self.data = data
        self.samples = []
        self.dataset_info = {}  # Store dataset information

    def validate_and_extract(self) -> List:
        """Validates and extracts samples from the data."""
        logger.info("Validating and extracting samples")
        
        if not self.data:
            logger.error("No data to validate")
            return []
        
        # Extract samples from the nested structure (multiple datasets)
        for dataset_name, dataset_info in self.data.items():
            if "samples" not in dataset_info:
                logger.warning(f"No samples found in dataset {dataset_name}")
                continue
                
            # Store dataset info for later use
            self.dataset_info[dataset_name] = {
                "dataset": dataset_info.get("dataset"),
                "config": dataset_info.get("config")
            }
            
            # Process samples from this dataset
            dataset_samples = []
            for sample in dataset_info["samples"]:
                if not self._is_valid_sample(sample):
                    continue
                
                # Add dataset name to sample for tracking
                sample["dataset_name"] = dataset_name
                dataset_samples.append(sample)
            
            logger.info(f"Extracted {len(dataset_samples)} valid samples from dataset {dataset_name}")
            self.samples.extend(dataset_samples)
            
        logger.info(f"Extracted {len(self.samples)} total valid samples across all datasets")
        return self.samples
    
    def _is_valid_sample(self, sample: Dict) -> bool:
        """Validates sample structure."""
        # Check if sample has required fields
        required_fields = ["question", "complexity_level", "model", "response"]
        for field in required_fields:
            if field not in sample:
                logger.warning(f"Sample missing required field: {field}")
                return False
            
        # Metrics should be present but can be empty
        if "metrics" not in sample:
            logger.warning("Sample missing metrics field")
            return False
            
        return True


class DataTransformer:
    """Transforms samples into structured DataFrame format."""
    
    def __init__(self, samples: List) -> None:
        """Initializes the transformer with samples."""
        self.samples = samples
        self.missing_features = set()  # Track which features are missing
    
    def transform(self) -> pd.DataFrame:
        """Converts samples to a structured DataFrame and handles infinities."""
        logger.info("Transforming samples to DataFrame")
        
        if not self.samples:
            logger.error("No samples to transform")
            return pd.DataFrame()
        
        # Get all metric paths
        metrics = REQUIRED_FEATURES
        rows = []
        
        # Create a row for each sample
        for i, sample in enumerate(self.samples):
            row = {
                "sample_id": i,
                "dataset_name": sample.get("dataset_name", "unknown"),
                "question": sample["question"],
                "complexity_level": sample["complexity_level"],
                "model": sample["model"],
                "response": sample["response"]
            }
            
            # Extract metrics values from nested structure
            for metric in metrics:
                row[metric] = self._get_nested_value(sample.get("metrics", {}), metric, sample_id=i)
            
            rows.append(row)
        
        # Create DataFrame and handle infinity values
        df = pd.DataFrame(rows)
        df = self._handle_infinity_values(df)
        
        # Log information about missing features
        if self.missing_features:
            logger.warning(f"The following features were missing in some or all samples: {', '.join(self.missing_features)}")
            logger.info(f"These columns will be kept with NaN values")
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(metrics)} metrics")
        return df
    
    def _get_nested_value(self, metrics: Dict, feature_path: str, sample_id: int = None) -> float:
        """Gets value from a nested path in the metrics dictionary."""
        value = None
        
        # Extract value based on feature path
        if feature_path == "jargon/roberta-large/abbr_general_density":
            value = metrics.get("jargon", {}).get("abbr_general_density")
            
        elif feature_path == "llm/mse_1.0+kl_0.0+ce_0.0+soft_0.0/dimension_scores/vocabulary_complexity":
            llm_metrics = metrics.get("llm", {})
            mse_metrics = llm_metrics.get("mse_1.0+kl_0.0+ce_0.0+soft_0.0", {})
            dimension_scores = mse_metrics.get("dimension_scores", {})
            value = dimension_scores.get("vocabulary_complexity")
            
        elif feature_path == "textstat/dale_chall":
            value = metrics.get("textstat", {}).get("dale_chall")
            
        elif feature_path == "textstat/type_token_ratio":
            value = metrics.get("textstat", {}).get("type_token_ratio")
            
        elif feature_path == "commonlit/albert":
            value = metrics.get("commonlit", {}).get("albert")
            
        elif feature_path == "jargon/roberta-large/jargon_count":
            value = metrics.get("jargon", {}).get("jargon_count")
            
        elif feature_path == "scigispy/PCREF_chunk_1p":
            value = metrics.get("scigispy", {}).get("PCREF_chunk_1p")
            
        elif feature_path == "scigispy/WRDIC":
            value = metrics.get("scigispy", {}).get("WRDIC")
            
        elif feature_path == "syntax/en_core_web_trf/embedding_depth":
            syntax_metrics = metrics.get("syntax", {})
            value = syntax_metrics.get("en_core_web_trf", {}).get("embedding_depth")
            
        elif feature_path == "syntax/en_core_web_trf/verb_ratio":
            syntax_metrics = metrics.get("syntax", {})
            value = syntax_metrics.get("en_core_web_trf", {}).get("verb_ratio")
            
        elif feature_path == "syntax/en_core_web_trf/function_ratio":
            syntax_metrics = metrics.get("syntax", {})
            value = syntax_metrics.get("en_core_web_trf", {}).get("function_ratio")
            
        elif feature_path == "cluster/medreadme":
            value = metrics.get("cluster", {}).get("medreadme")
            
        elif feature_path == "masked_prob/random/Bio_ClinicalBERT":
            masked_prob = metrics.get("masked_prob", {})
            value = masked_prob.get("random", {}).get("Bio_ClinicalBERT")
            
        elif feature_path == "syntax/en_core_web_trf/avg_dependency_distance":
            syntax_metrics = metrics.get("syntax", {})
            value = syntax_metrics.get("en_core_web_trf", {}).get("avg_dependency_distance")
            
        elif feature_path == "umls/en_core_web_trf/avg_cui_score":
            value = metrics.get("umls", {}).get("en_core_web_trf", {}).get("avg_cui_score")
            
        elif feature_path == "gispy/CoREF":
            value = metrics.get("gispy", {}).get("CoREF")
        
        # Log warning if feature is missing (only once per feature)
        if value is None and feature_path not in self.missing_features:
            self.missing_features.add(feature_path)
            sample_info = f" for sample_id {sample_id}" if sample_id is not None else ""
            logger.warning(f"Feature '{feature_path}' not found{sample_info}")
            
        return value

    def _handle_infinity_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinity values in the DataFrame."""
        infinity_found = False
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["sample_id", "complexity_level"]]
        
        for col in numeric_cols:
            # Skip if no infinities
            inf_mask = np.isinf(df[col])
            if not inf_mask.any():
                continue
                
            infinity_found = True
            finite_values = df[col][~inf_mask]
            
            # Replace positive infinities
            pos_inf_mask = np.isposinf(df[col])
            if pos_inf_mask.any() and len(finite_values) > 0:
                max_value = finite_values.max()
                logger.info(f"Replacing positive infinity values in '{col}' with maximum value: {max_value}")
                df.loc[pos_inf_mask, col] = max_value
            
            # Replace negative infinities
            neg_inf_mask = np.isneginf(df[col])
            if neg_inf_mask.any() and len(finite_values) > 0:
                min_value = finite_values.min()
                logger.info(f"Replacing negative infinity values in '{col}' with minimum value: {min_value}")
                df.loc[neg_inf_mask, col] = min_value
            
            # Handle case where all values are infinite
            if len(finite_values) == 0:
                logger.warning(f"All values in '{col}' are infinite, replacing with zeros")
                df.loc[inf_mask, col] = 0
        
        if infinity_found:
            logger.info("Handled infinity values during data transformation")
            
        return df

    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Checks for missing values and logs warnings."""
        if df.empty:
            return df
            
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found")
            return df
            
        logger.warning(f"Found {missing_count} missing values in the data")
        
        # Log columns with missing values
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing == 0:
                continue
                
            percentage = missing / len(df) * 100
            logger.warning(f"Column '{col}' has {missing} missing values ({percentage:.1f}%)")
            
        return df


class DataNormalizer:
    """Normalizes metrics data using different strategies."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initializes the normalizer with a DataFrame."""
        self.df = df
        self.results = {}
        self.scalers = {}  # Store scalers for later use

    def normalize(self) -> Dict[str, pd.DataFrame]:
        """Applies multiple normalization strategies."""
        logger.info("Normalizing metrics data")
        
        # Define normalization methods and corresponding scalers
        normalization_methods = {
            "z_score": StandardScaler(),
            "min_max": MinMaxScaler(),
            "robust": RobustScaler()
        }
        
        # Get columns that need normalization
        cols = self._get_columns_to_normalize()
        if not cols:
            logger.warning("No numeric columns found for normalization")
            return {}
        
        # Apply each normalization method
        for name, scaler in normalization_methods.items():
            self.results[name] = self._apply_normalization(scaler, cols, name)
            # Store the fitted scaler for later use
            self.scalers[name] = scaler
        
        return self.results
    
    def save_scalers(self, output_dir: Path) -> None:
        """Saves scalers to the output directory for future use."""
        if not self.scalers:
            logger.warning("No scalers to save")
            return
            
        scalers_dir = output_dir / "scalers"
        scalers_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each scaler
        for name, scaler in self.scalers.items():
            scaler_path = scalers_dir / f"{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {name} scaler to {scaler_path}")
    
    def _get_columns_to_normalize(self) -> List[str]:
        """Gets numeric columns that need normalization."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in ["sample_id", "complexity_level"]]
    
    def _apply_normalization(self, scaler, cols: List[str], name: str) -> pd.DataFrame:
        """Applies normalization using the specified scaler."""
        result = self.df.copy()
        
        if not cols:
            return result
            
        result[cols] = scaler.fit_transform(result[cols])
        logger.info(f"Applied {name} normalization to {len(cols)} columns")
        
        return result


class Phase3Pipeline:
    """Runs Phase 3 analysis pipeline for text complexity metrics."""

    def __init__(self, in_file: Union[str, Path], out_dir: Union[str, Path] = None) -> None:
        """Initializes the pipeline with input file and output directory."""
        self.in_file = Path(in_file)
        self.out_dir = Path(out_dir) if out_dir else Path("outputs/phase3")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # State variables
        self.data = None
        self.samples = []
        self.df = pd.DataFrame()
        self.normalized_dfs = {}
        self.normalizer = None
        self.dataset_info = {}  # Store dataset metadata
    
    def run(self) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """Executes the Phase 3 pipeline."""
        logger.info(f"Running Phase 3 pipeline on {self.in_file}")
        
        # Step 1: Load and validate data
        self._load()
        if not self.samples:
            logger.error("No samples loaded")
            return {}
        
        # Step 2: Transform data to DataFrame format    
        self._transform()
        if self.df.empty:
            logger.error("No data transformed")
            return {}

        # Step 3: Normalize data
        self._normalize()
        
        # Step 4: Save results
        self._save()
        
        return {
            "metrics_df": self.df, 
            "normalized_dfs": self.normalized_dfs
        }
    
    def _load(self) -> None:
        """Loads and validates data from input file."""
        logger.info(f"Loading data from {self.in_file}")
        
        if not self.in_file.exists():
            logger.error(f"File not found: {self.in_file}")
            raise FileNotFoundError(f"File not found: {self.in_file}")
        
        # Load JSON data - expecting multiple datasets with samples under each
        self.data = load_json(self.in_file)
        
        # Print dataset names for debugging
        logger.info(f"Found datasets: {list(self.data.keys())}")
            
        # Validate and extract samples
        validator = DataValidator(self.data)
        self.samples = validator.validate_and_extract()
        self.dataset_info = validator.dataset_info
        
        logger.info(f"Loaded and validated {len(self.samples)} valid samples across {len(self.dataset_info)} datasets")
    
    def _transform(self) -> None:
        """Transforms samples into DataFrame format."""
        # Create transformer and process samples
        transformer = DataTransformer(self.samples)
        self.df = transformer.transform()
        
        # Check for missing values
        self.df = transformer.check_missing_values(self.df)
        logger.info(f"Transformed data into DataFrame with {len(self.df)} rows")

    def _normalize(self) -> None:
        """Normalizes data with different strategies."""
        # Create normalizer and apply normalization methods
        self.normalizer = DataNormalizer(self.df)
        self.normalized_dfs = self.normalizer.normalize()
        
        if self.normalized_dfs:
            logger.info(f"Created {len(self.normalized_dfs)} normalized versions of the data")
    
    def _save(self) -> None:
        """Saves raw metrics data and normalized versions."""
        # Save raw metrics DataFrame
        metrics_path = self.out_dir / "metrics_df.csv"
        self.df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save each normalized version
        for name, df in self.normalized_dfs.items():
            norm_path = self.out_dir / f"normalized_df_{name}.csv"
            df.to_csv(norm_path, index=False)
            logger.info(f"Saved {name} normalized data to {norm_path}")
        
        # Save scalers for future use
        self.normalizer.save_scalers(self.out_dir)


def main() -> None:
    """Runs Phase 3 from CLI."""
    parser = argparse.ArgumentParser(description="Phase 3 of text complexity analysis pipeline")
    parser.add_argument("--input-file", required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", help="Directory for saving output files")
    args = parser.parse_args()
    
    pipeline = Phase3Pipeline(args.input_file, args.output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()