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


class DataValidator:
    """Validates and processes metric data from text samples."""

    def __init__(self, data: Dict) -> None:
        """Initializes the validator with input data."""
        self.data = data
        self.samples = []

    def validate_and_extract(self) -> List:
        """Validates and extracts samples from datasets."""
        logger.info("Validating and extracting samples")
        if not self.data:
            logger.error("No data to validate")
            return []
            
        # Find all valid dataset keys in the input data
        datasets = [
            key for key in self.data.keys() 
            if isinstance(self.data[key], dict) and "samples" in self.data[key]
        ]
        
        if not datasets:
            logger.error("No valid datasets found in the data file")
            raise ValueError("No valid datasets found in the data file")
        
        logger.info(f"Found {len(datasets)} datasets: {datasets}")
        
        # Process each dataset
        for dataset in datasets:
            if not self._is_valid_dataset(dataset):
                logger.warning(f"Skipping invalid dataset: {dataset}")
                continue
            
            # Add dataset name to each sample
            for sample in self.data[dataset]["samples"]:
                sample["simple"]["dataset"] = dataset
                sample["expert"]["dataset"] = dataset
            
            # Add samples to the collection
            self.samples.extend(self.data[dataset]["samples"])
            logger.info(f"Added {len(self.data[dataset]['samples'])} samples from '{dataset}'")
        
        return self.samples
    
    def _is_valid_dataset(self, dataset: str) -> bool:
        """Validates dataset structure."""
        if "samples" not in self.data[dataset] or not self.data[dataset]["samples"]:
            logger.error(f"Missing or empty 'samples' in dataset '{dataset}'")
            return False
        
        # Check if all samples in the dataset are valid
        return all(self._is_valid_sample(s, dataset) for s in self.data[dataset]["samples"])
    
    def _is_valid_sample(self, sample: Dict, dataset: str) -> bool:
        """Validates sample structure."""
        # Check if sample has both 'simple' and 'expert' text
        if not all(text_type in sample for text_type in ["simple", "expert"]):
            logger.error(f"Sample in dataset '{dataset}' missing 'simple' or 'expert' key")
            return False
        
        # Check if each text type has required fields
        for text_type in ["simple", "expert"]:
            if not all(key in sample[text_type] for key in ["text", "metrics"]):
                logger.error(f"Sample in dataset '{dataset}' missing required fields")
                return False
                
        return True


class DataTransformer:
    """Transforms samples into structured DataFrame format."""
    
    def __init__(self, samples: List) -> None:
        """Initializes the transformer with samples."""
        self.samples = samples
    
    def transform(self) -> pd.DataFrame:
        """Converts samples to a structured DataFrame and handles infinities."""
        logger.info("Transforming samples to DataFrame")
        
        if not self.samples:
            logger.error("No samples to transform")
            return pd.DataFrame()
        
        # Get all possible metric paths from the first sample
        metrics = self._get_all_metric_paths(self.samples[0])
        rows = []
        
        # Create a row for each text type in each sample
        for i, sample in enumerate(self.samples):
            for text_type in ["simple", "expert"]:
                row = {
                    "sample_id": i,
                    "text_type": text_type,
                    "dataset": sample[text_type].get("dataset", "unknown")
                }
                
                # Include original text
                row["text"] = sample[text_type]["text"]
                
                # Extract metrics values from nested structure
                for metric in metrics:
                    path = metric.split('/')
                    row[metric] = self._get_nested_value(sample[text_type]["metrics"], path)
                
                rows.append(row)
        
        # Create DataFrame and handle infinity values
        df = pd.DataFrame(rows)
        df = self._handle_infinity_values(df)
        
        logger.info(f"Created clean DataFrame with {len(df)} rows and {len(metrics)} metrics")
        return df
    
    def _get_all_metric_paths(self, sample: Dict) -> List[str]:
        """Extracts all metric paths from a sample."""
        simple_paths = []
        expert_paths = []
        
        # Ensure we extract paths from both simple and expert samples
        # since they might have different metrics
        self._extract_paths(sample["simple"]["metrics"], "", simple_paths)
        self._extract_paths(sample["expert"]["metrics"], "", expert_paths)
        
        return sorted(set(simple_paths + expert_paths))
    
    def _extract_paths(self, obj: Dict, prefix: str, paths: List) -> None:
        """Recursively extracts metric paths from nested dictionaries."""
        for key, value in obj.items():
            curr_path = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                # Continue recursion for nested dictionaries
                self._extract_paths(value, curr_path, paths)
            elif isinstance(value, (int, float)):
                # Add leaf nodes that are numeric values
                paths.append(curr_path)
    
    def _get_nested_value(self, obj: Dict, path: List[str]) -> float:
        """Gets value from a nested path in a dictionary."""
        for key in path:
            if key not in obj:
                return None
            obj = obj[key]

        # Return only numeric values
        return obj if isinstance(obj, (int, float)) else None

    def _handle_infinity_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinity values in the DataFrame during initial transformation."""
        infinity_found = False
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "sample_id"]
        
        for col in numeric_cols:
            # Check for infinities
            inf_mask = np.isinf(df[col])
            if not inf_mask.any():
                continue
                
            infinity_found = True
            finite_values = df[col][~inf_mask]
            
            # Replace positive infinities with maximum finite value
            pos_inf_mask = np.isposinf(df[col])
            if pos_inf_mask.any() and len(finite_values) > 0:
                max_value = finite_values.max()
                logger.info(f"Replacing positive infinity values in '{col}' with maximum value: {max_value}")
                df.loc[pos_inf_mask, col] = max_value
            
            # Replace negative infinities with minimum finite value
            neg_inf_mask = np.isneginf(df[col])
            if neg_inf_mask.any() and len(finite_values) > 0:
                min_value = finite_values.min()
                logger.info(f"Replacing negative infinity values in '{col}' with minimum value: {min_value}")
                df.loc[neg_inf_mask, col] = min_value
            
            # Handle case where all values in the column are infinite
            if len(finite_values) == 0:
                logger.warning(f"All values in '{col}' are infinite, replacing with zeros")
                df.loc[inf_mask, col] = 0
        
        if infinity_found:
            logger.info("Handled infinity values during data transformation")
            
        return df

    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Checks for missing values and raises error if found."""
        if df.empty:
            return df
            
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            logger.error(f"Found {missing_count} missing values in the data")
            raise ValueError(f"Dataset contains {missing_count} missing values")
            
        logger.info("No missing values found, data validation passed")
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
        
        # Apply each normalization method
        for name, scaler in normalization_methods.items():
            self.results[name] = self._apply_normalization(scaler, cols, name)
            # Store the fitted scaler for later use
            self.scalers[name] = scaler
        
        return self.results
    
    def save_scalers(self, output_dir: Path) -> None:
        """Saves scalers to the output directory for future use."""
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
        return [col for col in numeric_cols if col not in ["sample_id"]]
    
    def _apply_normalization(self, scaler, cols: List[str], name: str) -> pd.DataFrame:
        """Applies normalization using the specified scaler."""
        result = self.df.copy()
        
        if cols:
            result[cols] = scaler.fit_transform(result[cols])
            logger.info(f"Applied {name} normalization to {len(cols)} columns")
        
        return result


class Phase1Pipeline:
    """Runs Phase 1 analysis pipeline for text complexity metrics."""

    def __init__(self, in_file: Union[str, Path], out_dir: Union[str, Path] = None, ignore_features: Optional[List[str]] = None) -> None:
        """Initializes the pipeline with input file and output directory."""
        self.in_file = Path(in_file)
        self.out_dir = Path(out_dir) if out_dir else Path("outputs/phase1")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ignore_features = ignore_features or []
        
        # State variables
        self.data = None
        self.samples = []
        self.df = pd.DataFrame()
        self.normalized_dfs = {}
        self.normalizer = None
    
    def run(self) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """Executes the Phase 1 pipeline."""
        logger.info(f"Running Phase 1 pipeline on {self.in_file}")
        
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

        # Step 3: Filter and normalize data
        self._filter_ignored_features() 
        self._normalize()
        
        # Step 4: Save results
        self._save()
        
        return {"metrics_df": self.df, "normalized_dfs": self.normalized_dfs}
    
    def _load(self) -> None:
        """Loads and validates data from input file."""
        logger.info(f"Loading data from {self.in_file}")
        
        if not self.in_file.exists():
            logger.error(f"File not found: {self.in_file}")
            raise FileNotFoundError(f"File not found: {self.in_file}")
        
        # Load JSON data
        self.data = load_json(self.in_file)
            
        # Validate and extract samples
        validator = DataValidator(self.data)
        self.samples = validator.validate_and_extract()
        logger.info(f"Loaded and validated {len(self.samples)} samples")
    
    def _transform(self) -> None:
        """Transforms samples into DataFrame format."""
        # Create transformer and process samples
        transformer = DataTransformer(self.samples)
        self.df = transformer.transform()
        
        # Check for missing values
        self.df = transformer.check_missing_values(self.df)
        logger.info(f"Transformed data into DataFrame with {len(self.df)} rows")
    
    def _filter_ignored_features(self) -> None:
        """Filter out ignored features from DataFrame."""
        if not self.ignore_features:
            return
        
        # Find features to drop that exist in the DataFrame
        features_to_drop = [f for f in self.ignore_features if f in self.df.columns]
        if features_to_drop:
            self.df = self.df.drop(columns=features_to_drop, errors="ignore")
            logger.info(f"Ignored {len(features_to_drop)} features: {', '.join(features_to_drop)}")

    def _normalize(self) -> None:
        """Normalizes data with different strategies."""
        # Create normalizer and apply normalization methods
        self.normalizer = DataNormalizer(self.df)
        self.normalized_dfs = self.normalizer.normalize()
        logger.info(f"Created {len(self.normalized_dfs)} normalized versions of the data")
    
    def _save(self) -> None:
        """Saves raw and normalized data to output directory."""
        # Save raw metrics DataFrame
        metrics_path = self.out_dir / "metrics_df.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save each normalized version
        for name, df in self.normalized_dfs.items():
            norm_path = self.out_dir / f"normalized_df_{name}.csv"
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(norm_path, index=False)
            logger.info(f"Saved {name} normalized data to {norm_path}")
        
        # Save scalers for future use
        self.normalizer.save_scalers(self.out_dir)
        logger.info(f"Saved scalers to {self.out_dir / 'scalers'}")


def main() -> None:
    """Runs Phase 1 from CLI."""
    parser = argparse.ArgumentParser(description="Phase 1 of text complexity analysis pipeline")
    parser.add_argument("--input-file", required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", help="Directory for saving output files")
    parser.add_argument("--ignore-features", type=str, nargs="+", help="Features to ignore")
    args = parser.parse_args()
    
    pipeline = Phase1Pipeline(args.input_file, args.output_dir, args.ignore_features)
    pipeline.run()


if __name__ == "__main__":
    main()