import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from collections import Counter

from utils.helpers import setup_logging, load_json, save_json

# Initialize logging
logger = setup_logging()


class DataValidator:
    """Validates and cleans text samples and their variants."""

    def __init__(self, data: Dict):
        self.data = data
        self.samples = []
        self.filtered_count = Counter({
            "content_mismatch": 0,
            "empty_text": 0,
            "partial_empty": 0,
            "excluded_id": 0,
            "excluded_variant": 0,
            "artifacts_cleaned": 0,
        })
        self.flagged_samples = []

        # Precompile regex for AI artifacts
        artifacts = [
            "</assistant<|end_header_id|>",
            "<|endoftext|>",
            "<|im_end|>",
            "</assistant>",
            "<assistant>",
            "</human>",
            "<human>",
            "[AI]:",
            "[Human]:",
            "AI:",
            "Human:",
            "<|im_start|>",
            "<|assistant|>",
            "<|user|>",
            "<|eot_id|>",
            "<|startoftext|>",
            "<|sep|>",
        ]
        self.artifact_pattern = re.compile("|".join(map(re.escape, artifacts)))

    def validate_and_extract(self, exclude_ids: List[str] = None, exclude_variants: Dict[str, List[Tuple[int, int]]] = None) -> List:
        """Extract valid samples from the input data."""

        logger.info("Starting sample validation and extraction")

        if not self.data:
            logger.error("No data provided for validation")
            return []

        # Initialize exclusion lists
        exclude_ids = exclude_ids or []
        exclude_variants = exclude_variants or {}

        if exclude_ids:
            logger.info(f"Excluding {len(exclude_ids)} sample IDs")

        if exclude_variants:
            logger.info(f"Excluding variants for {len(exclude_variants)} samples")

        # Find valid datasets
        datasets = [
            key for key in self.data
            if isinstance(self.data[key], dict) and "samples" in self.data[key]
        ]

        if not datasets:
            logger.error("No valid datasets found")
            return []

        logger.info(f"Processing {len(datasets)} datasets: {', '.join(datasets)}")

        # Process each dataset
        for dataset_name in datasets:
            dataset_samples = self.data[dataset_name].get("samples", [])
            if not dataset_samples:
                logger.warning(f"Skipping empty dataset: {dataset_name}")
                continue

            valid_samples = self._process_dataset(dataset_name, dataset_samples, exclude_ids, exclude_variants)
            self.samples.extend(valid_samples)
            logger.info(f"Added {len(valid_samples)} samples from {dataset_name}")

        # Log filtering results
        logger.info(f"Filtering summary: {dict(self.filtered_count)}")
        if self.flagged_samples:
            logger.warning(f"Flagged {len(self.flagged_samples)} samples: {self.flagged_samples}")

        return self.samples

    def _process_dataset(self, dataset_name: str, dataset_samples: List[Dict], exclude_ids: List[str], exclude_variants: Dict[str, List[Tuple[int, int]]]) -> List:
        """Process samples in a dataset."""

        valid_samples = []

        for sample in dataset_samples:
            dataset_id = f"{dataset_name}:{sample.get('question_id', '')}"

            # Skip excluded samples
            if dataset_id in exclude_ids:
                logger.info(f"Excluding sample: {dataset_id}")
                self.filtered_count["excluded_id"] += 1
                continue

            # Add dataset name to sample
            sample["dataset"] = dataset_name

            # Filter excluded variants
            if dataset_id in exclude_variants:
                self._filter_variants(sample, exclude_variants[dataset_id], dataset_id)

            # Validate sample
            if self._validate_sample(sample, dataset_name, dataset_id):
                valid_samples.append(sample)

        return valid_samples

    def _filter_variants(self, sample: Dict, excluded_indices: List[Tuple[int, int]], dataset_id: str) -> None:
        """Remove specified variants from generations."""

        for gen_idx, generation in enumerate(sample.get("generations", [])):
            variants_to_exclude = [var_idx for g_idx, var_idx in excluded_indices if g_idx == gen_idx]
            if not variants_to_exclude:
                continue

            original_variants = generation.get("variants", [])
            filtered_variants = [
                variant for idx, variant in enumerate(original_variants)
                if idx not in variants_to_exclude
            ]

            excluded_count = len(original_variants) - len(filtered_variants)
            if excluded_count:
                logger.info(f"Excluded {excluded_count} variants from generation {gen_idx} of {dataset_id}")
                self.filtered_count["excluded_variant"] += excluded_count

            generation["variants"] = filtered_variants

    def _validate_sample(self, sample: Dict, dataset_name: str, dataset_id: str) -> bool:
        """Validate a sample's structure and content."""

        # Check required fields
        required_fields = ["question_id", "question", "original_answer", "generations"]
        if not all(field in sample for field in required_fields):
            logger.error(f"Missing required fields in sample from {dataset_name}")
            return False

        # Validate original answer
        original_answer = sample["original_answer"]
        if not all(field in original_answer for field in ["text", "metrics"]):
            logger.error(f"Missing fields in original answer from {dataset_name}")
            return False

        # Clean original answer text
        original_answer["text"] = self._clean_text(original_answer["text"])

        # Check for generations
        if not sample["generations"]:
            logger.warning(f"No generations in sample {dataset_id}")
            return False

        # Validate each generation
        for gen_idx, generation in enumerate(sample["generations"]):
            if not self._validate_generation(generation, dataset_id, gen_idx):
                return False

        return True

    def _validate_generation(self, generation: Dict, dataset_id: str, gen_idx: int) -> bool:
        """Validate a generation and its variants."""

        # Check required fields
        required_fields = ["model_used", "temperature", "variants"]
        if not all(field in generation for field in required_fields):
            logger.error(f"Missing fields in generation for {dataset_id}")
            return False

        # Check for variants
        if not generation["variants"]:
            logger.warning(f"No variants in generation {gen_idx} for {dataset_id}")
            return False

        empty_variants = []
        for var_idx, variant in enumerate(generation["variants"]):
            # Clean variant text
            text = variant.get("text", "")
            variant["text"] = self._clean_text(text)

            # Check for content mismatch
            if "CONTENT_MISMATCH" in variant["text"]:
                logger.info(f"Content mismatch in variant {gen_idx},{var_idx} of {dataset_id}")
                self.filtered_count["content_mismatch"] += 1
                return False

            # Parse complexity level
            complexity = variant.get("complexity_level", "")
            if isinstance(complexity, str):
                cleaned = self._clean_text(complexity)
                match = re.search(r'\d+', cleaned)
                variant["complexity_level"] = int(match.group()) if match else None
                if not match:
                    logger.warning(f"Invalid complexity in {dataset_id} at ({gen_idx},{var_idx})")

            # Check for empty text
            if not variant["text"].strip():
                empty_variants.append(var_idx)
                continue

            # Validate complexity level
            level = variant.get("complexity_level")
            if level not in [1, 2, 3, 4, 5, None]:
                flag_id = f"{dataset_id}:{gen_idx},{var_idx}"
                logger.warning(f"Invalid complexity level {level} in {flag_id}")
                self.flagged_samples.append(flag_id)

        # Handle empty variants
        if empty_variants:
            if len(empty_variants) == len(generation["variants"]):
                logger.info(f"All variants empty in generation {gen_idx} of {dataset_id}")
                self.filtered_count["empty_text"] += 1
                return False
            logger.warning(f"Empty variants {empty_variants} in generation {gen_idx} of {dataset_id}")
            self.filtered_count["partial_empty"] += 1
            self.flagged_samples.append(f"{dataset_id}:{gen_idx}")

        return True

    def _clean_text(self, text: str) -> str:
        """Remove AI artifacts from text."""

        if not text:
            return ""

        cleaned = self.artifact_pattern.sub("", text)
        cleaned = " ".join(cleaned.split())

        if cleaned != text:
            self.filtered_count["artifacts_cleaned"] += 1

        return cleaned


class DataTransformer:
    """Transforms validated samples into a structured DataFrame."""

    def __init__(self, samples: List):
        self.samples = samples

    def transform(self) -> pd.DataFrame:
        """Create a DataFrame from validated samples."""

        logger.info("Transforming samples to DataFrame")

        if not self.samples:
            logger.error("No samples to transform")
            return pd.DataFrame()

        # Get metric paths from the first sample
        metric_paths = self._get_metric_paths(self.samples[0])
        logger.info(f"Found {len(metric_paths)} metric paths")

        # Process all samples
        rows = []
        for sample in self.samples:
            rows.extend(self._process_sample(sample, metric_paths))

        if not rows:
            logger.error("No rows generated from samples")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns
        standard_cols = [
            "sample_id",
            "text_type",
            "dataset",
            "question",
            "complexity_level",
            "text",
            "model_used",
            "temperature"
        ]
        metric_cols = [col for col in df.columns if col not in standard_cols]
        df = df[standard_cols + metric_cols]

        # Handle infinity values
        df = self._handle_infinity(df)

        logger.info(f"Created DataFrame with {len(df)} rows, {len(df.columns)} columns")
        return df

    def _process_sample(self, sample: Dict, metric_paths: List[str]) -> List[Dict]:
        """Convert a sample into DataFrame rows."""

        dataset = sample.get("dataset", "unknown")
        sample_id = sample["question_id"]
        question = sample["question"]
        original = sample["original_answer"]

        # Row for original answer
        rows = [{
            "sample_id": sample_id,
            "text_type": "original",
            "dataset": dataset,
            "question": question,
            "complexity_level": None,
            "text": original["text"],
            "model_used": None,
            "temperature": None,
            **{path: self._get_metric(original["metrics"], path.split("/")) for path in metric_paths}
        }]

        # Rows for variants
        for gen_idx, generation in enumerate(sample.get("generations", [])):
            for var_idx, variant in enumerate(generation.get("variants", [])):
                if not variant.get("text", "").strip():
                    logger.warning(f"Skipping empty variant {gen_idx},{var_idx} in {sample_id}")
                    continue

                rows.append({
                    "sample_id": sample_id,
                    "text_type": "variant",
                    "dataset": dataset,
                    "question": question,
                    "complexity_level": variant.get("complexity_level"),
                    "text": variant["text"],
                    "model_used": generation["model_used"],
                    "temperature": generation["temperature"],
                    **{path: self._get_metric(variant.get("metrics", {}), path.split("/")) for path in metric_paths}
                })

        return rows

    def _get_metric_paths(self, sample: Dict) -> List[str]:
        """Extract all metric paths from a sample."""

        paths = []

        # Extract from original answer
        self._extract_paths(sample["original_answer"]["metrics"], "", paths)

        # Extract from first variant (if available)
        if sample["generations"] and sample["generations"][0]["variants"]:
            first_variant = sample["generations"][0]["variants"][0]
            if "metrics" in first_variant:
                self._extract_paths(first_variant["metrics"], "", paths)

        return sorted(set(paths))

    def _extract_paths(self, obj: Dict, prefix: str, paths: List) -> None:
        """Recursively collect metric paths from nested dictionaries."""

        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            path = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                self._extract_paths(value, path, paths)
                continue

            if isinstance(value, (int, float, bool, str)) or value is None:
                paths.append(path)

    def _get_metric(self, obj: Dict, path: List[str]) -> any:
        """Retrieve a nested metric value."""

        current = obj
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _handle_infinity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinity values in numeric columns."""

        if df.empty:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "sample_id"]

        for col in numeric_cols:
            inf_mask = np.isinf(df[col])
            if not inf_mask.any():
                continue

            finite = df[col][~inf_mask]
            if not len(finite):
                logger.warning(f"All values in {col} are infinite, using zeros")
                df.loc[inf_mask, col] = 0
                continue

            df.loc[np.isposinf(df[col]), col] = finite.max()
            df.loc[np.isneginf(df[col]), col] = finite.min()
            logger.info(f"Handled infinity in {col}")

        return df

    def check_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Report missing values in the DataFrame."""

        if df.empty:
            logger.info("No data to check for missing values")
            return df

        missing_total = df.isnull().sum().sum()
        if not missing_total:
            logger.info("No missing values found")
            return df

        logger.info(f"Found {missing_total} missing values")

        # Expected missing values for original answers
        expected_missing = {
            "complexity_level": df[df["text_type"] == "original"].shape[0],
            "model_used": df[df["text_type"] == "original"].shape[0],
            "temperature": df[df["text_type"] == "original"].shape[0]
        }

        for col in df.columns[df.isnull().any()]:
            count = df[col].isnull().sum()
            pct = (count / len(df)) * 100
            expected = expected_missing.get(col, 0)
            unexpected = count - expected

            if expected and unexpected <= 0:
                logger.info(f"  {col}: {count} missing ({pct:.1f}%) - expected")
                continue

            if unexpected > 0:
                logger.warning(f"  {col}: {count} missing ({pct:.1f}%) - {unexpected} unexpected")
                continue

            logger.info(f"  {col}: {count} missing ({pct:.1f}%)")

        return df


class Phase2Pipeline:
    """Orchestrates the text complexity analysis pipeline."""

    def __init__(self, in_file: Union[str, Path], out_dir: Union[str, Path] = None, exclude_ids: List[str] = None):
        self.in_file = Path(in_file)
        self.out_dir = Path(out_dir) if out_dir else Path("outputs/phase2")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.exclude_ids, self.exclude_variants = self._parse_exclusions(exclude_ids or [])
        self.samples = []
        self.df = pd.DataFrame()
        self.validator = None

    def _parse_exclusions(self, exclude_ids: List[str]) -> Tuple[List[str], Dict[str, List[Tuple[int, int]]]]:
        """Parse exclusion IDs for samples or variants."""

        full_exclusions = []
        variant_exclusions = {}

        for entry in exclude_ids:
            parts = entry.split(":")
            if len(parts) not in [2, 3]:
                logger.warning(f"Invalid exclusion format: {entry}")
                continue

            dataset, sample_id = parts[:2]
            full_id = f"{dataset}:{sample_id}"

            if len(parts) == 2:
                full_exclusions.append(full_id)
                logger.info(f"Excluding sample {full_id}")
                continue

            indices = parts[2].split(",")
            if len(indices) != 2 or not all(idx.isdigit() for idx in indices):
                logger.warning(f"Invalid variant indices in {entry}")
                continue

            gen_idx, var_idx = map(int, indices)
            if gen_idx < 0 or var_idx < 0:
                logger.warning(f"Negative indices in {entry}")
                continue

            variant_exclusions.setdefault(full_id, []).append((gen_idx, var_idx))
            logger.info(f"Excluding variant ({gen_idx},{var_idx}) from {full_id}")

        return full_exclusions, variant_exclusions

    def run(self) -> pd.DataFrame:
        """Execute the pipeline."""

        logger.info(f"Starting pipeline with {self.in_file}")

        self._load_and_validate()

        if not self.samples:
            logger.error("No valid samples to process")
            return pd.DataFrame()

        self._transform()
        self._save()

        logger.info(f"Pipeline complete: {len(self.samples)} samples, {len(self.df)} rows")
        return self.df

    def _load_and_validate(self) -> None:
        """Load and validate input data."""

        if not self.in_file.exists():
            logger.error(f"Input file not found: {self.in_file}")
            return

        data = load_json(self.in_file)
        self.validator = DataValidator(data)
        self.samples = self.validator.validate_and_extract(self.exclude_ids, self.exclude_variants)
        logger.info(f"Validated {len(self.samples)} samples")

    def _transform(self) -> None:
        """Transform samples into a DataFrame."""

        transformer = DataTransformer(self.samples)
        self.df = transformer.transform()

        if self.df.empty:
            logger.error("No data transformed")
            return

        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        if initial_rows > len(self.df):
            logger.info(f"Removed {initial_rows - len(self.df)} duplicate rows")

        self.df = transformer.check_missing(self.df)

    def _save(self) -> None:
        """Save output files."""

        # Save full DataFrame
        metrics_path = self.out_dir / "metrics_df.csv"
        self.df.to_csv(metrics_path, index=False)
        logger.info(f"Saved {len(self.df)} rows to {metrics_path}")

        # Save sample if large
        if len(self.df) > 100:
            sample_path = self.out_dir / "metrics_df_sample.csv"
            self.df.head(100).to_csv(sample_path, index=False)
            logger.info(f"Saved sample of 100 rows to {sample_path}")

        # Save flagged samples
        if self.validator.flagged_samples:
            flagged_path = self.out_dir / "flagged_samples.json"
            save_json(self.validator.flagged_samples, flagged_path)
            logger.info(f"Saved {len(self.validator.flagged_samples)} flagged samples to {flagged_path}")


def main():
    """Run the pipeline from the command line."""

    parser = argparse.ArgumentParser(description="Phase 2: Text complexity analysis pipeline")
    parser.add_argument("--input-file", required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", help="Path to output directory")
    parser.add_argument("--exclude-ids", nargs="*", help="IDs to exclude (e.g., 'dataset:id' or 'dataset:id:gen_idx,var_idx')")

    args = parser.parse_args()

    pipeline = Phase2Pipeline(
        in_file=args.input_file,
        out_dir=args.output_dir,
        exclude_ids=args.exclude_ids
    )

    pipeline.run()


if __name__ == "__main__":
    main()