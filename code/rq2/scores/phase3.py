import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
from dataclasses import dataclass

import pandas as pd

from utils.helpers import setup_logging, save_json
from rq2.scores.impl.evaluation import (
    TextEvaluationMetrics, 
    RougeMetric, 
    BleuMetric, 
    MeteorMetric,
    BertScoreMetric,
    SemScoreMetric,
    BleurtMetric,
    BartScoreMetric,
    AlignScoreMetric,
    SummaCMetric,
    FactCCMetric,
    UniEvalSumMetric,
    UniEvalFactMetric,
    BaseMetric
)

# Initialize logging
logger = setup_logging()

# Constants
METADATA_COLS = ["dataset", "sample_id", "model", "temperature", 
                 "complexity_level", "variant_text", "original_text", "question"]

RELEVANCE_PREFIXES = ('rouge', 'bleu', 'meteor', 'bertscore', 'semscore', 
                      'bleurt', 'bartscore', 'unieval-sum')
FACTUALITY_PREFIXES = ('alignscore', 'summac', 'factcc', 'unieval-fact')

REQUIRED_COLUMNS = ['sample_id', 'text_type', 'dataset', 'question', 'text', 'complexity_level']

METRIC_INITIALIZERS = {
    "rouge": RougeMetric,
    "bleu": BleuMetric,
    "meteor": MeteorMetric,
    "bertscore": BertScoreMetric,
    "semscore": SemScoreMetric,
    "bleurt": BleurtMetric,
    "bartscore": BartScoreMetric,
    "alignscore": AlignScoreMetric,
    "summac": SummaCMetric,
    "factcc": FactCCMetric,
    "unieval-sum": UniEvalSumMetric,
    "unieval-fact": UniEvalFactMetric
}

DEFAULT_METRICS = [
    "rouge", "bleu", "meteor", "bertscore", "semscore", 
    "bleurt", "bartscore", "alignscore", "summac",
    "factcc", "unieval-sum", "unieval-fact"
]


@dataclass
class SampleData:
    """Data structure for a sample to be processed."""
    dataset: str
    sample_id: str
    question: str
    model: str
    temperature: float
    complexity_level: int
    variant_text: str
    original_text: str


class DataLoader:
    """Handles data loading and validation."""
    
    def __init__(self, input_file: Path):
        self.input_file = input_file
        self.df = None
    
    def load_and_validate(self) -> bool:
        """Load data from CSV file and validate structure."""
        if not self._check_file_exists():
            return False
        
        if not self._load_csv():
            return False
        
        if not self._validate_structure():
            return False
        
        self._log_data_summary()
        return True
    
    def _check_file_exists(self) -> bool:
        """Check if input file exists and is readable."""
        if not self.input_file.exists():
            logger.error(f"File not found: {self.input_file}")
            return False
        
        if not os.access(self.input_file, os.R_OK):
            logger.error(f"File is not readable: {self.input_file}")
            return False
        
        return True
    
    def _load_csv(self) -> bool:
        """Load CSV file into DataFrame."""
        self.df = pd.read_csv(self.input_file)
        logger.info(f"Loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
        return True
    
    def _validate_structure(self) -> bool:
        """Validate DataFrame structure and required columns."""
        if self.df.empty:
            logger.error(f"CSV file is empty: {self.input_file}")
            return False
        
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _log_data_summary(self) -> None:
        """Log summary of loaded data."""
        dataset_counts = self.df['dataset'].value_counts()
        logger.info(f"Datasets in the data: {', '.join(dataset_counts.index)}")


class MetricsInitializer:
    """Handles initialization of evaluation metrics."""
    
    def __init__(self, metrics_to_use: List[str], batch_size: int):
        self.metrics_to_use = metrics_to_use
        self.batch_size = batch_size
    
    def create_evaluator(self) -> TextEvaluationMetrics:
        """Create and initialize the text evaluation metrics."""
        evaluator = TextEvaluationMetrics(metrics=[], batch_size=self.batch_size)
        
        added_metrics = []
        for metric_name in self.metrics_to_use:
            if not self._add_metric_to_evaluator(evaluator, metric_name):
                continue
            added_metrics.append(metric_name)
        
        self._log_metrics_summary(evaluator)
        return evaluator
    
    def _add_metric_to_evaluator(self, evaluator: TextEvaluationMetrics, metric_name: str) -> bool:
        """Add a single metric to the evaluator."""
        if metric_name not in METRIC_INITIALIZERS:
            logger.warning(f"Metric {metric_name} not recognized; skipping")
            return False
        
        initializer = METRIC_INITIALIZERS[metric_name]
        metric_instance = initializer()
        evaluator.add_metric(metric_instance)
        return True
    
    def _log_metrics_summary(self, evaluator: TextEvaluationMetrics) -> None:
        """Log summary of added metrics."""
        metrics_added = evaluator.get_available_metrics()
        logger.info(f"Initialized metrics: "
                    f"Relevance({len(metrics_added['relevance'])}): {', '.join(metrics_added['relevance'])}, "
                    f"Factuality({len(metrics_added['factuality'])}): {', '.join(metrics_added['factuality'])}")


class SampleCollector:
    """Handles collection and preparation of samples for processing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def collect_samples(self) -> List[SampleData]:
        """Collect and organize sample data for batch processing."""
        logger.info("Organizing data for batch processing")
        
        original_texts, original_questions = self._extract_original_data()
        if not original_texts:
            logger.warning("No original texts found in the data")
            return []
        
        valid_variants = self._get_valid_variants()
        if valid_variants.empty:
            logger.warning("No valid variants found after filtering missing values")
            return []
        
        samples = self._create_sample_objects(valid_variants, original_texts, original_questions)
        logger.info(f"Prepared {len(samples)} variants for batch processing")
        return samples
    
    def _extract_original_data(self) -> tuple[Dict, Dict]:
        """Extract original texts and questions from the DataFrame."""
        original_df = self.df[self.df['text_type'] == 'original']
        if original_df.empty:
            return {}, {}
        
        original_df = original_df.set_index(['dataset', 'sample_id'])
        original_texts = original_df['text'].to_dict()
        original_questions = original_df['question'].to_dict()
        
        return original_texts, original_questions
    
    def _get_valid_variants(self) -> pd.DataFrame:
        """Get valid variant data from the DataFrame."""
        variant_df = self.df[self.df['text_type'] == 'variant']
        if variant_df.empty:
            logger.warning("No variants found in the data")
            return pd.DataFrame()
        
        return variant_df.dropna(subset=['model_used', 'temperature', 'complexity_level'])
    
    def _create_sample_objects(self, variants_df: pd.DataFrame, 
                              original_texts: Dict, original_questions: Dict) -> List[SampleData]:
        """Create SampleData objects from variant DataFrame."""
        samples = []
        
        for _, row in variants_df.iterrows():
            sample = self._create_single_sample(row, original_texts, original_questions)
            if sample:
                samples.append(sample)
        
        return samples
    
    def _create_single_sample(self, row: pd.Series, 
                             original_texts: Dict, original_questions: Dict) -> Optional[SampleData]:
        """Create a single SampleData object from a row."""
        key = (row['dataset'], row['sample_id'])
        
        if key not in original_texts:
            logger.warning(f"No original text found for {row['dataset']}:{row['sample_id']}, skipping")
            return None
        
        return SampleData(
            dataset=row['dataset'],
            sample_id=row['sample_id'],
            question=original_questions.get(key, ''),
            model=row['model_used'],
            temperature=row['temperature'],
            complexity_level=int(row['complexity_level']),
            variant_text=row['text'],
            original_text=original_texts[key]
        )


class BatchProcessor:
    """Handles batch processing of samples for evaluation."""
    
    def __init__(self, evaluator: TextEvaluationMetrics, batch_size: int):
        self.evaluator = evaluator
        self.batch_size = batch_size
    
    def process_samples(self, samples: List[SampleData]) -> List[Dict]:
        """Process samples in batches for efficient evaluation."""
        logger.info("Preloading evaluation models")
        self.evaluator.preload_all_models()
        
        all_scores = []
        total_batches = (len(samples) - 1) // self.batch_size + 1
        
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_scores = self._process_single_batch(batch, batch_num)
            all_scores.extend(batch_scores)
        
        return all_scores
    
    def _process_single_batch(self, batch: List[SampleData], batch_num: int) -> List[Dict]:
        """Process a single batch of samples."""
        valid_batch = self._filter_valid_samples(batch, batch_num)
        if not valid_batch:
            return []
        
        # Extract texts for evaluation
        variant_texts = [sample.variant_text for sample in valid_batch]
        original_texts = [sample.original_text for sample in valid_batch]
        questions = [sample.question for sample in valid_batch]
        
        # Calculate scores
        batch_scores = self.evaluator.batch_compute_all(
            variant_texts, original_texts, input_texts=questions
        )
        
        if not self._validate_batch_results(batch_scores, valid_batch, batch_num):
            return []
        
        return self._combine_metadata_with_scores(valid_batch, batch_scores)
    
    def _filter_valid_samples(self, batch: List[SampleData], batch_num: int) -> List[SampleData]:
        """Filter out samples with empty texts."""
        valid_samples = [
            sample for sample in batch 
            if sample.variant_text and sample.original_text
        ]
        
        if len(valid_samples) < len(batch):
            logger.warning(f"Found {len(batch) - len(valid_samples)} invalid entries in batch {batch_num}")
        
        return valid_samples
    
    def _validate_batch_results(self, batch_scores: List[Dict], 
                               valid_batch: List[SampleData], batch_num: int) -> bool:
        """Validate that batch processing returned expected results."""
        if not batch_scores or len(batch_scores) != len(valid_batch):
            logger.error(f"Evaluation returned unexpected results for batch {batch_num}")
            return False
        return True
    
    def _combine_metadata_with_scores(self, samples: List[SampleData], 
                                     scores: List[Dict]) -> List[Dict]:
        """Combine sample metadata with calculated scores."""
        combined_results = []
        
        for sample, score in zip(samples, scores):
            combined_result = {
                'dataset': sample.dataset,
                'sample_id': sample.sample_id,
                'question': sample.question,
                'model': sample.model,
                'temperature': sample.temperature,
                'complexity_level': sample.complexity_level,
                'variant_text': sample.variant_text,
                'original_text': sample.original_text,
                **score
            }
            combined_results.append(combined_result)
        
        return combined_results


class ResultsOrganizer:
    """Handles organization of results into hierarchical structure."""
    
    def __init__(self):
        self.results = {}
    
    def organize_results(self, all_scores: List[Dict]) -> Dict:
        """Organize scores into the required hierarchical structure."""
        logger.info("Organizing results into hierarchical structure")
        
        datasets = self._group_by_dataset(all_scores)
        
        for dataset, scores in datasets.items():
            self.results[dataset] = self._process_dataset(dataset, scores)
        
        return self.results
    
    def _group_by_dataset(self, all_scores: List[Dict]) -> Dict[str, List[Dict]]:
        """Group scores by dataset."""
        datasets = {}
        for score in all_scores:
            dataset = score['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(score)
        return datasets
    
    def _process_dataset(self, dataset: str, scores: List[Dict]) -> Dict:
        """Process a single dataset's scores."""
        samples_by_id = self._group_by_sample_id(scores)
        
        dataset_result = {
            "dataset": dataset,
            "samples": []
        }
        
        for sample_id, sample_scores in samples_by_id.items():
            sample = self._create_sample_structure(dataset, sample_id, sample_scores)
            if sample:
                dataset_result["samples"].append(sample)
        
        return dataset_result
    
    def _group_by_sample_id(self, scores: List[Dict]) -> Dict[str, List[Dict]]:
        """Group scores by sample ID."""
        samples_by_id = {}
        for score in scores:
            sample_id = score['sample_id']
            if sample_id not in samples_by_id:
                samples_by_id[sample_id] = []
            samples_by_id[sample_id].append(score)
        return samples_by_id
    
    def _create_sample_structure(self, dataset: str, sample_id: str, 
                               sample_scores: List[Dict]) -> Optional[Dict]:
        """Create a structured sample entry with all variants."""
        if not sample_scores:
            return None
        
        # Use first score for common metadata
        first_score = sample_scores[0]
        
        sample = {
            "question_id": sample_id,
            "question": first_score['question'],
            "original_answer": {
                "text": first_score['original_text']
            },
            "generations": []
        }
        
        model_groups = self._group_by_model_and_temperature(sample_scores)
        
        for (model, temperature), model_scores in model_groups.items():
            variants = self._create_variant_structures(model_scores)
            
            sample["generations"].append({
                "model_used": model,
                "temperature": temperature,
                "variants": sorted(variants, key=lambda v: v["complexity_level"])
            })
        
        return sample
    
    def _group_by_model_and_temperature(self, scores: List[Dict]) -> Dict[tuple, List[Dict]]:
        """Group scores by model and temperature combination."""
        model_groups = {}
        for score in scores:
            model_key = (score['model'], score['temperature'])
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(score)
        return model_groups
    
    def _create_variant_structures(self, scores: List[Dict]) -> List[Dict]:
        """Create variant structures from score entries."""
        variants = []
        
        for score in scores:
            relevance_scores, factuality_scores, other_scores = self._categorize_scores(score)
            
            variant = {
                "complexity_level": score['complexity_level'],
                "text": score['variant_text'],
                "scores": {
                    "relevance": relevance_scores,
                    "factuality": factuality_scores
                }
            }
            
            if other_scores:
                variant["scores"]["other"] = other_scores
            
            variants.append(variant)
        
        return variants
    
    def _categorize_scores(self, score: Dict) -> tuple[Dict, Dict, Dict]:
        """Categorize scores into relevance, factuality, and other."""
        relevance_scores = {}
        factuality_scores = {}
        other_scores = {}
        
        for key, value in score.items():
            if key in METADATA_COLS:
                continue
            
            if any(key.startswith(prefix) for prefix in RELEVANCE_PREFIXES):
                relevance_scores[key] = value
            elif any(key.startswith(prefix) for prefix in FACTUALITY_PREFIXES):
                factuality_scores[key] = value
            else:
                other_scores[key] = value
        
        return relevance_scores, factuality_scores, other_scores


class ResultsSaver:
    """Handles saving of results to files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, scores_df: pd.DataFrame, hierarchical_results: Dict) -> None:
        """Save results to output files."""
        self._save_scores_dataframe(scores_df)
        self._save_hierarchical_results(hierarchical_results)
    
    def _save_scores_dataframe(self, scores_df: pd.DataFrame) -> None:
        """Save scores DataFrame to CSV."""
        if scores_df.empty:
            logger.error("No scores dataframe to save")
            return
        
        # Remove duplicates and convert numeric columns
        scores_df = scores_df.drop_duplicates()
        scores_df = self._convert_numeric_columns(scores_df)
        
        logger.info(f"Removed duplicates, remaining {len(scores_df)} rows")
        
        # Save full CSV
        csv_path = self.output_dir / "scores_df.csv"
        scores_df.to_csv(csv_path, index=False)
        logger.info(f"Saved scores dataframe to {csv_path}")
        
        # Save sample for large datasets
        if len(scores_df) > 100:
            sample_path = self.output_dir / "scores_sample.csv"
            scores_df.head(100).to_csv(sample_path, index=False)
            logger.info(f"Saved sample scores (first 100 rows) to {sample_path}")
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert appropriate columns to numeric types."""
        for col in df.columns:
            if col not in ['variant_text', 'original_text', 'question', 'dataset', 'model', 'sample_id']:
                df[col] = pd.to_numeric(df[col], errors='ignore')
        return df
    
    def _save_hierarchical_results(self, results: Dict) -> None:
        """Save hierarchical results to JSON."""
        if not results:
            logger.error("No hierarchical results to save")
            return
        
        json_path = self.output_dir / "scores.json"
        save_json(results, json_path)
        logger.info(f"Saved hierarchical results to {json_path}")


class Phase3Pipeline:
    """Phase 3 of text complexity analysis pipeline."""

    def __init__(self, 
                input_file: Union[str, Path], 
                output_dir: Union[str, Path] = None, 
                metrics_to_use: Optional[List[str]] = None,
                batch_size: int = 128) -> None:
        """Initialize the pipeline."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/phase3")
        self.batch_size = batch_size
        self.metrics_to_use = metrics_to_use or DEFAULT_METRICS
        
        logger.info(f"Initialized Phase3Pipeline with batch size {self.batch_size}")
    
    def run(self) -> Dict:
        """Execute the Phase 3 pipeline."""
        logger.info(f"Running Phase 3 pipeline on {self.input_file}")
        
        # Load and validate data
        data_loader = DataLoader(self.input_file)
        if not data_loader.load_and_validate():
            return {}
        
        # Initialize metrics
        metrics_initializer = MetricsInitializer(self.metrics_to_use, self.batch_size)
        evaluator = metrics_initializer.create_evaluator()
        
        # Collect samples
        sample_collector = SampleCollector(data_loader.df)
        samples = sample_collector.collect_samples()
        if not samples:
            logger.error("No valid samples found for processing")
            return {}
        
        # Process samples
        batch_processor = BatchProcessor(evaluator, self.batch_size)
        all_scores = batch_processor.process_samples(samples)
        if not all_scores:
            logger.error("Score calculation failed")
            return {}
        
        # Create scores DataFrame
        scores_df = pd.DataFrame(all_scores)
        logger.info(f"Created scores dataframe with {len(scores_df)} rows")
        
        # Organize results
        results_organizer = ResultsOrganizer()
        hierarchical_results = results_organizer.organize_results(all_scores)
        
        # Save results
        results_saver = ResultsSaver(self.output_dir)
        results_saver.save_results(scores_df, hierarchical_results)
        
        return hierarchical_results


def main() -> None:
    """Run Phase 3 from CLI."""
    parser = argparse.ArgumentParser(description="Phase 3 of text complexity analysis pipeline")
    parser.add_argument("--input-file", required=True, help="Path to input CSV file from Phase 2")
    parser.add_argument("--output-dir", required=True, help="Directory for saving output files")
    parser.add_argument("--metrics", nargs="+", 
                        choices=["rouge", "bleu", "meteor", "bertscore", "semscore", 
                                 "bleurt", "bartscore", "alignscore", "summac",
                                 "factcc", "unieval-sum", "unieval-fact", "all"],
                        default=["unieval-sum", "unieval-fact", "factcc", "rouge", "bertscore"],
                        help="Metrics to use for evaluation")
    parser.add_argument("--batch-size", type=int, default=128, 
                        help="Batch size for processing (default: 128)")
    args = parser.parse_args()
    
    # Handle 'all' metrics option
    metrics_to_use = None if "all" in args.metrics else args.metrics

    # Run pipeline
    pipeline = Phase3Pipeline(
        input_file=args.input_file, 
        output_dir=args.output_dir,
        metrics_to_use=metrics_to_use,
        batch_size=args.batch_size,
    )

    pipeline.run()


if __name__ == "__main__":
    main()