import argparse
import re
from pathlib import Path
from collections import defaultdict

from utils.helpers import setup_logging, save_json, load_json

logger = setup_logging()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate readability metrics from JSON files")
    parser.add_argument("--input-dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output-file", required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--pattern",
        default="**/readability_metrics.json",
        help="Glob pattern to find JSON files",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Glob pattern to exclude files",
    )
    return parser.parse_args()


def initialize_samples(base_data: dict, dataset: str) -> list:
    """Initialize the samples structure with basic data and empty metrics."""
    samples = []
    for sample in base_data[dataset]["samples"]:
        aggregated_sample = {
            "question": sample["question"],
            "complexity_level": sample["complexity_level"],
            "model": sample["model"],
            "response": sample["response"],
            "metrics": {}
        }
        samples.append(aggregated_sample)
    return samples


def apply_metric_value(target_metrics: dict, metric_type: str, subcategories: list, value: dict) -> None:
    """Apply a metric value to the target metrics dictionary."""
    if metric_type not in target_metrics:
        target_metrics[metric_type] = {}

    # No subcategories: directly assign value
    if not subcategories:
        target_metrics[metric_type] = value
        return

    # Traverse subcategories to set nested metric value
    current = target_metrics[metric_type]
    for subcat in subcategories[:-1]:
        if subcat not in current:
            current[subcat] = {}
        current = current[subcat]
    
    current[subcategories[-1]] = value


def merge_metrics(aggregated_samples: list, file_data: dict, dataset: str, metric_type: str, subcategories: list = None) -> None:
    """Merge metrics from a file into the aggregated samples."""
    file_samples = file_data[dataset]["samples"]
    
    for agg_sample, file_sample in zip(aggregated_samples, file_samples):
        metrics_dict = file_sample["metrics"]
        if metric_type not in metrics_dict:
            continue
            
        metrics_value = metrics_dict[metric_type]
        apply_metric_value(agg_sample["metrics"], metric_type, subcategories, metrics_value)


def get_metric_subcategories(metric_type: str, config: dict) -> list:
    """Get subcategories for a given metric type based on config."""
    # Mapping from metric types to lambda functions extracting subcategories
    handlers = {
        "syntax": lambda c: [c["spacy_model_syntax"]] if c.get("spacy_model_syntax") else None,
        "textstat": lambda c: None,
        "umls": lambda c: [c["spacy_model_umls"]] if c.get("spacy_model_umls") else None,
        "commonlit": lambda c: [c["commonlit_model_type"]] if c.get("commonlit_model_type") else None,
        "masked_prob": lambda c: [
            c.get("masked_prob_method", "random"),
            c["masked_prob_model"].split('/')[-1]
        ] if c.get("masked_prob_model") else None,
        "gispy": lambda c: None,
        "scigispy": lambda c: None,
        "cluster": lambda c: [c["cluster_model_path"].split('/')[-1]] if c.get("cluster_model_path") else None,
        "llm": lambda c: [c["llm_model_path"].split('/')[-1]] if c.get("llm_model_path") else None,
        "jargon": lambda c: [c["jargon_pretrained_model"]] if c.get("jargon_pretrained_model") else None,
    }
    
    handler = handlers.get(metric_type)
    if not handler:
        return None
    
    return handler(config)


def determine_metric_info(file_data: dict, dataset: str) -> tuple[str, list]:
    """Determine metric type and subcategories from the config and data."""
    config = file_data[dataset]["config"]
    samples = file_data[dataset]["samples"]
    metric_type = list(samples[0]["metrics"].keys())[0]  # Assume only one metric per file
    subcategories = get_metric_subcategories(metric_type, config)
    return metric_type, subcategories


def calculate_metric_average(models: dict, metric_name: str, metric_value) -> any:
    """Calculate average for a single metric across all valid models."""
    if isinstance(metric_value, str):
        return None  # Skip string values (e.g., labels)

    if isinstance(metric_value, dict):
        # Recursively average each sub-key
        nested_avg = {}
        for sub_key in metric_value:
            sub_values = [models[model][metric_name][sub_key] for model in models.keys()]
            nested_avg[sub_key] = sum(sub_values) / len(sub_values)
        return nested_avg

    if isinstance(metric_value, (int, float)):
        values = [models[model][metric_name] for model in models.keys()]
        return sum(values) / len(values)
    
    return None


def should_exclude_model(model_name: str) -> bool:
    """Check if a model should be excluded from averaging."""
    # Exclude models with special mse+kl+ce+soft pattern in their name
    mse_pattern = re.compile(r'mse_\d+\.\d+\+kl_\d+\.\d+\+ce_\d+\.\d+\+soft_\d+\.\d+')
    return mse_pattern.match(model_name) is not None


def calculate_llm_average(aggregated_samples: list) -> None:
    """Calculate the average of all LLM models for each sample, ignoring files with mse+kl+ce+soft patterns."""
    for sample in aggregated_samples:
        metrics = sample["metrics"]
        if "llm" not in metrics or len(metrics["llm"]) <= 1:
            continue
            
        models = metrics["llm"]
        # Exclude special LLM models from average
        valid_models = {name: data for name, data in models.items() if not should_exclude_model(name)}
        
        if not valid_models:
            continue
        
        first_model = next(iter(valid_models.values()))
        average_metrics = {}
        
        # Compute average for each metric field
        for metric_name, metric_value in first_model.items():
            avg_result = calculate_metric_average(valid_models, metric_name, metric_value)
            if avg_result is not None:
                average_metrics[metric_name] = avg_result
        
        metrics["llm"]["average"] = average_metrics


def find_and_filter_files(input_dir: Path, pattern: str, exclude: str = None) -> list[Path]:
    """Find all files matching the pattern and exclude those matching the exclude pattern."""
    logger.info(f"Searching for files in {input_dir} with pattern {pattern}")
    files = list(input_dir.glob(pattern))
    
    if exclude:
        logger.info(f"Excluding files matching pattern {exclude}")
        exclude_files = set(input_dir.glob(exclude))
        files = [f for f in files if f not in exclude_files]
    
    logger.info(f"Found {len(files)} files after filtering")
    return files

def group_files_by_dataset(files: list[Path]) -> dict[str, list[Path]]:
    """Group files by their dataset name."""
    dataset_files = defaultdict(list)
    for file in files:
        logger.info(f"Loading file to determine dataset: {file}")
        data = load_json(file)
        dataset = list(data.keys())[0]  # Assumes top-level key is the dataset name
        dataset_files[dataset].append(file)
        logger.info(f"File {file} belongs to dataset {dataset}")
    return dataset_files


def aggregate_data_for_dataset(dataset: str, file_list: list[Path]) -> dict:
    """Aggregate data for a single dataset."""
    base_file = file_list[0] # First file provides structure
    logger.info(f"Loading base file for dataset {dataset}: {base_file}")
    base_data = load_json(base_file)
    aggregated_samples = initialize_samples(base_data, dataset)
    config = base_data[dataset]["config"]

    for file in file_list:
        logger.info(f"Loading file for metric aggregation in dataset {dataset}: {file}")
        file_data = load_json(file)
        metric_type, subcategory = determine_metric_info(file_data, dataset)
        logger.info(f"Merging metrics for {metric_type} (subcategories: {subcategory}) from {file}")
        merge_metrics(aggregated_samples, file_data, dataset, metric_type, subcategory)

    # Compute average across valid LLM models
    logger.info(f"Calculating LLM average for dataset {dataset}")
    calculate_llm_average(aggregated_samples)

    return {
        "dataset": dataset,
        "config": config,
        "samples": aggregated_samples
    }


def main():
    """Aggregate readability metrics from JSON files into a single JSON file."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    # Find and filter files
    files = find_and_filter_files(input_dir, args.pattern, args.exclude)

    # Group files by dataset
    dataset_files = group_files_by_dataset(files)

    # Aggregate data for each dataset
    aggregated_data = {}
    for dataset, file_list in dataset_files.items():
        logger.info(f"Aggregating data for dataset {dataset} with {len(file_list)} files")
        aggregated_data[dataset] = aggregate_data_for_dataset(dataset, file_list)

    # Save the aggregated data to the output file
    logger.info(f"Saving aggregated data to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(aggregated_data, output_file)


if __name__ == "__main__":
    main()
