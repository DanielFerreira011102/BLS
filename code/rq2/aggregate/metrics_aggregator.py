import argparse
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
    """Initialize the samples structure with text and empty metrics."""
    samples = []
    for sample in base_data[dataset]["samples"]:
        aggregated_sample = {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "original_answer": {
                "text": sample["original_answer"]["text"],
                "metrics": {}  # Placeholder to receive merged metrics
            },
            "generations": []
        }
        
        for generation in sample["generations"]:
            new_generation = {
                "model_used": generation["model_used"],
                "temperature": generation["temperature"],
                "raw_response": generation.get("raw_response", ""),
                "variants": []
            }
            
            for variant in generation["variants"]:
                new_variant = {
                    "complexity_level": variant["complexity_level"],
                    "text": variant["text"],
                    "metrics": {}  # Placeholder for variant-level metrics
                }
                new_generation["variants"].append(new_variant)
            
            aggregated_sample["generations"].append(new_generation)
        
        samples.append(aggregated_sample)
    
    return samples


def apply_metric_value(target_metrics: dict, metric_type: str, subcategories: list, value: dict) -> None:
    """Apply a metric value to the target metrics dictionary."""
    if metric_type not in target_metrics:
        target_metrics[metric_type] = {}

    if not subcategories:
        # Direct assignment if no subcategories (e.g. for flat metrics like textstat)
        target_metrics[metric_type] = value
        return

    # Traverse or create nested subcategory structure
    current = target_metrics[metric_type]
    for subcat in subcategories[:-1]:
        if subcat not in current:
            current[subcat] = {}
        current = current[subcat]
    
    # Set the final subcategory value
    current[subcategories[-1]] = value


def merge_metrics(aggregated_samples: list, file_data: dict, dataset: str, metric_type: str, subcategories: list = None) -> None:
    """Merge metrics from a file into the aggregated samples."""
    file_samples = file_data[dataset]["samples"]
    
    for agg_sample, file_sample in zip(aggregated_samples, file_samples):
        # Merge original answer metrics
        if metric_type in file_sample["original_answer"]["metrics"]:
            metrics_value = file_sample["original_answer"]["metrics"][metric_type]
            target_metrics = agg_sample["original_answer"]["metrics"]
            apply_metric_value(target_metrics, metric_type, subcategories, metrics_value)
        
        # Merge variant metrics
        for agg_gen, file_gen in zip(agg_sample["generations"], file_sample["generations"]):
            for agg_var, file_var in zip(agg_gen["variants"], file_gen["variants"]):
                if metric_type not in file_var["metrics"]:
                    continue
                    
                metrics_value = file_var["metrics"][metric_type]
                target_metrics = agg_var["metrics"]
                apply_metric_value(target_metrics, metric_type, subcategories, metrics_value)


def get_metric_subcategories(metric_type: str, config: dict) -> list:
    """Get subcategories for a given metric type based on config."""
    # Handlers define how to extract subcategories for different metric types
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
    metric_type = list(samples[0]["original_answer"]["metrics"].keys())[0]  # Assume one metric per file
    subcategories = get_metric_subcategories(metric_type, config)
    return metric_type, subcategories


def calculate_metric_average(models: dict, metric_name: str, metric_value) -> any:
    """Calculate average for a single metric across all models."""
    if isinstance(metric_value, str):
        return None  # Cannot average strings
    
    if isinstance(metric_value, dict):
        # Average values per sub-key in nested metrics
        nested_avg = {}
        for sub_key in metric_value:
            sub_values = [models[model][metric_name][sub_key] for model in models.keys()]
            nested_avg[sub_key] = sum(sub_values) / len(sub_values)
        return nested_avg
    
    if isinstance(metric_value, (int, float)):
        values = [models[model][metric_name] for model in models.keys()]
        return sum(values) / len(values)
    
    return None


def calculate_llm_average_for_metrics(metrics: dict) -> None:
    """Calculate LLM average for a single metrics dictionary."""
    if "llm" not in metrics or len(metrics["llm"]) <= 1:
        return  # Skip if only one model (nothing to average)
        
    models = metrics["llm"]
    model_names = list(models.keys())
    
    first_model = models[model_names[0]]
    average_metrics = {}
    
    for metric_name, metric_value in first_model.items():
        avg_result = calculate_metric_average(models, metric_name, metric_value)
        if avg_result is not None:
            average_metrics[metric_name] = avg_result
    
    # Store average under special "average" key
    metrics["llm"]["average"] = average_metrics


def calculate_llm_average(aggregated_samples: list) -> None:
    """Calculate the average of all LLM models for each sample."""
    for sample in aggregated_samples:
        # Calculate for original answer
        calculate_llm_average_for_metrics(sample["original_answer"]["metrics"])
        
        # Calculate for variants
        for generation in sample["generations"]:
            for variant in generation["variants"]:
                calculate_llm_average_for_metrics(variant["metrics"])


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
    base_file = file_list[0]  # First file provides structure
    logger.info(f"Loading base file for dataset {dataset}: {base_file}")
    base_data = load_json(base_file)
    aggregated_samples = initialize_samples(base_data, dataset)

    for file in file_list:
        logger.info(f"Loading file for metric aggregation in dataset {dataset}: {file}")
        file_data = load_json(file)
        metric_type, subcategory = determine_metric_info(file_data, dataset)
        logger.info(f"Merging metrics for {metric_type} (subcategories: {subcategory}) from {file}")
        merge_metrics(aggregated_samples, file_data, dataset, metric_type, subcategory)

    # Calculate average across valid LLM models
    logger.info(f"Calculating LLM average for dataset {dataset}")
    calculate_llm_average(aggregated_samples)

    return {
        "dataset": dataset,
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