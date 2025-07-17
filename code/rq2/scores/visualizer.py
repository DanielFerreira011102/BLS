import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from utils.helpers import setup_logging, save_json
from rq2.scores.impl.evaluation import TextEvaluationMetrics, BaseMetric
from rq2.scores.phase3 import ScoreVisualizer, METADATA_COLS

# Initialize logging
logger = setup_logging()


def main() -> None:
    """Generate visualizations from an existing scores_df.csv."""
    parser = argparse.ArgumentParser(description="Generate visualizations from an existing scores_df.csv file.")
    parser.add_argument("--input-file", type=str, default="scores_df.csv", help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for saving output files")
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # Check if input file exists and is readable
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return
    if not input_file.is_file():
        logger.error(f"Input path is not a file: {input_file}")
        return
    if not os.access(input_file, os.R_OK):
        logger.error(f"Input file is not readable: {input_file}")
        return
    
    # Load the dataframe
    logger.info(f"Loading scores from {input_file}")
    df = pd.read_csv(input_file)
    
    if df.empty:
        logger.error("Loaded dataframe is empty")
        return
    
    # Check for missing columns
    missing_columns = [col for col in METADATA_COLS if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}. Some plots may not generate correctly.")
    
    # Initialize a dummy evaluator with empty metrics
    evaluator = TextEvaluationMetrics(metrics=[], batch_size=128)
    
    # Initialize the visualizer
    visualizer = ScoreVisualizer(output_dir=output_dir, evaluator=evaluator)
    
    # Generate all visualizations
    logger.info("Starting visualization generation")
    visualization_results = visualizer.generate_all_visualizations(df)
    
    # Log completion
    plots = visualization_results.get("plots", [])
    logger.info(f"Generated {len(plots)} visualizations in {output_dir}")

if __name__ == "__main__":
    main()