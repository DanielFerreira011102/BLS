import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv('/data/home/djbf/storage/bls/rq2/outputs/phase3/merged/scores_df.csv')

# Get all metric columns (excluding non-metric columns)
all_columns = df.columns.tolist()
excluded_columns = ['dataset', 'sample_id', 'model', 'complexity_level', 'question', 
                    'original_text', 'variant_text', 'temperature']
METRIC_COLUMNS = [col for col in all_columns if col not in excluded_columns]

# Columns needed for the report
TEXT_COLUMNS = ['question', 'original_text', 'variant_text']
META_COLUMNS = ['dataset', 'sample_id', 'model', 'complexity_level']
OUTPUT_PATH = '/data/home/djbf/storage/bls/rq2/outputs/phase3/merged/'

def create_quantile_bins(df, metric, num_bins=5):
    """
    Create bins based on quantiles for a given metric
    Returns the bin labels and ranges
    """
    # Handle edge case with too few unique values
    unique_values = df[metric].dropna().unique()
    if len(unique_values) <= num_bins:
        bins = sorted(unique_values)
    else:
        # Create quantile-based bins
        quantiles = np.linspace(0, 1, num_bins + 1)
        bins = df[metric].quantile(quantiles).unique()
    
    # Create labels and bin ranges
    bin_labels = [f"{metric}_bin_{i+1}" for i in range(len(bins)-1)]
    bin_ranges = {}
    
    for i in range(len(bins)-1):
        label = bin_labels[i]
        bin_ranges[label] = f"{bins[i]:.3f} to {bins[i+1]:.3f}"
    
    # Add bin column to dataframe
    df[f"{metric}_bin"] = pd.cut(
        df[metric], 
        bins=bins, 
        labels=bin_labels, 
        include_lowest=True
    )
    
    return bin_labels, bin_ranges

def select_representative_samples(df, metric, bin_name):
    """
    Select 5 representative samples from a bin:
    - Minimum value sample
    - 25th percentile sample
    - Median (50th percentile) sample
    - 75th percentile sample
    - Maximum value sample
    """
    # Get all samples in this bin
    bin_df = df[df[f"{metric}_bin"] == bin_name].copy()
    
    # If bin is empty, return empty dataframe
    if len(bin_df) == 0:
        return pd.DataFrame()
    
    # If bin has 5 or fewer samples, return all of them
    if len(bin_df) <= 5:
        return bin_df
    
    # Sort by metric value
    bin_df = bin_df.sort_values(by=metric)
    
    # Calculate indices for the percentiles
    min_idx = 0
    p25_idx = int(len(bin_df) * 0.25)
    median_idx = int(len(bin_df) * 0.5)
    p75_idx = int(len(bin_df) * 0.75)
    max_idx = len(bin_df) - 1
    
    # Select the samples
    samples = pd.concat([
        bin_df.iloc[[min_idx]],     # Minimum
        bin_df.iloc[[p25_idx]],     # 25th percentile
        bin_df.iloc[[median_idx]],  # Median (50th percentile)
        bin_df.iloc[[p75_idx]],     # 75th percentile
        bin_df.iloc[[max_idx]]      # Maximum
    ])
    
    return samples

def generate_report():
    """
    Main function to generate the binning report
    """
    print(f"Dataset has {len(df)} samples")
    
    # Create a report file
    with open(f"{OUTPUT_PATH}metric_bin_report.txt", "w") as report:
        # Write descriptive statistics for all metrics
        report.write(f"{'='*80}\n")
        report.write(f"DESCRIPTIVE STATISTICS FOR ALL METRICS\n")
        report.write(f"{'='*80}\n\n")
        
        stats_df = df[METRIC_COLUMNS].describe()
        report.write(stats_df.to_string())
        report.write("\n\n")
        
        # Process each metric
        for metric in METRIC_COLUMNS:
            # Skip metrics with missing values
            if df[metric].isna().sum() > 0:
                print(f"Metric {metric} has missing values - skipping")
                continue
                
            report.write(f"\n\n{'='*80}\n")
            report.write(f"METRIC: {metric}\n")
            report.write(f"{'='*80}\n\n")
            
            # Create bins for this metric
            bin_labels, bin_ranges = create_quantile_bins(df, metric)
            
            # Process each bin
            for bin_name in bin_labels:
                # Count samples in this bin
                bin_count = df[df[f"{metric}_bin"] == bin_name].shape[0]
                bin_range = bin_ranges[bin_name]
                
                report.write(f"\n{'-'*80}\n")
                report.write(f"BIN: {bin_name} - Range: {bin_range} - Count: {bin_count} samples\n")
                report.write(f"{'-'*80}\n\n")
                
                # Select representative samples
                samples = select_representative_samples(df, metric, bin_name)
                
                # Write each sample to the report
                position_names = ["MIN", "25th", "MEDIAN", "75th", "MAX"]
                for i, (idx, row) in enumerate(samples.iterrows()):
                    position = position_names[i] if i < len(position_names) else f"Sample {i+1}"
                    report.write(f"Sample {position} - {metric}: {row[metric]:.4f}\n")
                    report.write(f"Source: {row['dataset']} - ID: {row['sample_id']}\n")
                    
                    report.write("\nQUESTION:\n")
                    report.write(f"{row['question']}\n\n")
                    
                    report.write("ORIGINAL ANSWER:\n")
                    report.write(f"{row['original_text']}\n\n")
                    
                    report.write("VARIANT ANSWER:\n")
                    report.write(f"{row['variant_text']}\n\n")
                    
                    # Add ALL metrics for this sample
                    report.write("METRICS:\n")
                    for metric_name in METRIC_COLUMNS:
                        if metric_name in row:
                            report.write(f"- {metric_name}: {row[metric_name]:.4f}\n")
                    
                    report.write(f"\n{'-'*40}\n\n")
    
    print(f"Report generated at {OUTPUT_PATH}metric_bin_report.txt")

# Run the analysis
generate_report()